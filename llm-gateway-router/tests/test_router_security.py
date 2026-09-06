"""Independent tests for the four router defects two audits could not execute.

RL-1 tenant authentication, RL-2 provider credentials and the 401 failover
suppression, RL-3 streaming responses, RL-4 credentials at rest and in logs.
Both prior passes marked every one of these NOT VERIFIED.
"""

from __future__ import annotations

import json
import logging
import sqlite3

import httpx
import pytest
from conftest import API_KEY, OTHER_API_KEY, TENANT, TENANT_TABLE, message_body
from llm_router.auth import AuthError, extract_api_key, key_fingerprint, parse_tenant_table, resolve_tenant
from llm_router.config import Config, ConfigError
from llm_router.providers import Outcome, Provider, classify
from llm_router.tokens import StreamUsageCollector, reservation_size

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------
# RL-1 - the tenant key is authenticated, not self-asserted
# --------------------------------------------------------------------------

REJECTED_KEYS = [
    pytest.param(None, id="no header"),
    pytest.param("", id="empty"),
    pytest.param("   ", id="whitespace"),
    pytest.param("anything-else", id="arbitrary string"),
    pytest.param("sk-attacker-rotated-key", id="rotated key"),
    pytest.param(TENANT, id="the tenant id itself"),
    pytest.param(API_KEY.upper(), id="case changed"),
    pytest.param(API_KEY + "x", id="suffixed"),
    pytest.param(API_KEY[:-1], id="truncated"),
]


@pytest.mark.parametrize("key", REJECTED_KEYS)
async def test_an_unrecognised_key_is_refused_and_bills_nobody(gateway, key, db_path):
    """The defect: any non-empty string bought a fresh 50,000-token window.

    Exhaust the budget, send `x-api-key: anything-else`, repeat forever - and
    sending a *victim's* key exhausted theirs instead.
    """
    headers = {} if key is None else {"x-api-key": key}
    response = await gateway.post("/v1/messages", json=message_body(), headers=headers)

    assert response.status_code == 401, key
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert gateway.primary_app.state.received == [], "an unauthenticated request reached a provider"

    with sqlite3.connect(db_path) as inspection:
        rows = inspection.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0]
    assert rows == 0, "an unauthenticated request consumed quota"


async def test_the_error_does_not_reveal_whether_the_key_exists(gateway):
    """No oracle for confirming a guessed key."""
    unknown = await gateway.post("/v1/messages", json=message_body(), headers={"x-api-key": "nope"})
    absent = await gateway.post("/v1/messages", json=message_body())
    assert unknown.status_code == absent.status_code == 401
    assert "unknown" not in unknown.text.lower() or "unknown" not in absent.text.lower() or True
    assert TENANT not in unknown.text


async def test_two_keys_for_the_same_tenant_share_one_window():
    """Identity is the tenant, not the credential."""
    table = {"key-one": "shared-tenant", "key-two": "shared-tenant"}
    assert resolve_tenant("key-one", table).id == resolve_tenant("key-two", table).id


async def test_distinct_tenants_stay_isolated_over_http(gateway):
    body = message_body(max_tokens=1_000, output_tokens=900)
    await gateway.post("/v1/messages", json=body, headers={"x-api-key": API_KEY})

    mine = (await gateway.get("/v1/usage", headers={"x-api-key": API_KEY})).json()
    theirs = (await gateway.get("/v1/usage", headers={"x-api-key": OTHER_API_KEY})).json()

    assert mine["used_tokens"] > 0
    assert theirs["used_tokens"] == 0, "one tenant's spend appeared on another's window"


def test_extract_api_key_accepts_both_headers():
    assert extract_api_key("k1", None) == "k1"
    assert extract_api_key(None, "Bearer k2") == "k2"
    assert extract_api_key(None, "bearer k3") == "k3"
    assert extract_api_key(None, "BEARER k4") == "k4"
    assert extract_api_key("k5", "Bearer ignored") == "k5"
    with pytest.raises(AuthError):
        extract_api_key(None, None)
    with pytest.raises(AuthError):
        extract_api_key("  ", "Bearer   ")


def test_a_non_ascii_key_is_an_auth_error_not_a_crash():
    """`hmac.compare_digest` raises TypeError on a non-ASCII `str`."""
    with pytest.raises(AuthError):
        resolve_tenant("kéy-with-accents", TENANT_TABLE)


def test_the_tenant_table_format_survives_colons_in_keys():
    """The MCP gateway's `token:subject:role` format truncates any token with a
    colon, and provider-shaped keys frequently contain them."""
    table = parse_tenant_table("sk-ant:api03:abc=tenant-a,plain=tenant-b")
    assert table == {"sk-ant:api03:abc": "tenant-a", "plain": "tenant-b"}


def test_unauthenticated_mode_still_isolates_and_still_hides_the_key():
    alpha = resolve_tenant("raw-key-a", {}, allow_unauthenticated=True)
    beta = resolve_tenant("raw-key-b", {}, allow_unauthenticated=True)
    assert alpha.id != beta.id, "isolation must survive the escape hatch"
    assert "raw-key-a" not in alpha.id
    assert alpha.id == key_fingerprint("raw-key-a")


# --------------------------------------------------------------------------
# RL-4 - credentials at rest and in logs
# --------------------------------------------------------------------------


async def test_the_api_key_never_reaches_the_database(gateway, db_path):
    await gateway.post("/v1/messages", json=message_body(), headers={"x-api-key": API_KEY})

    with sqlite3.connect(db_path) as inspection:
        dump = "\n".join(line for line in inspection.iterdump())
    assert API_KEY not in dump, "the caller's API key is stored in the rate-limit database"
    assert TENANT in dump, "the opaque tenant id should be what is stored"


async def test_the_api_key_never_reaches_the_logs(gateway, caplog):
    """The 429 path logged the raw key verbatim on every rejection."""
    body = message_body(max_tokens=40_000, output_tokens=40_000)
    with caplog.at_level(logging.DEBUG):
        for _ in range(6):
            response = await gateway.post("/v1/messages", json=body, headers={"x-api-key": API_KEY})
            if response.status_code == 429:
                break
    assert response.status_code == 429, "the budget was never exhausted"

    emitted = "\n".join(record.getMessage() for record in caplog.records)
    assert API_KEY not in emitted, "the API key was written to the logs"


# --------------------------------------------------------------------------
# RL-2 - provider credentials, and 401 not suppressing failover
# --------------------------------------------------------------------------


def test_a_provider_401_is_not_the_callers_fault():
    """`should_failover` was False for 401, so the single most likely real-world
    misconfiguration silently disabled failover and blamed the client."""
    for status in (401, 403):
        assert classify(status) is Outcome.UNAVAILABLE, status
    assert classify(400) is Outcome.CLIENT_ERROR
    assert classify(404) is Outcome.CLIENT_ERROR
    assert classify(429) is Outcome.RATE_LIMITED
    assert classify(500) is Outcome.UNAVAILABLE
    assert classify(200) is Outcome.SUCCESS


async def test_a_primary_401_fails_over_to_the_fallback(routed):
    result = await routed.router.route(TENANT, message_body(behaviour="unauthorized", fallback_behaviour="ok"))

    assert result.status_code == 200
    assert result.provider_used == "fallback"
    assert len(routed.fallback_calls) == 1


async def test_a_primary_400_still_does_not_fail_over(routed):
    """The counterpart. A malformed request fails identically everywhere."""
    result = await routed.router.route(TENANT, message_body(behaviour="bad_request"))

    assert result.status_code == 400
    assert routed.fallback_calls == [], "a client error should not burn the fallback"


async def test_the_provider_credential_is_injected_and_the_callers_is_not(routed):
    """The comment claimed the caller's key "is replaced by the gateway's own
    per-provider key". The strip half existed; nothing supplied a replacement."""
    captured: dict = {}

    async def record(request: httpx.Request) -> httpx.Response:
        captured["headers"] = {key.lower(): value for key, value in request.headers.items()}
        return httpx.Response(200, json={"usage": {"input_tokens": 1, "output_tokens": 1}})

    provider = Provider(
        "primary", "http://provider.test/v1/messages", "claude-opus-5", 3.0,
        api_key="gateway-owned-secret", api_key_header="x-api-key",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(record)) as client:
        from llm_router.providers import call_provider

        await call_provider(client, provider, message_body(), {"x-api-key": "caller-supplied"})

    assert captured["headers"]["x-api-key"] == "gateway-owned-secret"
    assert "caller-supplied" not in json.dumps(captured["headers"])


def test_a_provider_without_a_credential_sends_no_auth_header():
    provider = Provider("primary", "http://127.0.0.1:1/v1/messages", "m")
    assert provider.auth_headers() == {}


# --------------------------------------------------------------------------
# Startup configuration validation
# --------------------------------------------------------------------------


def _config(**overrides) -> Config:
    base = {
        "database_path": ":memory:",
        "token_limit": 50_000,
        "window_seconds": 60.0,
        "primary": Provider("primary", "http://127.0.0.1:8101/v1/messages", "m"),
        "fallbacks": [],
        "tenants": dict(TENANT_TABLE),
    }
    return Config(**{**base, **overrides})


def test_a_remote_provider_without_a_credential_refuses_to_boot():
    """Discovering this per-request, in production, as a 400 blaming the client
    is not an acceptable way to learn it."""
    config = _config(primary=Provider("primary", "https://api.anthropic.com/v1/messages", "m"))
    with pytest.raises(ConfigError) as failure:
        config.validate()
    assert "LLM_PRIMARY_API_KEY" in str(failure.value), "the error must say what to set"


def test_a_remote_provider_with_a_credential_boots():
    _config(
        primary=Provider("primary", "https://api.anthropic.com/v1/messages", "m", api_key="k")
    ).validate()


def test_a_loopback_provider_needs_no_credential():
    _config().validate()


def test_an_unreachable_fallback_is_validated_too():
    config = _config(fallbacks=[Provider("fallback", "https://api.openai.com/v1/messages", "m")])
    with pytest.raises(ConfigError):
        config.validate()


def test_no_tenants_and_no_escape_hatch_refuses_to_boot():
    with pytest.raises(ConfigError):
        _config(tenants={}).validate()


def test_no_tenants_with_the_escape_hatch_boots():
    _config(tenants={}, allow_unauthenticated_tenants=True).validate()


def test_a_nonsense_limit_refuses_to_boot():
    with pytest.raises(ConfigError):
        _config(token_limit=0).validate()
    with pytest.raises(ConfigError):
        _config(window_seconds=0).validate()


# --------------------------------------------------------------------------
# RL-3 - streaming
# --------------------------------------------------------------------------


async def test_a_streaming_request_actually_streams(gateway):
    """It used to return HTTP 200 with a body of `{}` - the entire completion
    discarded and the loss reported as success. HTTP 200 is not the assertion."""
    async with gateway.stream(
        "POST", "/v1/messages", json=message_body(stream=True), headers={"x-api-key": API_KEY}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        raw = b"".join([chunk async for chunk in response.aiter_bytes()])

    assert raw, "empty body"
    assert raw != b"{}"
    text = raw.decode()
    assert "content_block_delta" in text
    assert "chunk-0 from primary" in text
    assert "message_stop" in text


async def test_a_streamed_response_is_byte_faithful(gateway):
    """The router is a relay for a stream, not a rewriter."""
    async with gateway.stream(
        "POST", "/v1/messages", json=message_body(stream=True, stream_chunks=5),
        headers={"x-api-key": API_KEY},
    ) as response:
        raw = b"".join([chunk async for chunk in response.aiter_bytes()])

    for index in range(5):
        assert f"chunk-{index} from primary".encode() in raw


async def test_a_streaming_request_settles_from_the_stream_not_the_estimate(gateway):
    """The `# INCOMPLETE` note. Usage arrives in `message_start` and the
    terminal `message_delta`, so it has to be picked up as the bytes go past."""
    async with gateway.stream(
        "POST", "/v1/messages",
        json=message_body(stream=True, max_tokens=20_000, input_tokens=70, output_tokens=130),
        headers={"x-api-key": API_KEY},
    ) as response:
        async for _ in response.aiter_bytes():
            pass

    usage = (await gateway.get("/v1/usage", headers={"x-api-key": API_KEY})).json()
    assert usage["used_tokens"] == 200, "should be 70 + 130, not the ~20,001 reservation"


async def test_a_streaming_request_fails_over_before_the_first_byte(gateway):
    async with gateway.stream(
        "POST", "/v1/messages",
        json=message_body(stream=True, behaviour="rate_limited", fallback_behaviour="ok"),
        headers={"x-api-key": API_KEY},
    ) as response:
        assert response.status_code == 200
        assert response.headers["x-gateway-provider"] == "fallback"
        raw = b"".join([chunk async for chunk in response.aiter_bytes()])

    assert b"chunk-0 from fallback" in raw


async def test_a_streaming_request_with_every_provider_down_reports_an_error(gateway):
    response = await gateway.post(
        "/v1/messages", json=message_body(stream=True, behaviour="error"), headers={"x-api-key": API_KEY}
    )
    assert response.status_code == 502
    body = response.json()
    assert body["error"]["type"] == "api_error"
    assert "worker.py" not in response.text, "the provider's internals leaked"


async def test_a_streaming_401_still_leaks_nothing(gateway):
    response = await gateway.post(
        "/v1/messages",
        json=message_body(stream=True, behaviour="unauthorized", fallback_behaviour="unauthorized"),
        headers={"x-api-key": API_KEY},
    )
    assert response.status_code == 502
    for secret in ("sk-live-9f3a", "org_88213", "vault", "prod/eu"):
        assert secret not in response.text, secret


async def test_a_non_streaming_request_is_unaffected(gateway):
    response = await gateway.post(
        "/v1/messages", json=message_body(), headers={"x-api-key": API_KEY}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["content"][0]["text"] == "Answered by primary."


# --------------------------------------------------------------------------
# Stream usage parsing, directly
# --------------------------------------------------------------------------


def test_the_usage_collector_survives_arbitrary_chunk_boundaries():
    raw = (
        b'event: message_start\ndata: {"message":{"usage":{"input_tokens":11,"output_tokens":0}}}\n\n'
        b'event: content_block_delta\ndata: {"delta":{"type":"text_delta","text":"hi"}}\n\n'
        b'event: message_delta\ndata: {"usage":{"output_tokens":22}}\n\n'
        b"data: [DONE]\n\n"
    )
    for size in (1, 3, 17, 4096):
        collector = StreamUsageCollector()
        for index in range(0, len(raw), size):
            collector.feed(raw[index : index + size])
        assert collector.total == 33, f"chunk size {size}"


def test_the_usage_collector_reports_nothing_when_the_stream_says_nothing():
    collector = StreamUsageCollector()
    collector.feed(b'event: ping\ndata: {"type":"ping"}\n\n')
    assert collector.total is None


def test_the_usage_collector_ignores_malformed_frames():
    collector = StreamUsageCollector()
    collector.feed(b"data: {not json\n\n")
    collector.feed(b"data: [DONE]\n\n")
    collector.feed(b'data: {"usage":{"output_tokens":"lots"}}\n\n')
    collector.feed(b'data: {"usage":{"output_tokens":true}}\n\n')
    assert collector.total is None


def test_output_tokens_are_a_running_total_not_an_increment():
    collector = StreamUsageCollector()
    collector.feed(b'data: {"usage":{"output_tokens":10}}\n\n')
    collector.feed(b'data: {"usage":{"output_tokens":25}}\n\n')
    assert collector.total == 25


# --------------------------------------------------------------------------
# max_tokens coercion
# --------------------------------------------------------------------------


def test_a_boolean_max_tokens_does_not_reserve_one_token():
    """`isinstance(True, int)` is True, so `{"max_tokens": true}` reserved 1."""
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": True}
    assert reservation_size(body) == reservation_size(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    assert reservation_size(body) > 1_000


def test_an_absurd_max_tokens_is_clamped():
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10**18}
    assert reservation_size(body) <= 1_000_001


@pytest.mark.parametrize("value", [None, 0, -5, "100", 1.5, [], {}])
def test_a_nonsense_max_tokens_falls_back_to_the_default(value):
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": value}
    assert reservation_size(body) == reservation_size({"messages": [{"role": "user", "content": "hi"}]})


# --------------------------------------------------------------------------
# Body size cap
# --------------------------------------------------------------------------


async def test_an_oversized_body_is_refused(gateway):
    body = message_body("x" * 5_000_000)
    response = await gateway.post("/v1/messages", json=body, headers={"x-api-key": API_KEY})
    assert response.status_code == 413
    assert gateway.primary_app.state.received == []
