"""Mid-stream failure, the trust boundary, and the request-size cap.

GR-6: a mid-stream upstream failure propagated as a raw exception, so the client
saw a truncation indistinguishable from a short completion - and any held-back,
potentially PII-bearing tail was lost with it.

SEC-1/SEC-2: no authentication and no body cap. The first is a deliberate
deployment posture that was never written down or enforceable; the second was
simply missing.
"""

from __future__ import annotations

import json

import httpx
import pytest
from llm_guardrail.app import create_app
from llm_guardrail.config import Config
from llm_guardrail.redactor import StreamRedactor
from llm_guardrail.stream import UPSTREAM_FAILED_MESSAGE, redact_sse_stream

pytestmark = pytest.mark.asyncio

UPSTREAM = "http://provider.test/v1/messages"


def delta(text: str) -> bytes:
    payload = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}
    return f"event: content_block_delta\ndata: {json.dumps(payload)}\n\n".encode()


# --------------------------------------------------------------------------
# GR-6 - mid-stream failure
# --------------------------------------------------------------------------


async def _collect(upstream) -> bytes:
    out = b""
    async for chunk in redact_sse_stream(upstream, StreamRedactor()):
        out += chunk
    return out


async def test_a_mid_stream_failure_produces_a_terminal_error_event():
    async def upstream():
        yield delta("here is some text ")
        yield delta("and some more ")
        raise RuntimeError("upstream exploded mid-stream")

    out = await _collect(upstream())

    assert b"event: error" in out, "the client got a truncation with no error frame"
    assert UPSTREAM_FAILED_MESSAGE.encode() in out
    assert b"here is some text" in out, "text delivered before the failure should survive"


async def test_the_mid_stream_error_does_not_leak_the_exception():
    async def upstream():
        yield delta("hello ")
        raise RuntimeError("connect to internal-llm.prod.svc.cluster.local:8091 failed (key sk-abc123)")

    out = await _collect(upstream())

    for secret in (b"internal-llm.prod.svc", b"sk-abc123", b"RuntimeError", b"Traceback", b"8091"):
        assert secret not in out, secret


async def test_the_held_tail_is_flushed_before_the_error_event():
    """A response failing mid-partial-match must not silently drop its tail."""

    async def upstream():
        yield delta("call me at 123-4")
        raise RuntimeError("boom")

    out = await _collect(upstream())

    assert b"123-4" in out, "the held tail was lost when the stream failed"
    assert b"event: error" in out


async def test_a_held_secret_is_still_redacted_when_the_stream_fails():
    """The flush on the error path must go through the redactor, not around it."""

    async def upstream():
        yield delta("my ssn is 123-45-6789")
        raise RuntimeError("boom")

    out = await _collect(upstream())

    assert b"123-45-6789" not in out, "the failure path released PII"
    assert b"[REDACTED]" in out


async def test_the_generator_does_not_raise_out_of_the_relay():
    """The ASGI server aborting the response is what made the truncation
    indistinguishable from a normal short completion."""

    async def upstream():
        yield delta("x")
        raise RuntimeError("boom")

    # Completing without an exception is the assertion.
    await _collect(upstream())


# --------------------------------------------------------------------------
# Trust boundary
# --------------------------------------------------------------------------


def _config(**overrides) -> Config:
    base = {"upstream_url": UPSTREAM, "request_timeout_seconds": 5.0, "read_timeout_seconds": 30.0}
    return Config(**{**base, **overrides})


async def _client_for(config: Config) -> httpx.AsyncClient:
    from mock_provider.app import create_app as create_provider

    app = create_app(config)
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_provider()))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://guardrail.test")
    client._app = app
    client._upstream = upstream
    return client


async def test_an_unauthenticated_proxy_states_its_posture_in_the_log(caplog):
    config = _config()
    assert config.requires_authentication is False
    description = config.describe_trust_boundary()
    assert "UNAUTHENTICATED" in description
    assert "LLM_GUARDRAIL_TOKENS" in description, "the log must say how to close it"


def test_configuring_tokens_switches_the_posture():
    config = _config(tokens=frozenset({"secret-token"}), trusted_network=False)
    assert config.requires_authentication
    assert "authenticated" in config.describe_trust_boundary()
    assert "secret-token" not in config.describe_trust_boundary(), "the log must not print the token"


def test_tokens_are_parsed_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_GUARDRAIL_TOKENS", " tok-a , tok-b ,, ")
    config = Config.from_env()
    assert config.tokens == frozenset({"tok-a", "tok-b"})
    assert config.requires_authentication
    assert config.trusted_network is False


def test_no_tokens_in_the_environment_means_the_open_posture(monkeypatch):
    monkeypatch.delenv("LLM_GUARDRAIL_TOKENS", raising=False)
    config = Config.from_env()
    assert config.tokens == frozenset()
    assert config.trusted_network is True


REJECTED = [
    pytest.param({}, id="no header"),
    pytest.param({"Authorization": "Bearer wrong"}, id="wrong token"),
    pytest.param({"Authorization": "Basic c2VjcmV0"}, id="wrong scheme"),
    pytest.param({"Authorization": "secret-token"}, id="no scheme"),
    pytest.param({"Authorization": "Bearer "}, id="empty token"),
    pytest.param({"Authorization": "Bearer secret-token-x"}, id="suffixed"),
    pytest.param({"Authorization": "Bearer SECRET-TOKEN"}, id="case changed"),
]


@pytest.mark.parametrize("headers", REJECTED)
async def test_a_token_protected_proxy_rejects_bad_credentials(headers):
    config = _config(tokens=frozenset({"secret-token"}), trusted_network=False)
    client = await _client_for(config)
    async with client._upstream as upstream, client as http:
        async with client._app.router.lifespan_context(client._app):
            client._app.state.http_client = upstream
            response = await http.post(
                "/v1/messages", json={"model": "m", "stream": False, "text": "hi"}, headers=headers
            )
    assert response.status_code == 401, headers
    assert response.headers["www-authenticate"].startswith("Bearer")


async def test_a_token_protected_proxy_accepts_a_good_credential():
    config = _config(tokens=frozenset({"secret-token"}), trusted_network=False)
    client = await _client_for(config)
    async with client._upstream as upstream, client as http:
        async with client._app.router.lifespan_context(client._app):
            client._app.state.http_client = upstream
            response = await http.post(
                "/v1/messages",
                json={"model": "m", "stream": False, "text": "hi"},
                headers={"Authorization": "Bearer secret-token"},
            )
    assert response.status_code == 200


async def test_an_unauthenticated_proxy_still_serves(guardrail):
    """The open posture must keep working; it is a supported deployment."""
    response = await guardrail.post("/v1/messages", json={"model": "m", "stream": False, "text": "hi"})
    assert response.status_code == 200


# --------------------------------------------------------------------------
# SEC-2 - request size cap
# --------------------------------------------------------------------------


async def test_an_oversized_body_is_refused(guardrail, config):
    payload = {"model": "m", "stream": False, "text": "x" * (config.max_body_bytes + 4096)}
    response = await guardrail.post("/v1/messages", json=payload)
    assert response.status_code == 413
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_a_body_under_the_cap_is_accepted(guardrail, config):
    payload = {"model": "m", "stream": False, "text": "x" * 1024}
    assert len(json.dumps(payload)) < config.max_body_bytes
    response = await guardrail.post("/v1/messages", json=payload)
    assert response.status_code == 200


def test_the_cap_is_configurable(monkeypatch):
    monkeypatch.setenv("LLM_GUARDRAIL_MAX_BODY_BYTES", "1024")
    assert Config.from_env().max_body_bytes == 1024
