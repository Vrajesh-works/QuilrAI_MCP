"""Error sanitisation: nothing internal may reach the client.

The mock providers return deliberately leaky error bodies - internal deployment
names, account ids, cluster hostnames, source file paths. Every test here checks
those specific strings are absent from what the gateway returns.
"""

from __future__ import annotations

import json

import pytest

from conftest import TENANT, message_body

pytestmark = pytest.mark.asyncio

# Strings the mock providers put in their error bodies. None may escape.
LEAKS = [
    "prod-eu-3",
    "acct_9931",
    "internal-llm.prod.svc",
    "/srv/model/worker.py",
    "worker.py:412",
]


def assert_clean(payload) -> str:
    serialized = json.dumps(payload)
    for leak in LEAKS:
        assert leak not in serialized, f"internal detail {leak!r} leaked to the client"
    assert "Traceback" not in serialized
    assert "File \"" not in serialized
    return serialized


async def test_upstream_error_bodies_are_not_relayed(routed):
    result = await routed.router.route(TENANT, message_body(behaviour="error"))

    assert result.status_code == 502
    assert_clean(result.body)


async def test_upstream_429_body_is_not_relayed(routed):
    """The provider's 429 names an internal deployment and a tenant account id."""
    result = await routed.router.route(TENANT, message_body(behaviour="rate_limited"))

    assert result.status_code == 502
    assert_clean(result.body)


async def test_client_error_status_is_relayed_but_body_is_not(routed):
    """The caller needs the status code; the provider's prose is not theirs."""
    result = await routed.router.route(TENANT, message_body(behaviour="bad_request"))

    assert result.status_code == 400
    assert_clean(result.body)
    assert "max_tokens exceeds model limit" not in json.dumps(result.body)


async def test_connection_errors_do_not_leak_the_internal_address(routed):
    """A ConnectError message embeds the host and port it failed to reach."""
    broken = message_body()
    routed.router._primary = routed.router._primary.__class__(
        name="primary", url="http://nowhere.invalid/v1/messages", model="claude-opus-5", timeout_seconds=1.0
    )
    routed.router._fallbacks = [
        routed.router._fallbacks[0].__class__(
            name="fallback", url="http://also-nowhere.invalid/v1/messages", model="claude-sonnet-5", timeout_seconds=1.0
        )
    ]

    result = await routed.router.route(TENANT, broken)

    assert result.status_code == 502
    serialized = assert_clean(result.body)
    assert "nowhere.invalid" not in serialized


class TestErrorShape:
    """One consistent payload, whatever failed."""

    async def test_every_error_uses_the_same_envelope(self, routed):
        await routed.limiter.check_and_reserve(TENANT, 50_000)
        limited = await routed.router.route(TENANT, message_body())

        upstream = await routed.router.route("sk-other", message_body(behaviour="error"))

        for result in (limited, upstream):
            assert result.body["type"] == "error"
            assert isinstance(result.body["error"]["type"], str)
            assert isinstance(result.body["error"]["message"], str)

    async def test_rate_limit_errors_explain_the_budget(self, routed):
        """Gateway-owned diagnostics are safe and make the limit actionable."""
        await routed.limiter.check_and_reserve(TENANT, 50_000)
        result = await routed.router.route(TENANT, message_body())

        detail = result.body["gateway"]
        assert detail["limit_tokens"] == 50_000
        assert detail["used_tokens"] == 50_000
        assert detail["window_seconds"] == 60.0
        assert detail["retry_after_seconds"] > 0

    async def test_failure_detail_names_providers_and_outcomes(self, routed):
        """Provider names and outcome categories are our vocabulary, not upstream
        text, so returning them is safe and makes failures debuggable."""
        result = await routed.router.route(TENANT, message_body(behaviour="error"))

        attempts = result.body["gateway"]["attempts"]
        assert [attempt["provider"] for attempt in attempts] == ["primary", "fallback"]
        assert all(attempt["outcome"] == "unavailable" for attempt in attempts)
        assert all(isinstance(attempt["elapsed_ms"], int) for attempt in attempts)


class TestRateLimitHeaders:
    async def test_quota_headers_are_present_on_success(self, routed):
        result = await routed.router.route(TENANT, message_body())

        assert result.headers["x-ratelimit-limit-tokens"] == "50000"
        assert int(result.headers["x-ratelimit-remaining-tokens"]) < 50_000

    async def test_retry_after_is_an_integer_number_of_seconds(self, routed, clock):
        """RFC 9110 §10.2.3 requires an integer, and rounding up avoids
        handing back a value that is still too early."""
        await routed.limiter.check_and_reserve(TENANT, 50_000)
        clock.advance(20.4)

        result = await routed.router.route(TENANT, message_body())

        retry_after = result.headers["retry-after"]
        assert retry_after.isdigit()
        assert int(retry_after) >= 39


class TestHttpSurface:
    """The gateway's own request validation."""

    async def test_missing_api_key_is_rejected(self, gateway):
        response = await gateway.post("/v1/messages", json=message_body())

        assert response.status_code == 401
        assert response.json()["error"]["type"] == "invalid_request_error"

    async def test_bearer_and_x_api_key_are_both_accepted(self, gateway):
        for headers in ({"x-api-key": TENANT}, {"authorization": f"Bearer {TENANT}"}):
            response = await gateway.post("/v1/messages", json=message_body(), headers=headers)
            assert response.status_code == 200, headers

    async def test_malformed_json_is_rejected(self, gateway):
        response = await gateway.post(
            "/v1/messages", content=b"{not json", headers={"x-api-key": TENANT, "content-type": "application/json"}
        )

        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"

    async def test_body_without_messages_is_rejected(self, gateway):
        response = await gateway.post("/v1/messages", json={"max_tokens": 10}, headers={"x-api-key": TENANT})

        assert response.status_code == 400

    async def test_the_caller_credential_is_not_forwarded_upstream(self, gateway):
        """The gateway holds its own provider credentials; the tenant key is an
        identity for billing, not something the provider should see."""
        await gateway.post("/v1/messages", json=message_body(), headers={"x-api-key": TENANT})

        forwarded = gateway.primary_app.state.received
        assert len(forwarded) == 1
        # The body is what reaches the provider; the tenant key must not be in it.
        assert TENANT not in json.dumps(forwarded[0])

    async def test_rate_limit_returns_429_over_http(self, gateway):
        """Exhaust the budget with requests that really consume tokens.

        Note `output_tokens`: a large `max_tokens` alone will not do it, because
        a successful request settles its reservation down to what the provider
        actually reported. Only real spend depletes the window - which is the
        behaviour that makes the reserve/settle design worth having.
        """
        headers = {"x-api-key": TENANT}
        body = message_body(max_tokens=9_000, output_tokens=9_000)

        limited = None
        for _ in range(20):
            response = await gateway.post("/v1/messages", json=body, headers=headers)
            if response.status_code == 429:
                limited = response
                break

        assert limited is not None, "the budget was never exhausted"
        assert limited.headers["retry-after"].isdigit()
        assert int(limited.headers["x-ratelimit-remaining-tokens"]) < 9_000
        assert_clean(limited.json())

    async def test_a_successful_request_settles_below_its_reservation(self, gateway):
        """The reservation is a ceiling, not a charge."""
        headers = {"x-api-key": TENANT}
        await gateway.post("/v1/messages", json=message_body(max_tokens=20_000), headers=headers)

        usage = (await gateway.get("/v1/usage", headers=headers)).json()

        # Reserved ~20,001; charged the 150 the provider reported.
        assert usage["used_tokens"] == 150

    async def test_usage_endpoint_reports_the_window(self, gateway):
        headers = {"x-api-key": TENANT}
        await gateway.post("/v1/messages", json=message_body(), headers=headers)

        response = await gateway.get("/v1/usage", headers=headers)
        body = response.json()

        assert body["limit_tokens"] == 50_000
        assert body["used_tokens"] == 150
        assert body["remaining_tokens"] == 49_850

    async def test_health_endpoint(self, gateway):
        response = await gateway.get("/healthz")
        assert response.json() == {"status": "ok", "providers": ["primary", "fallback"]}
