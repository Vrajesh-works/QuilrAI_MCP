"""Failover: 429 and timeout on the primary must fall through to the backup."""

from __future__ import annotations

import time

import pytest

from conftest import TENANT, message_body
from llm_router.providers import Outcome

pytestmark = pytest.mark.asyncio


async def test_healthy_primary_serves_the_request(routed):
    result = await routed.router.route(TENANT, message_body())

    assert result.status_code == 200
    assert result.provider_used == "primary"
    assert len(routed.primary_calls) == 1
    assert routed.fallback_calls == [], "the fallback must not be touched when the primary works"


async def test_primary_429_fails_over_to_the_backup(routed):
    result = await routed.router.route(TENANT, message_body(behaviour="rate_limited", fallback_behaviour="ok"))

    assert result.status_code == 200
    assert result.provider_used == "fallback"
    assert len(routed.primary_calls) == 1
    assert len(routed.fallback_calls) == 1
    assert [attempt.outcome for attempt in result.attempts] == [Outcome.RATE_LIMITED, Outcome.SUCCESS]


async def test_primary_timeout_fails_over_to_the_backup(routed):
    """The 3s primary budget.

    The mock sleeps far longer than the primary's timeout, so the primary
    attempt is abandoned and the fallback serves the request.
    """
    started = time.perf_counter()
    result = await routed.router.route(TENANT, message_body(behaviour="slow", delay_ms=10_000, fallback_behaviour="ok", fallback_delay_ms=0))
    elapsed = time.perf_counter() - started

    assert result.status_code == 200
    assert result.provider_used == "fallback"
    assert result.attempts[0].outcome is Outcome.TIMEOUT
    # The timeout fires at ~3s; it must not wait out the provider's 10s sleep.
    assert elapsed < 8, f"the request took {elapsed:.1f}s - the timeout did not fire"


async def test_the_primary_timeout_is_three_seconds(routed):
    """Pin the configured budget, so a change to it is visible here."""
    assert routed.router.providers[0].timeout_seconds == 3.0

    started = time.perf_counter()
    await routed.router.route(TENANT, message_body(behaviour="slow", delay_ms=30_000, fallback_behaviour="ok", fallback_delay_ms=0))
    elapsed = time.perf_counter() - started

    assert 2.5 < elapsed < 6.0, f"primary attempt took {elapsed:.2f}s; expected the 3s timeout to fire"


async def test_primary_5xx_fails_over(routed):
    result = await routed.router.route(TENANT, message_body(behaviour="error", fallback_behaviour="ok"))

    assert result.status_code == 200
    assert result.provider_used == "fallback"


async def test_a_4xx_does_not_fail_over(routed):
    """A malformed request fails identically everywhere.

    Failing over would burn the backup's capacity to produce the same error
    twice, and doubles the blast radius of one bad client.
    """
    result = await routed.router.route(TENANT, message_body(behaviour="bad_request"))

    assert result.status_code == 400
    assert len(routed.primary_calls) == 1
    assert routed.fallback_calls == [], "a client error must not be retried on the backup"


async def test_both_providers_failing_returns_502(routed):
    # No per-provider override, so `behaviour` applies to both.
    result = await routed.router.route(TENANT, message_body(behaviour="error"))

    assert result.status_code == 502
    assert result.provider_used is None
    assert len(result.attempts) == 2


async def test_both_timing_out_returns_504(fast_routed):
    """Distinguish "nobody answered in time" from "everybody answered badly"."""
    routed = fast_routed
    result = await routed.router.route(TENANT, message_body(behaviour="slow", delay_ms=20_000))

    assert result.status_code == 504
    assert result.body["error"]["type"] == "timeout_error"
    assert [attempt.outcome for attempt in result.attempts] == [Outcome.TIMEOUT, Outcome.TIMEOUT]


async def test_the_provider_model_is_substituted(routed):
    """Each provider is called with its own model, not the caller's."""
    await routed.router.route(TENANT, message_body(behaviour="rate_limited", fallback_behaviour="ok"))

    assert routed.primary_calls[0]["model"] == "claude-opus-5"
    assert routed.fallback_calls[0]["model"] == "claude-sonnet-5"


async def test_the_response_names_the_provider_that_served_it(routed):
    result = await routed.router.route(TENANT, message_body(behaviour="rate_limited", fallback_behaviour="ok"))

    assert result.headers["x-gateway-provider"] == "fallback"
    assert result.headers["x-gateway-attempts"] == "2"


class TestFailoverAccounting:
    """Failover must not corrupt the quota accounting."""

    async def test_a_failed_over_request_is_billed_once(self, routed):
        await routed.router.route(TENANT, message_body(behaviour="rate_limited", fallback_behaviour="ok", max_tokens=1_000))

        # Only the fallback produced usage (50 in + 100 out), so that is the charge.
        assert await routed.limiter.usage(TENANT) == 150

    async def test_total_failure_releases_the_reservation(self, routed):
        """A request nobody served must not hold quota for the rest of the window."""
        result = await routed.router.route(TENANT, message_body(behaviour="error", max_tokens=5_000))

        assert result.status_code == 502
        assert await routed.limiter.usage(TENANT) == 0
        assert await routed.limiter.row_count() == 0

    async def test_a_timeout_is_still_charged(self, fast_routed):
        """A timed-out request may still be generating upstream, and billing.

        Releasing the quota would make timeouts free, letting a tenant drive
        unbounded load by always timing out.
        """
        routed = fast_routed
        body = message_body("hello", max_tokens=2_000, behaviour="slow", delay_ms=20_000)
        result = await routed.router.route(TENANT, body)

        assert result.status_code == 504
        assert await routed.limiter.usage(TENANT) > 0, "a timeout must not be free"

    async def test_usage_reflects_the_provider_not_the_estimate(self, routed):
        """Settling replaces a 5,000-token reservation with the real 150."""
        await routed.router.route(TENANT, message_body(max_tokens=5_000))

        assert await routed.limiter.usage(TENANT) == 150

    async def test_a_provider_reporting_no_usage_is_charged_the_estimate(self, routed):
        """Charging zero would make an unreported response a free bypass."""
        await routed.router.route(TENANT, message_body(max_tokens=3_000, omit_usage=True))

        usage = await routed.limiter.usage(TENANT)
        assert usage >= 3_000

    async def test_rate_limited_requests_never_reach_a_provider(self, routed):
        """Admission control runs before any upstream call, so a limited tenant
        costs nothing upstream."""
        await routed.limiter.check_and_reserve(TENANT, 50_000)
        result = await routed.router.route(TENANT, message_body())

        assert result.status_code == 429
        assert routed.primary_calls == []
        assert routed.fallback_calls == []
