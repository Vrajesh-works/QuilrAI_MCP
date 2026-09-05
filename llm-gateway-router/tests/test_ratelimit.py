"""The sliding window: accounting, eviction, and per-tenant isolation."""

from __future__ import annotations

import pytest

from conftest import OTHER_TENANT, TENANT

pytestmark = pytest.mark.asyncio


async def test_requests_under_the_limit_are_admitted(limiter):
    decision = await limiter.check_and_reserve(TENANT, 1_000)

    assert decision.allowed is True
    assert decision.used_tokens == 1_000
    assert decision.remaining == 49_000
    assert decision.reservation is not None


async def test_the_limit_is_enforced(limiter):
    await limiter.check_and_reserve(TENANT, 49_500)
    decision = await limiter.check_and_reserve(TENANT, 1_000)

    assert decision.allowed is False
    assert decision.used_tokens == 49_500


async def test_a_request_landing_exactly_on_the_limit_is_admitted(limiter):
    """The check is `used + requested > limit`, so exact fit is allowed."""
    first = await limiter.check_and_reserve(TENANT, 49_000)
    exact = await limiter.check_and_reserve(TENANT, 1_000)

    assert first.allowed and exact.allowed
    assert exact.used_tokens == 50_000
    assert (await limiter.check_and_reserve(TENANT, 1)).allowed is False


async def test_tenants_are_isolated(limiter):
    await limiter.check_and_reserve(TENANT, 50_000)

    assert (await limiter.check_and_reserve(TENANT, 1)).allowed is False
    assert (await limiter.check_and_reserve(OTHER_TENANT, 50_000)).allowed is True


class TestSlidingWindow:
    """The behaviour that separates a sliding window from a fixed one."""

    async def test_quota_frees_up_gradually_as_events_age_out(self, limiter, clock):
        await limiter.check_and_reserve(TENANT, 30_000)
        clock.advance(30)
        await limiter.check_and_reserve(TENANT, 20_000)

        assert (await limiter.check_and_reserve(TENANT, 1)).allowed is False

        # 31s later the first event has left the window; the second has not.
        clock.advance(31)
        assert await limiter.usage(TENANT) == 20_000
        assert (await limiter.check_and_reserve(TENANT, 30_000)).allowed is True

    async def test_a_fixed_window_burst_is_rejected(self, limiter, clock):
        """The exact abuse a fixed window permits.

        Under "50,000 per calendar minute", spending the full quota at 11:59:59
        and again at 12:00:00 is legal - 100,000 tokens in one second. Here the
        second burst is refused because the first is still inside the window.
        """
        assert (await limiter.check_and_reserve(TENANT, 50_000)).allowed is True

        clock.advance(1)
        assert (await limiter.check_and_reserve(TENANT, 50_000)).allowed is False

        # Only once the first burst fully ages out does the second fit.
        clock.advance(60)
        assert (await limiter.check_and_reserve(TENANT, 50_000)).allowed is True

    async def test_usage_is_zero_after_the_window_passes(self, limiter, clock):
        await limiter.check_and_reserve(TENANT, 50_000)
        clock.advance(61)

        assert await limiter.usage(TENANT) == 0


class TestEviction:
    """Expired state must actually be deleted, not merely excluded from sums."""

    async def test_expired_rows_are_deleted_from_disk(self, limiter, clock):
        for _ in range(20):
            await limiter.check_and_reserve(TENANT, 100)
        assert await limiter.row_count() == 20

        clock.advance(61)
        await limiter.check_and_reserve(TENANT, 100)

        # The 20 old rows are gone; only the new one remains.
        assert await limiter.row_count() == 1

    async def test_table_stays_bounded_over_a_long_run(self, limiter, clock):
        """Without eviction this table grows forever, and the sum query with it."""
        for _ in range(300):
            await limiter.check_and_reserve(TENANT, 10)
            clock.advance(1)

        # At one event per second in a 60s window, only ~60 can be live.
        assert await limiter.row_count() <= 61

    async def test_eviction_does_not_touch_live_rows(self, limiter, clock):
        await limiter.check_and_reserve(TENANT, 5_000)
        clock.advance(59)
        await limiter.check_and_reserve(TENANT, 5_000)

        assert await limiter.usage(TENANT) == 10_000


class TestReserveAndSettle:
    """Quota is held on an estimate and corrected against real usage."""

    async def test_settling_lower_returns_the_difference(self, limiter):
        decision = await limiter.check_and_reserve(TENANT, 10_000)
        assert await limiter.usage(TENANT) == 10_000

        await limiter.settle(decision.reservation, 250)

        assert await limiter.usage(TENANT) == 250

    async def test_settling_higher_charges_the_overrun(self, limiter):
        decision = await limiter.check_and_reserve(TENANT, 1_000)
        await limiter.settle(decision.reservation, 4_000)

        assert await limiter.usage(TENANT) == 4_000

    async def test_releasing_returns_the_whole_reservation(self, limiter):
        decision = await limiter.check_and_reserve(TENANT, 10_000)
        await limiter.release(decision.reservation)

        assert await limiter.usage(TENANT) == 0
        assert await limiter.row_count() == 0

    async def test_reservations_block_concurrent_requests_before_settling(self, limiter):
        """The point of reserving: in-flight requests count against the limit.

        A limiter that only counts completed requests admits any number of
        concurrent ones, which is precisely when the limit matters.
        """
        await limiter.check_and_reserve(TENANT, 30_000)
        second = await limiter.check_and_reserve(TENANT, 30_000)

        assert second.allowed is False

    async def test_settling_keeps_the_original_timestamp(self, limiter, clock):
        """An event ages out from when the request started, not when it finished.

        Re-stamping on settle would let a slow request hold quota for longer
        than the window.
        """
        decision = await limiter.check_and_reserve(TENANT, 10_000)
        clock.advance(50)
        await limiter.settle(decision.reservation, 10_000)

        clock.advance(11)  # 61s after the request started
        assert await limiter.usage(TENANT) == 0


class TestRetryAfter:
    """Clients should be told the minimum useful wait, not a flat 60s."""

    async def test_retry_after_points_at_the_next_expiry(self, limiter, clock):
        await limiter.check_and_reserve(TENANT, 50_000)
        clock.advance(20)

        decision = await limiter.check_and_reserve(TENANT, 1_000)

        assert decision.allowed is False
        # The blocking event expires 60s after it was written, i.e. in 40s.
        assert decision.retry_after_seconds == pytest.approx(40.0, abs=0.01)

    async def test_retry_after_accounts_for_partial_expiry(self, limiter, clock):
        """Only enough events to cover the shortfall need to expire."""
        await limiter.check_and_reserve(TENANT, 25_000)
        clock.advance(10)
        await limiter.check_and_reserve(TENANT, 25_000)

        decision = await limiter.check_and_reserve(TENANT, 1_000)

        # The first event alone frees 25,000 - far more than the 1,000 shortfall
        # - so the wait is until that one expires, 50s from now.
        assert decision.retry_after_seconds == pytest.approx(50.0, abs=0.01)

    async def test_retry_after_is_zero_when_not_limited(self, limiter):
        decision = await limiter.check_and_reserve(TENANT, 100)
        assert decision.retry_after_seconds is None


async def test_state_survives_a_restart(db_path, clock):
    """Restarting must not reset every tenant's quota.

    An in-memory limiter would hand a full fresh window to every tenant on each
    deploy, which is a trivial way to bypass the limit entirely.
    """
    from llm_router.ratelimit import RateLimiter
    from llm_router.store import Store

    first = Store(db_path)
    try:
        await RateLimiter(first, limit=50_000, window_seconds=60.0, clock=clock).check_and_reserve(TENANT, 40_000)
    finally:
        first.close()

    second = Store(db_path)
    try:
        limiter = RateLimiter(second, limit=50_000, window_seconds=60.0, clock=clock)
        assert await limiter.usage(TENANT) == 40_000
        assert (await limiter.check_and_reserve(TENANT, 20_000)).allowed is False
    finally:
        second.close()
