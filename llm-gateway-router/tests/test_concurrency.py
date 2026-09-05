"""Concurrency: the limit must hold when requests arrive simultaneously.

This is where a limiter that looks correct in isolation usually breaks. The
admission check is read-then-write, and if those two steps are not atomic, two
requests can both read the same usage total and both reserve against it -
admitting more than the limit allows. That bug never shows up in a sequential
test.

Within one `Store` the atomicity comes from its `asyncio.Lock`; the database
transaction mode is what protects the *cross-connection* case, and is verified
separately in `test_transactions.py`, which forces the interleaving these tests
cannot reach.
"""

from __future__ import annotations

import asyncio

import pytest

from conftest import TENANT, message_body

pytestmark = pytest.mark.asyncio


async def test_concurrent_requests_cannot_exceed_the_limit(limiter):
    """50 requests of 2,000 tokens against a 50,000 limit: exactly 25 fit.

    Remove the serialisation in `Store.execute` and this over-admits, because
    every coroutine reads the usage total before any of them has written a
    reservation.
    """
    decisions = await asyncio.gather(*(limiter.check_and_reserve(TENANT, 2_000) for _ in range(50)))

    admitted = [decision for decision in decisions if decision.allowed]
    assert len(admitted) == 25, f"admitted {len(admitted)} of 50; the limit was not enforced atomically"
    assert await limiter.usage(TENANT) == 50_000


async def test_the_total_never_overshoots_under_uneven_load(limiter):
    """Mixed sizes arriving at once must still sum to at most the limit."""
    sizes = [1_000, 5_000, 20_000, 7_500, 12_500, 30_000, 2_500, 15_000]
    decisions = await asyncio.gather(*(limiter.check_and_reserve(TENANT, size) for size in sizes))

    admitted_total = sum(
        decision.reservation.estimated_tokens for decision in decisions if decision.allowed
    )
    assert admitted_total <= 50_000
    assert await limiter.usage(TENANT) == admitted_total


async def test_concurrent_tenants_do_not_interfere(limiter):
    """Isolation must survive interleaving, not just sequential calls."""
    tenants = [f"sk-tenant-{index}" for index in range(10)]

    async def spend(tenant: str) -> int:
        decisions = await asyncio.gather(*(limiter.check_and_reserve(tenant, 10_000) for _ in range(10)))
        return sum(1 for decision in decisions if decision.allowed)

    admitted = await asyncio.gather(*(spend(tenant) for tenant in tenants))

    # Each tenant independently gets 5 of its 10 requests (50,000 / 10,000).
    assert admitted == [5] * 10
    for tenant in tenants:
        assert await limiter.usage(tenant) == 50_000


async def test_concurrent_settles_do_not_corrupt_the_total(limiter):
    """Reserve many, settle them all at once, and check the arithmetic."""
    decisions = await asyncio.gather(*(limiter.check_and_reserve(TENANT, 1_000) for _ in range(30)))
    reservations = [decision.reservation for decision in decisions if decision.allowed]
    assert len(reservations) == 30

    await asyncio.gather(*(limiter.settle(reservation, 100) for reservation in reservations))

    assert await limiter.usage(TENANT) == 3_000
    assert await limiter.row_count() == 30


async def test_concurrent_release_and_reserve_interleave_safely(limiter):
    """Freed quota must become available without ever double-counting."""
    first = await asyncio.gather(*(limiter.check_and_reserve(TENANT, 10_000) for _ in range(5)))
    reservations = [decision.reservation for decision in first if decision.allowed]
    assert len(reservations) == 5
    assert (await limiter.check_and_reserve(TENANT, 1)).allowed is False

    # Release three while five more requests contend for the freed space.
    releases = [limiter.release(reservation) for reservation in reservations[:3]]
    reserves = [limiter.check_and_reserve(TENANT, 10_000) for _ in range(5)]
    results = await asyncio.gather(*releases, *reserves)

    newly_admitted = [item for item in results[3:] if item and item.allowed]
    assert len(newly_admitted) <= 3, "more admitted than the released quota allowed"
    assert await limiter.usage(TENANT) <= 50_000


async def test_the_event_loop_is_not_blocked_by_database_work(limiter):
    """SQLite calls are blocking and must run off the event loop.

    If they ran inline, a heartbeat coroutine would be starved while the
    database works - and under real load that stalls every other in-flight
    request, exactly when the gateway is busiest.
    """
    ticks = 0
    running = True

    async def heartbeat():
        nonlocal ticks
        while running:
            ticks += 1
            await asyncio.sleep(0)

    task = asyncio.create_task(heartbeat())
    await asyncio.gather(*(limiter.check_and_reserve(f"sk-{index}", 100) for index in range(100)))
    running = False
    await task

    assert ticks > 50, f"heartbeat only ran {ticks} times; the loop was blocked on SQLite"


async def test_concurrent_routing_respects_the_limit_end_to_end(routed):
    """The same guarantee through the full router, not just the limiter.

    Each request reserves prompt + max_tokens = ~1,004 tokens, so a 50,000
    budget admits about 49 of 80.
    """
    bodies = [message_body("hi", max_tokens=1_000) for _ in range(80)]
    results = await asyncio.gather(*(routed.router.route(TENANT, body) for body in bodies))

    served = [result for result in results if result.status_code == 200]
    limited = [result for result in results if result.status_code == 429]

    assert len(served) + len(limited) == 80
    assert len(limited) > 0, "nothing was rate limited; the budget should not have covered 80 requests"
    # Settled usage reflects what the provider reported (150 each), not the
    # estimates, so the total lands well under the ceiling.
    assert await routed.limiter.usage(TENANT) <= 50_000
    assert len(routed.primary_calls) == len(served)


class TestAcrossConnections:
    """Two connections to the same file - the multi-worker deployment shape.

    The `asyncio.Lock` inside `Store` only serialises callers sharing one
    `Store` instance. Run the gateway as several uvicorn workers against one
    SQLite file and each has its own connection, so the lock provides nothing
    and the atomicity has to come from the database.

    These cover the accounting across connections. They do **not** by themselves
    prove the transaction mode matters - `asyncio.to_thread` dispatches quickly
    enough that the two checks finish one after the other, so the dangerous
    window never opens. `test_transactions.py` forces it open.
    """

    async def test_two_connections_cannot_both_reserve_the_same_quota(self, db_path, clock):
        from llm_router.ratelimit import RateLimiter
        from llm_router.store import Store

        stores = [Store(db_path) for _ in range(2)]
        try:
            limiters = [RateLimiter(store, limit=50_000, window_seconds=60.0, clock=clock) for store in stores]

            # Each connection sees 40,000 already used and asks for 20,000. Only
            # one can be admitted; a deferred transaction lets both through.
            await limiters[0].check_and_reserve(TENANT, 40_000)
            decisions = await asyncio.gather(
                limiters[0].check_and_reserve(TENANT, 20_000),
                limiters[1].check_and_reserve(TENANT, 20_000),
            )

            assert sum(1 for decision in decisions if decision.allowed) == 0
            assert await limiters[0].usage(TENANT) == 40_000
        finally:
            for store in stores:
                store.close()

    async def test_interleaved_reservations_across_connections_stay_bounded(self, db_path, clock):
        from llm_router.ratelimit import RateLimiter
        from llm_router.store import Store

        stores = [Store(db_path) for _ in range(4)]
        try:
            limiters = [RateLimiter(store, limit=50_000, window_seconds=60.0, clock=clock) for store in stores]

            decisions = await asyncio.gather(
                *(limiters[index % 4].check_and_reserve(TENANT, 5_000) for index in range(40))
            )

            admitted = [decision for decision in decisions if decision.allowed]
            assert len(admitted) == 10, f"admitted {len(admitted)}; the limit leaked across connections"
            assert await limiters[0].usage(TENANT) == 50_000
        finally:
            for store in stores:
                store.close()
