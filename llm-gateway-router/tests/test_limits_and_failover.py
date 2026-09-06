"""Independent verification of the rate limiter and the fallback policy.

Both prior audits marked every item in this file NOT VERIFIED - the source
looked right and the repo's own tests passed, but neither was executed. These
are written from the requirement rather than from the implementation:

    50,000 tokens / minute / tenant, sliding window, persisted in SQLite
    primary -> fallback on 429, on 3000 ms timeout, and on 5xx

so they check the numbers the brief states, at their boundaries, against the
real store on a real file.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time

import httpx
import pytest
from conftest import OTHER_TENANT, TENANT, message_body
from llm_router.config import Config
from llm_router.providers import Outcome, Provider, call_provider
from llm_router.ratelimit import DEFAULT_TOKEN_LIMIT, DEFAULT_WINDOW_SECONDS, RateLimiter
from llm_router.router import Router
from llm_router.store import Store

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------
# The configured values are the ones the brief asks for
# --------------------------------------------------------------------------


def test_the_shipped_limit_is_fifty_thousand_tokens_per_minute(monkeypatch):
    """Read from the real defaults rather than assumed from the brief."""
    for variable in ("LLM_ROUTER_TOKEN_LIMIT", "LLM_ROUTER_WINDOW_SECONDS", "LLM_PRIMARY_TIMEOUT"):
        monkeypatch.delenv(variable, raising=False)
    config = Config.from_env()

    assert config.token_limit == 50_000
    assert config.window_seconds == 60.0
    assert (DEFAULT_TOKEN_LIMIT, DEFAULT_WINDOW_SECONDS) == (50_000, 60.0)
    assert config.primary.timeout_seconds == 3.0
    assert config.fallbacks[0].timeout_seconds == 10.0


def test_the_limit_is_configurable(monkeypatch):
    monkeypatch.setenv("LLM_ROUTER_TOKEN_LIMIT", "1234")
    monkeypatch.setenv("LLM_ROUTER_WINDOW_SECONDS", "7.5")
    config = Config.from_env()
    assert (config.token_limit, config.window_seconds) == (1234, 7.5)


# --------------------------------------------------------------------------
# Exact boundary
# --------------------------------------------------------------------------


async def test_a_request_one_token_under_the_limit_is_admitted(limiter):
    decision = await limiter.check_and_reserve(TENANT, 49_999)
    assert decision.allowed
    assert decision.remaining == 1


async def test_a_request_landing_exactly_on_the_limit_is_admitted(limiter):
    decision = await limiter.check_and_reserve(TENANT, 50_000)
    assert decision.allowed
    assert decision.remaining == 0


async def test_one_token_over_the_limit_is_refused(limiter):
    assert (await limiter.check_and_reserve(TENANT, 50_001)).allowed is False


async def test_the_very_next_token_after_an_exact_fit_is_refused(limiter):
    assert (await limiter.check_and_reserve(TENANT, 50_000)).allowed
    refused = await limiter.check_and_reserve(TENANT, 1)
    assert refused.allowed is False
    assert refused.used_tokens == 50_000
    assert refused.limit == 50_000


async def test_the_boundary_holds_when_assembled_from_many_requests(limiter):
    for _ in range(50):
        assert (await limiter.check_and_reserve(TENANT, 1_000)).allowed
    assert (await limiter.check_and_reserve(TENANT, 1)).allowed is False


async def test_the_brief_scenario_twenty_twenty_fifteen(limiter):
    """20k + 20k admitted, 15k refused - the worked example in the brief."""
    assert (await limiter.check_and_reserve(TENANT, 20_000)).allowed
    assert (await limiter.check_and_reserve(TENANT, 20_000)).allowed
    third = await limiter.check_and_reserve(TENANT, 15_000)
    assert third.allowed is False
    assert third.used_tokens == 40_000


# --------------------------------------------------------------------------
# Sliding-window expiry
# --------------------------------------------------------------------------


async def test_quota_returns_as_events_age_out(limiter, clock):
    assert (await limiter.check_and_reserve(TENANT, 30_000)).allowed
    clock.advance(30)
    assert (await limiter.check_and_reserve(TENANT, 20_000)).allowed
    assert (await limiter.check_and_reserve(TENANT, 1)).allowed is False

    clock.advance(31)  # the first 30k is now 61s old
    assert await limiter.usage(TENANT) == 20_000
    assert (await limiter.check_and_reserve(TENANT, 30_000)).allowed


async def test_an_event_exactly_on_the_window_edge_has_expired(limiter, clock):
    await limiter.check_and_reserve(TENANT, 50_000)
    clock.advance(60.0)
    assert await limiter.usage(TENANT) == 0, "an event exactly `window` old is outside the window"
    assert (await limiter.check_and_reserve(TENANT, 50_000)).allowed


async def test_an_event_just_inside_the_window_still_counts(limiter, clock):
    await limiter.check_and_reserve(TENANT, 50_000)
    clock.advance(59.999)
    assert await limiter.usage(TENANT) == 50_000
    assert (await limiter.check_and_reserve(TENANT, 1)).allowed is False


async def test_a_fixed_window_burst_is_rejected(limiter, clock):
    """The 11:59:59 / 12:00:00 double-spend a calendar window permits."""
    assert (await limiter.check_and_reserve(TENANT, 50_000)).allowed
    clock.advance(1.0)
    assert (await limiter.check_and_reserve(TENANT, 50_000)).allowed is False


async def test_retry_after_is_the_minimum_useful_wait(limiter, clock):
    await limiter.check_and_reserve(TENANT, 25_000)
    clock.advance(10)
    await limiter.check_and_reserve(TENANT, 25_000)

    refused = await limiter.check_and_reserve(TENANT, 25_000)
    assert refused.allowed is False
    # The first event expires 50s from now, and that alone frees enough.
    assert 49.0 < refused.retry_after_seconds <= 50.0


# --------------------------------------------------------------------------
# Tenant isolation
# --------------------------------------------------------------------------


async def test_one_tenant_cannot_consume_anothers_quota(limiter):
    assert (await limiter.check_and_reserve(TENANT, 50_000)).allowed
    assert (await limiter.check_and_reserve(TENANT, 1)).allowed is False
    assert (await limiter.check_and_reserve(OTHER_TENANT, 50_000)).allowed
    assert await limiter.usage(TENANT) == 50_000
    assert await limiter.usage(OTHER_TENANT) == 50_000


async def test_ten_tenants_each_get_a_full_window(limiter):
    tenants = [f"tenant-{index}" for index in range(10)]
    for tenant in tenants:
        assert (await limiter.check_and_reserve(tenant, 50_000)).allowed
    for tenant in tenants:
        assert (await limiter.check_and_reserve(tenant, 1)).allowed is False
        assert await limiter.usage(tenant) == 50_000


# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------


#: Every concurrent gather here is wrapped in this. A contention bug in the
#: store shows up as a stalled suite otherwise, which is the least useful
#: possible way to learn about it - `wait_for` turns it into a named failure.
CONCURRENCY_TIMEOUT = 90.0


async def test_a_hundred_concurrent_requests_cannot_exceed_the_limit(limiter):
    """Check-then-insert without a transaction would admit far more than 50."""
    decisions = await asyncio.wait_for(
        asyncio.gather(*(limiter.check_and_reserve(TENANT, 1_000) for _ in range(100))),
        timeout=CONCURRENCY_TIMEOUT,
    )
    admitted = [decision for decision in decisions if decision.allowed]

    assert len(admitted) == 50, f"{len(admitted)} admitted; the window fits exactly 50"
    assert await limiter.usage(TENANT) == 50_000


async def test_concurrent_tenants_do_not_interfere(limiter):
    tenants = [f"tenant-{index}" for index in range(10)]
    decisions = await asyncio.wait_for(
        asyncio.gather(*(limiter.check_and_reserve(tenant, 10_000) for tenant in tenants for _ in range(10))),
        timeout=CONCURRENCY_TIMEOUT,
    )
    admitted_by_tenant: dict[str, int] = {}
    for tenant, decision in zip([t for t in tenants for _ in range(10)], decisions, strict=True):
        admitted_by_tenant[tenant] = admitted_by_tenant.get(tenant, 0) + int(decision.allowed)

    assert set(admitted_by_tenant.values()) == {5}, admitted_by_tenant


async def test_concurrent_admission_across_separate_connections(db_path, clock):
    """Stands in for several gateway processes sharing one volume.

    Each limiter opens its own connection, so the in-process `asyncio.Lock`
    protects nothing and only `BEGIN IMMEDIATE` is left.
    """
    stores = [Store(db_path) for _ in range(6)]
    try:
        limiters = [RateLimiter(store, limit=50_000, window_seconds=60.0, clock=clock) for store in stores]
        decisions = await asyncio.wait_for(
            asyncio.gather(
                *(limiter.check_and_reserve(TENANT, 10_000) for limiter in limiters for _ in range(3))
            ),
            timeout=CONCURRENCY_TIMEOUT,
        )
        admitted = sum(1 for decision in decisions if decision.allowed)
        assert admitted == 5, f"{admitted} admitted across connections; the window fits exactly 5"
    finally:
        for store in stores:
            store.close()


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


async def test_state_survives_a_restart(db_path, clock):
    first = Store(db_path)
    limiter = RateLimiter(first, limit=50_000, window_seconds=60.0, clock=clock)
    assert (await limiter.check_and_reserve(TENANT, 40_000)).allowed
    first.close()

    second = Store(db_path)
    try:
        restarted = RateLimiter(second, limit=50_000, window_seconds=60.0, clock=clock)
        assert await restarted.usage(TENANT) == 40_000
        assert (await restarted.check_and_reserve(TENANT, 20_000)).allowed is False, (
            "bouncing the gateway reset the window"
        )
    finally:
        second.close()


async def test_the_state_is_really_in_sqlite_on_disk(limiter, db_path):
    await limiter.check_and_reserve(TENANT, 1_234)

    assert db_path.exists(), "no database file was created"
    with sqlite3.connect(db_path) as inspection:
        inspection.row_factory = sqlite3.Row
        rows = inspection.execute(
            "SELECT tenant, tokens, state FROM token_usage WHERE tenant = ?", (TENANT,)
        ).fetchall()
    assert [(row["tenant"], row["tokens"], row["state"]) for row in rows] == [(TENANT, 1_234, "reserved")]


async def test_expired_rows_are_physically_deleted_not_just_filtered(limiter, clock):
    for _ in range(20):
        await limiter.check_and_reserve(TENANT, 100)
        clock.advance(1)

    clock.advance(120)
    await limiter.check_and_reserve(TENANT, 100)
    assert await limiter.row_count() == 1, "the table grows without bound if rows are only filtered"


async def test_settle_keeps_the_original_timestamp(limiter, clock):
    decision = await limiter.check_and_reserve(TENANT, 20_000)
    clock.advance(59)
    await limiter.settle(decision.reservation, 500)
    assert await limiter.usage(TENANT) == 500

    clock.advance(2)  # 61s after the request started
    assert await limiter.usage(TENANT) == 0, (
        "a slow request must not hold quota beyond the window it started in"
    )


async def test_release_returns_the_whole_reservation(limiter):
    decision = await limiter.check_and_reserve(TENANT, 50_000)
    await limiter.release(decision.reservation)
    assert await limiter.usage(TENANT) == 0
    assert (await limiter.check_and_reserve(TENANT, 50_000)).allowed


# --------------------------------------------------------------------------
# Fallback - outcomes
# --------------------------------------------------------------------------


async def test_primary_success_uses_the_primary(routed):
    result = await routed.router.route(TENANT, message_body())
    assert result.provider_used == "primary"
    assert len(routed.primary_calls) == 1
    assert routed.fallback_calls == []


@pytest.mark.parametrize(
    ("behaviour", "reason"),
    [
        ("rate_limited", "429 is about this provider's capacity"),
        ("error", "5xx is about this provider's health"),
        ("unauthorized", "401 means our key for this provider is wrong"),
        ("forbidden", "403 likewise"),
    ],
)
async def test_a_failing_primary_fails_over(routed, behaviour, reason):
    result = await routed.router.route(
        TENANT, message_body(behaviour=behaviour, fallback_behaviour="ok")
    )
    assert result.status_code == 200, reason
    assert result.provider_used == "fallback"
    assert len(routed.fallback_calls) == 1


async def test_a_primary_timeout_fails_over(fast_routed):
    result = await fast_routed.router.route(
        TENANT, message_body(behaviour="slow", fallback_behaviour="ok")
    )
    assert result.provider_used == "fallback"


async def test_a_client_error_does_not_fail_over(routed):
    result = await routed.router.route(TENANT, message_body(behaviour="bad_request"))
    assert result.status_code == 400
    assert routed.fallback_calls == []


async def test_both_providers_failing_returns_a_sanitised_502(routed):
    result = await routed.router.route(TENANT, message_body(behaviour="error"))
    assert result.status_code == 502
    assert result.body["error"]["type"] == "api_error"


async def test_both_providers_timing_out_returns_504(fast_routed):
    result = await fast_routed.router.route(TENANT, message_body(behaviour="slow"))
    assert result.status_code == 504
    assert result.body["error"]["type"] == "timeout_error"


LEAKY_STRINGS = [
    "prod-eu-3", "acct_9931", "internal-llm.prod.svc",  # 429 body
    "worker.py", "/srv/model",  # 500 body
    "sk-live-9f3a", "org_88213", "vault", "prod/eu",  # 401 body
    "Traceback", "httpx", "asyncio",
]


@pytest.mark.parametrize("behaviour", ["rate_limited", "error", "unauthorized", "bad_request"])
async def test_no_provider_internals_reach_the_client(routed, behaviour):
    import json as _json

    result = await routed.router.route(TENANT, message_body(behaviour=behaviour))
    rendered = _json.dumps(result.body)
    for secret in LEAKY_STRINGS:
        assert secret not in rendered, f"{secret!r} leaked for behaviour={behaviour}"


async def test_the_error_contract_is_stable(routed):
    """The Anthropic-shaped envelope, plus gateway-owned diagnostics under
    `gateway` - and nothing else. A new top-level key is how upstream detail
    would start leaking."""
    result = await routed.router.route(TENANT, message_body(behaviour="error"))
    assert set(result.body) <= {"type", "error", "gateway"}
    assert set(result.body) >= {"type", "error"}
    assert result.body["type"] == "error"
    assert set(result.body["error"]) == {"type", "message"}
    # `gateway` carries provider names and outcome categories: gateway
    # vocabulary, not upstream text.
    assert set(result.body.get("gateway", {})) <= {"attempts", "limit_tokens", "used_tokens",
                                                   "requested_tokens", "window_seconds",
                                                   "retry_after_seconds", "provider_status"}


# --------------------------------------------------------------------------
# Fallback - the 3000 ms boundary
# --------------------------------------------------------------------------


def _delayed_client(delay_seconds: float) -> httpx.AsyncClient:
    """A provider that answers after exactly `delay_seconds`.

    A deterministic double rather than a sleeping mock server: the assertion is
    about *the router's* deadline, and anything else in the path adds jitter
    that the 1 ms boundary cannot tolerate.
    """

    async def respond(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(delay_seconds)
        return httpx.Response(200, json={"usage": {"input_tokens": 1, "output_tokens": 1}})

    return httpx.AsyncClient(transport=httpx.MockTransport(respond))


PRIMARY = Provider("primary", "http://primary.test/v1/messages", "claude-opus-5", timeout_seconds=3.0)


@pytest.mark.parametrize(
    ("delay_ms", "expected"),
    [
        (2_500, Outcome.SUCCESS),
        (2_900, Outcome.SUCCESS),
        (3_200, Outcome.TIMEOUT),
        (4_000, Outcome.TIMEOUT),
    ],
)
async def test_the_three_second_budget_at_its_boundary(delay_ms, expected):
    async with _delayed_client(delay_ms / 1000) as client:
        started = time.perf_counter()
        attempt = await asyncio.wait_for(call_provider(client, PRIMARY, message_body()), timeout=30.0)
        elapsed = time.perf_counter() - started

    assert attempt.outcome is expected, f"{delay_ms} ms -> {attempt.outcome}"
    if expected is Outcome.TIMEOUT:
        assert 2.9 <= elapsed < 3.6, f"the deadline fired at {elapsed:.3f}s, not ~3.0s"
    else:
        assert elapsed < 3.1


@pytest.mark.parametrize("delay_ms", [2_999, 3_000, 3_001])
async def test_the_deadline_is_not_resolvable_to_a_single_millisecond(delay_ms):
    """2999 / 3000 / 3001 ms, asserted honestly.

    The brief asks for these three exact values. They cannot be *pinned* to an
    outcome: `asyncio.timeout` fires on an event-loop wakeup, and loop
    scheduling plus `asyncio.sleep`'s own granularity are worth more than a
    millisecond on any real machine - 2,999 ms against a 3,000 ms budget was
    observed resolving as a timeout. Asserting SUCCESS there would be pinning a
    coin flip and would produce a flaky suite that people learn to re-run.

    What *is* guaranteed, and is what the budget actually promises, is asserted
    here: within a millisecond of the deadline the outcome is one of the two
    legitimate ones, never a hang and never an unclassified error, and the
    attempt is bounded at roughly the budget either way. The decisive
    behavioural assertions live in the test above, at 2,900 and 3,200 ms.
    """
    async with _delayed_client(delay_ms / 1000) as client:
        started = time.perf_counter()
        attempt = await asyncio.wait_for(call_provider(client, PRIMARY, message_body()), timeout=30.0)
        elapsed = time.perf_counter() - started

    assert attempt.outcome in (Outcome.SUCCESS, Outcome.TIMEOUT), attempt.outcome
    assert 2.9 <= elapsed < 3.6, f"{delay_ms} ms resolved in {elapsed:.3f}s"


async def test_the_deadline_is_wall_clock_not_per_phase():
    """A provider trickling bytes forever never trips httpx's per-phase
    timeouts; only a total deadline bounds it."""

    async def trickle(request: httpx.Request) -> httpx.Response:
        async def slow_body():
            # Bounded at ~10s so that a *failure* of the outer deadline shows up
            # as a slow assertion failure rather than as a stalled suite.
            for _ in range(200):
                await asyncio.sleep(0.05)
                yield b" "

        return httpx.Response(200, content=slow_body())

    async with httpx.AsyncClient(transport=httpx.MockTransport(trickle)) as client:
        started = time.perf_counter()
        attempt = await asyncio.wait_for(call_provider(client, PRIMARY, message_body()), timeout=30.0)
        elapsed = time.perf_counter() - started

    assert attempt.outcome is Outcome.TIMEOUT
    assert elapsed < 4.0, f"the trickling provider held the attempt for {elapsed:.1f}s"


async def test_the_fallback_gets_its_own_longer_budget(limiter):
    """Having spent the primary's budget, giving the backup the same 3s risks
    failing a request that was about to succeed."""
    slow_primary = Provider("primary", "http://p.test/v1/messages", "m", timeout_seconds=0.2)
    patient_fallback = Provider("fallback", "http://f.test/v1/messages", "m", timeout_seconds=2.0)

    async def respond(request: httpx.Request) -> httpx.Response:
        if request.url.host == "p.test":
            await asyncio.sleep(5)
        await asyncio.sleep(0.5)
        return httpx.Response(200, json={"usage": {"input_tokens": 1, "output_tokens": 1}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        router = Router(limiter, slow_primary, [patient_fallback], client)
        result = await router.route(TENANT, message_body())

    assert result.provider_used == "fallback"


# --------------------------------------------------------------------------
# Quota accounting under failure
# --------------------------------------------------------------------------


async def test_a_timeout_is_charged_not_released(fast_routed):
    """A timeout aborts our wait, not the provider's work. Releasing would let a
    tenant drive unlimited load by timing out every request."""
    await fast_routed.router.route(TENANT, message_body(behaviour="slow", max_tokens=5_000))
    assert await fast_routed.limiter.usage(TENANT) > 0


async def test_a_clean_failure_releases_the_reservation(routed):
    await routed.router.route(TENANT, message_body(behaviour="error", max_tokens=5_000))
    assert await routed.limiter.usage(TENANT) == 0


async def test_a_success_settles_to_reported_usage(routed):
    await routed.router.route(TENANT, message_body(max_tokens=20_000, input_tokens=40, output_tokens=60))
    assert await routed.limiter.usage(TENANT) == 100


async def test_a_provider_reporting_no_usage_is_charged_the_estimate(routed):
    """Charging zero would make an unreported response a way to bypass the
    limit entirely."""
    body = message_body(max_tokens=5_000, omit_usage=True)
    await routed.router.route(TENANT, body)
    assert await routed.limiter.usage(TENANT) >= 5_000


async def test_an_exception_inside_the_gateway_releases_the_reservation(limiter):
    class Exploding(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            raise RuntimeError("something in the gateway broke")

    async with httpx.AsyncClient(transport=Exploding()) as client:
        router = Router(limiter, PRIMARY, [], client)
        with pytest.raises(RuntimeError):
            await router.route(TENANT, message_body(max_tokens=5_000))

    assert await limiter.usage(TENANT) == 0, "nothing was billed for a request that blew up here"
