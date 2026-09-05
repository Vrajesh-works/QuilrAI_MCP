"""Walk the router through rate limiting and failover.

    uv run python scripts/demo.py

Runs in process against mock providers, on a temporary SQLite file. Each step
prints what the client saw and which provider actually served it.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import httpx
from starlette.applications import Starlette

from llm_router.providers import Provider
from llm_router.ratelimit import RateLimiter
from llm_router.router import Router
from llm_router.store import Store
from mock_model_provider.app import create_app as create_provider

TENANT = "sk-tenant-alpha"


class DualTransport(httpx.AsyncBaseTransport):
    def __init__(self, apps: dict[str, Starlette]):
        self._transports = {host: httpx.ASGITransport(app=app) for host, app in apps.items()}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._transports[request.url.host].handle_async_request(request)


def body(**extra) -> dict:
    return {"messages": [{"role": "user", "content": "Summarise the incident report."}], "max_tokens": 500, **extra}


def banner(title: str) -> None:
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


async def main() -> None:
    database = Path(tempfile.mkdtemp()) / "router.sqlite"
    store = Store(database)
    limiter = RateLimiter(store, limit=50_000, window_seconds=60.0)

    primary_app = create_provider("primary")
    fallback_app = create_provider("fallback")
    transport = DualTransport({"primary.test": primary_app, "fallback.test": fallback_app})

    primary = Provider("primary", "http://primary.test/v1/messages", "claude-opus-5", timeout_seconds=3.0)
    fallback = Provider("fallback", "http://fallback.test/v1/messages", "claude-sonnet-5", timeout_seconds=10.0)

    print(f"on-disk database: {database}\n")

    async with httpx.AsyncClient(transport=transport) as client:
        router = Router(limiter, primary, [fallback], client)

        async def show(label: str, request: dict, tenant: str = TENANT) -> None:
            banner(label)
            before = len(primary_app.state.received), len(fallback_app.state.received)
            result = await router.route(tenant, request)
            after = len(primary_app.state.received), len(fallback_app.state.received)

            print(f"HTTP {result.status_code}")
            if result.status_code == 200:
                print(f"served by:  {result.provider_used}")
            else:
                print(f"error:      {result.body['error']['type']}: {result.body['error']['message']}")
                if "gateway" in result.body:
                    print(f"detail:     {json.dumps(result.body['gateway'])}")
            print(f"providers called: primary={after[0] - before[0]} fallback={after[1] - before[1]}")
            print(f"window usage:     {await limiter.usage(tenant)} / {limiter.limit} tokens")
            print()

        await show("Healthy primary", body())

        await show(
            "Primary returns 429 -> automatic failover",
            body(behaviour="rate_limited", fallback_behaviour="ok"),
        )

        await show(
            "Primary hangs past its 3000ms budget -> failover",
            body(behaviour="slow", delay_ms=30_000, fallback_behaviour="ok", fallback_delay_ms=0),
        )

        await show(
            "Both providers down -> sanitised 502, quota released",
            body(behaviour="error"),
        )

        await show(
            "Malformed request -> 4xx relayed, backup NOT burned",
            body(behaviour="bad_request"),
        )

        banner("Exhausting the token budget")
        spender = "sk-tenant-heavy"
        served = 0
        while True:
            result = await router.route(spender, body(max_tokens=9_000, output_tokens=9_000))
            if result.status_code == 429:
                print(f"served {served} request(s), then rate limited")
                print(f"used:        {result.body['gateway']['used_tokens']} / {result.body['gateway']['limit_tokens']}")
                print(f"retry after: {result.body['gateway']['retry_after_seconds']}s")
                print(f"headers:     retry-after={result.headers['retry-after']} "
                      f"remaining={result.headers['x-ratelimit-remaining-tokens']}")
                break
            served += 1
        print()

        banner("Concurrency: 40 simultaneous requests against a fresh budget")
        burst_tenant = "sk-tenant-burst"
        results = await asyncio.gather(
            *(router.route(burst_tenant, body(max_tokens=4_000, output_tokens=4_000)) for _ in range(40))
        )
        codes: dict[int, int] = {}
        for result in results:
            codes[result.status_code] = codes.get(result.status_code, 0) + 1
        usage = await limiter.usage(burst_tenant)
        print(f"status codes: {codes}")
        print(f"usage:        {usage} / {limiter.limit} tokens")
        print(f"within limit: {usage <= limiter.limit}")
        print(f"retained rows: {await limiter.row_count()}")

    store.close()


if __name__ == "__main__":
    asyncio.run(main())
