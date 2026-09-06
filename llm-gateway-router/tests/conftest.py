"""Fixtures: an on-disk store, a controllable clock, and a wired router."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from starlette.applications import Starlette

from llm_router.providers import Provider
from llm_router.ratelimit import RateLimiter
from llm_router.router import Router
from llm_router.store import Store
from mock_model_provider.app import create_app as create_provider

# Rate-limit keys used when calling `Router.route` directly. These are already
# resolved, opaque tenant ids - the router never sees a credential.
TENANT = "tenant-alpha"
OTHER_TENANT = "tenant-beta"

# The credentials a client presents over HTTP, and the tenant ids they resolve
# to. Distinct from the ids on purpose: a test that passed a key straight
# through as a rate-limit key would not notice if authentication were removed.
API_KEY = "test-key-alpha"
OTHER_API_KEY = "test-key-beta"
TENANT_TABLE = {API_KEY: TENANT, OTHER_API_KEY: OTHER_TENANT}


class FakeClock:
    """A hand-cranked clock.

    The window is 60 seconds; sleeping through it in tests would make the suite
    unusable, and `time.sleep` in a concurrency test hides exactly the races
    these tests exist to find.
    """

    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """A real file on disk, not :memory:.

    Using an in-memory database here would skip WAL, the busy timeout and the
    file locking that the concurrency tests depend on.
    """
    return tmp_path / "router.sqlite"


@pytest.fixture
def store(db_path: Path) -> Iterator[Store]:
    store = Store(db_path)
    try:
        yield store
    finally:
        store.close()


@pytest.fixture
def limiter(store: Store, clock: FakeClock) -> RateLimiter:
    return RateLimiter(store, limit=50_000, window_seconds=60.0, clock=clock)


def message_body(text: str = "hello", max_tokens: int = 100, **extra) -> dict:
    return {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, **extra}


class RoutedProviders:
    """Primary and fallback apps plus the router wired to them."""

    def __init__(self, primary: Starlette, fallback: Starlette, router: Router, limiter: RateLimiter):
        self.primary = primary
        self.fallback = fallback
        self.router = router
        self.limiter = limiter

    @property
    def primary_calls(self) -> list[dict]:
        return self.primary.state.received

    @property
    def fallback_calls(self) -> list[dict]:
        return self.fallback.state.received


class _DualTransport(httpx.AsyncBaseTransport):
    """Routes to one of two in-process apps by URL host."""

    def __init__(self, apps: dict[str, Starlette]):
        self._transports = {host: httpx.ASGITransport(app=app) for host, app in apps.items()}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        transport = self._transports.get(request.url.host)
        if transport is None:
            raise httpx.ConnectError(f"no mock provider for host {request.url.host}")
        return await transport.handle_async_request(request)


async def _build_routed(
    limiter: RateLimiter, primary_timeout: float, fallback_timeout: float
) -> AsyncGenerator[RoutedProviders, None]:
    primary_app = create_provider("primary")
    fallback_app = create_provider("fallback")

    transport = _DualTransport({"primary.test": primary_app, "fallback.test": fallback_app})

    primary = Provider(
        name="primary", url="http://primary.test/v1/messages", model="claude-opus-5", timeout_seconds=primary_timeout
    )
    fallback = Provider(
        name="fallback",
        url="http://fallback.test/v1/messages",
        model="claude-sonnet-5",
        timeout_seconds=fallback_timeout,
    )

    async with httpx.AsyncClient(transport=transport) as client:
        yield RoutedProviders(primary_app, fallback_app, Router(limiter, primary, [fallback], client), limiter)


@pytest_asyncio.fixture
async def routed(limiter: RateLimiter) -> AsyncGenerator[RoutedProviders, None]:
    """The production timeouts: a 3s primary budget, 10s on the fallback."""
    async for routed in _build_routed(limiter, primary_timeout=3.0, fallback_timeout=10.0):
        yield routed


@pytest_asyncio.fixture
async def fast_routed(limiter: RateLimiter) -> AsyncGenerator[RoutedProviders, None]:
    """Same wiring with short budgets.

    Tests that only need to exercise a timeout *path* should not spend 13 real
    seconds proving it. The tests that pin the actual 3000ms budget use `routed`.
    """
    async for routed in _build_routed(limiter, primary_timeout=0.3, fallback_timeout=0.5):
        yield routed


@pytest_asyncio.fixture
async def gateway(limiter: RateLimiter, db_path: Path) -> AsyncGenerator[httpx.AsyncClient, None]:
    """The full HTTP app, wired to in-process providers."""
    from llm_router.app import create_app
    from llm_router.config import Config

    primary_app = create_provider("primary")
    fallback_app = create_provider("fallback")
    transport = _DualTransport({"primary.test": primary_app, "fallback.test": fallback_app})

    config = Config(
        database_path=str(db_path),
        token_limit=50_000,
        window_seconds=60.0,
        primary=Provider("primary", "http://primary.test/v1/messages", "claude-opus-5", 3.0),
        fallbacks=[Provider("fallback", "http://fallback.test/v1/messages", "claude-sonnet-5", 10.0)],
        tenants=dict(TENANT_TABLE),
    )
    app = create_app(config)

    async with httpx.AsyncClient(transport=transport) as provider_client:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
        ) as client:
            async with app.router.lifespan_context(app):
                # Swap in the transport that reaches the in-process providers.
                app.state.router = Router(app.state.limiter, config.primary, config.fallbacks, provider_client)
                client.primary_app = primary_app
                client.fallback_app = fallback_app
                yield client
