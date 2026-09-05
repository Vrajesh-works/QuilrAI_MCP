"""Fixtures wiring the gateway to the mock downstream, entirely in process.

The gateway's pooled `httpx.AsyncClient` is swapped for one backed by an
`ASGITransport` around the mock server, so requests traverse the real gateway
code path - auth, parsing, policy, header handling, response merging - without
binding a port. The mock records what it receives, which is how the tests assert
that blocked calls never arrive.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import httpx
import pytest
import pytest_asyncio

from mcp_gateway.app import create_app
from mcp_gateway.config import Config
from mock_downstream import app as mock_app_module

ADMIN_TOKEN = "admin-token-abc123"
VIEWER_TOKEN = "viewer-token-xyz789"

DOWNSTREAM_URL = "http://downstream.test/mcp"


def auth(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


@pytest.fixture(autouse=True)
def _clear_downstream_log() -> None:
    mock_app_module.received.clear()


@pytest.fixture
def received() -> list[dict]:
    """Messages the downstream server actually received."""
    return mock_app_module.received


@pytest.fixture
def config() -> Config:
    return Config(
        downstream_url=DOWNSTREAM_URL,
        request_timeout_seconds=5.0,
        filter_tools_list=False,
    )


def _build_client(config: Config, transport: httpx.AsyncBaseTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport)


@pytest_asyncio.fixture
async def gateway(config: Config) -> AsyncGenerator[httpx.AsyncClient, None]:
    """An HTTP client pointed at the gateway, which proxies to the mock server."""
    async for client in _gateway_with(config, httpx.ASGITransport(app=mock_app_module.create_app())):
        yield client


@pytest_asyncio.fixture
async def filtering_gateway(config: Config) -> AsyncGenerator[httpx.AsyncClient, None]:
    """A gateway with tools/list filtering switched on."""
    filtering = Config(
        downstream_url=config.downstream_url,
        request_timeout_seconds=config.request_timeout_seconds,
        filter_tools_list=True,
    )
    async for client in _gateway_with(filtering, httpx.ASGITransport(app=mock_app_module.create_app())):
        yield client


async def _gateway_with(
    config: Config, downstream_transport: httpx.AsyncBaseTransport
) -> AsyncGenerator[httpx.AsyncClient, None]:
    app = create_app(config)
    async with httpx.AsyncClient(transport=downstream_transport) as downstream_client:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
        ) as client:
            # Enter the app's lifespan so state is populated, then point its
            # pooled client at the in-process downstream.
            async with app.router.lifespan_context(app):
                app.state.http_client = downstream_client
                yield client


@pytest_asyncio.fixture
async def capturing_gateway(config: Config) -> AsyncGenerator[tuple[httpx.AsyncClient, dict], None]:
    """A gateway whose downstream records the headers it was sent."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    captured: dict = {}

    async def record(request: httpx.Request) -> JSONResponse:
        captured["headers"] = {key.lower(): value for key, value in request.headers.items()}
        return JSONResponse({"jsonrpc": "2.0", "id": 1, "result": {"tools": []}})

    downstream = Starlette(routes=[Route("/mcp", record, methods=["POST"])])
    async for client in _gateway_with(config, httpx.ASGITransport(app=downstream)):
        yield client, captured


class BrokenTransport(httpx.AsyncBaseTransport):
    """A downstream that always fails, for the error-sanitisation tests."""

    def __init__(self, exc: Exception):
        self.exc = exc

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise self.exc


@pytest_asyncio.fixture
async def timing_out_gateway(config: Config) -> AsyncGenerator[httpx.AsyncClient, None]:
    transport = BrokenTransport(httpx.ConnectTimeout("timed out connecting to 10.1.2.3:8081"))
    async for client in _gateway_with(config, transport):
        yield client


@pytest_asyncio.fixture
async def unreachable_gateway(config: Config) -> AsyncGenerator[httpx.AsyncClient, None]:
    transport = BrokenTransport(
        httpx.ConnectError("[Errno 111] Connection refused to internal-mcp.prod.svc.cluster.local:8081")
    )
    async for client in _gateway_with(config, transport):
        yield client
