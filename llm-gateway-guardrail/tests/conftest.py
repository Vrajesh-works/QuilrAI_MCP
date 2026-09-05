"""Fixtures wiring the guardrail to the mock provider, in process."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import httpx
import pytest
import pytest_asyncio

from llm_guardrail.app import create_app
from llm_guardrail.config import Config
from llm_guardrail.sse import SSEParser
from mock_provider.app import create_app as create_provider

UPSTREAM_URL = "http://provider.test/v1/messages"


@pytest.fixture
def config() -> Config:
    return Config(upstream_url=UPSTREAM_URL, request_timeout_seconds=5.0, read_timeout_seconds=30.0)


async def _guardrail_with(
    config: Config, transport: httpx.AsyncBaseTransport
) -> AsyncGenerator[httpx.AsyncClient, None]:
    app = create_app(config)
    async with httpx.AsyncClient(transport=transport) as upstream:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://guardrail.test"
        ) as client:
            async with app.router.lifespan_context(app):
                app.state.http_client = upstream
                yield client


@pytest_asyncio.fixture
async def guardrail(config: Config) -> AsyncGenerator[httpx.AsyncClient, None]:
    async for client in _guardrail_with(config, httpx.ASGITransport(app=create_provider())):
        yield client


class FailingTransport(httpx.AsyncBaseTransport):
    """An upstream that always fails, for the error-sanitisation tests."""

    def __init__(self, exc: Exception):
        self.exc = exc

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise self.exc


@pytest_asyncio.fixture
async def unreachable_guardrail(config: Config) -> AsyncGenerator[httpx.AsyncClient, None]:
    transport = FailingTransport(httpx.ConnectError("refused by 10.4.5.6:8091 (internal-llm.prod.svc)"))
    async for client in _guardrail_with(config, transport):
        yield client


def _delta_text(payload: object) -> str | None:
    if not isinstance(payload, dict) or payload.get("type") != "content_block_delta":
        return None
    delta = payload.get("delta")
    if not isinstance(delta, dict) or delta.get("type") != "text_delta":
        return None
    return delta.get("text")


async def collect_text(response: httpx.Response) -> str:
    """Reassemble the assistant text from a redacted SSE response."""
    parser = SSEParser()
    pieces: list[str] = []

    async for chunk in response.aiter_bytes():
        for event in parser.feed(chunk):
            text = _delta_text(event.json())
            if text is not None:
                pieces.append(text)
    for event in parser.flush():
        text = _delta_text(event.json())
        if text is not None:
            pieces.append(text)

    return "".join(pieces)


async def collect_raw(response: httpx.Response) -> bytes:
    raw = b""
    async for chunk in response.aiter_bytes():
        raw += chunk
    return raw


def request_body(text: str, **overrides) -> dict:
    return {"model": "claude-sonnet-5", "stream": True, "text": text, **overrides}


# --- Live servers ---------------------------------------------------------
#
# httpx's ASGITransport accumulates the whole response body before returning it
# (see `handle_async_request`: it waits on `response_complete`), so it cannot be
# used to measure time-to-first-token. Anything asserting on streaming latency
# end-to-end needs real sockets.


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _ThreadedServer:
    """A uvicorn server on its own thread, for tests that need real sockets."""

    def __init__(self, app, port: int):
        import uvicorn

        self.port = port
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
        )

    def start(self) -> None:
        import threading

        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

    def wait_until_ready(self, timeout: float = 20.0) -> None:
        import socket
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._server.started:
                return
            with socket.socket() as probe:
                probe.settimeout(0.2)
                if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                    return
            time.sleep(0.05)
        raise TimeoutError(f"server on port {self.port} never came up")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


@pytest_asyncio.fixture
async def live_guardrail() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Guardrail and provider on real ports, so streaming is genuinely streaming."""
    provider_port = _free_port()
    guardrail_port = _free_port()

    provider = _ThreadedServer(create_provider(), provider_port)
    guardrail = _ThreadedServer(
        create_app(
            Config(
                upstream_url=f"http://127.0.0.1:{provider_port}/v1/messages",
                request_timeout_seconds=5.0,
                read_timeout_seconds=60.0,
            )
        ),
        guardrail_port,
    )

    provider.start()
    guardrail.start()
    try:
        provider.wait_until_ready()
        guardrail.wait_until_ready()
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{guardrail_port}", timeout=30.0) as client:
            yield client
    finally:
        guardrail.stop()
        provider.stop()
