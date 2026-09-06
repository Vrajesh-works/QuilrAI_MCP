"""Shared fixtures: a real client session talking to the real server.

The session runs over in-memory streams rather than a subprocess, so tests
exercise the genuine protocol layer - request framing, error serialisation,
result parsing - without process startup cost. `test_stdout_purity.py` is the
deliberate exception: descriptor behaviour can only be proved for real.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

from customer_mcp import store
from customer_mcp.idempotency import IdempotencyConfig
from customer_mcp.server import build_server


@pytest.fixture(autouse=True)
def _clean_store(tmp_path) -> None:
    """Every test starts from seed data with an empty refund ledger.

    The replay ledger is a real SQLite file, so each test gets its own under
    `tmp_path`. Pointing it at `:memory:` would be convenient and would also
    quietly stop testing the thing that matters - that the record survives the
    process.
    """
    store.configure_ledger(IdempotencyConfig(database_path=str(tmp_path / "ledger.sqlite")))
    store.reset_store()


@asynccontextmanager
async def connected_session() -> AsyncGenerator[ClientSession, None]:
    """Yield an initialized ClientSession wired to an in-process server."""
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams

        server = build_server()

        async with anyio.create_task_group() as tg:

            async def _run_server() -> None:
                await server.run(
                    server_read,
                    server_write,
                    server.create_initialization_options(),
                    # Surface handler bugs as test failures rather than letting
                    # them serialise into an opaque error response.
                    raise_exceptions=False,
                )

            tg.start_soon(_run_server)

            async with ClientSession(client_read, client_write) as session:
                await session.initialize()
                yield session

            tg.cancel_scope.cancel()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
