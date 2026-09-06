"""On-disk SQLite backing the rate limiter.

Three decisions here carry the weight:

**WAL mode.** The default rollback journal takes an exclusive lock for every
write, so concurrent requests serialise on the filesystem and throughput
collapses under load. WAL lets readers proceed during a write, which is the
right shape for "many concurrent requests, each doing one small write".

**One connection, `check_same_thread=False`, guarded by a lock.** `sqlite3`
connections are not safe to share across threads without care, and every DB call
here is dispatched to a worker thread (see `execute`) so the event loop is never
blocked on disk I/O. A single connection plus an `asyncio.Lock` gives a simple,
provably-serialised writer rather than a pool whose interleavings have to be
reasoned about. **This lock is what makes the admission check atomic between
coroutines sharing one `Store`** - and only those.

**`BEGIN IMMEDIATE` for the admission check.** The lock above does nothing
across *connections*: run the gateway as several uvicorn workers against one
SQLite file and each process has its own connection. There, reading the usage
total and writing a reservation must be atomic at the database level.

Under a deferred transaction both connections read the same WAL snapshot, and
the second one to write fails with SQLITE_BUSY_SNAPSHOT - reported as
``database is locked``. Note that `busy_timeout` does **not** retry a snapshot
conflict, so this surfaces as a hard error rather than a wait: under contention
the gateway would return 500s instead of rate-limit decisions. `BEGIN IMMEDIATE`
takes the write lock before reading, so the second transaction waits (honouring
`busy_timeout`) and then reads a snapshot that includes the first one's commit.

The test suite forces that interleaving and fails if this becomes
`BEGIN DEFERRED`.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

SCHEMA = """
CREATE TABLE IF NOT EXISTS token_usage (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant      TEXT    NOT NULL,
    request_id  TEXT    NOT NULL,
    tokens      INTEGER NOT NULL,
    created_at  REAL    NOT NULL,
    state       TEXT    NOT NULL CHECK (state IN ('reserved', 'settled'))
);

-- The limiter's only hot query is "sum tokens for one tenant since T", and
-- eviction deletes by time. This index serves both.
CREATE INDEX IF NOT EXISTS idx_usage_tenant_time ON token_usage (tenant, created_at);

-- Reconciliation and release look a reservation up by its request id.
CREATE UNIQUE INDEX IF NOT EXISTS idx_usage_request ON token_usage (request_id);
"""


class Store:
    """A thin async wrapper over one SQLite connection."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        # Serialises access from the async side; see the module docstring.
        self._lock = asyncio.Lock()
        self._configure()

    def _configure(self) -> None:
        cursor = self._connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        # NORMAL is the usual WAL pairing: durable across process crashes, and
        # avoids an fsync per commit. A rate-limit counter losing the last
        # milliseconds of writes to a power cut is an acceptable trade.
        cursor.execute("PRAGMA synchronous=NORMAL")
        # Rather than failing instantly when another writer holds the lock, wait.
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.executescript(SCHEMA)
        cursor.close()

    async def execute(self, work: Callable[[sqlite3.Connection], T]) -> T:
        """Run `work` against the connection on a worker thread, under the lock.

        SQLite calls are blocking. Running them inline would stall the event loop
        for every other in-flight request, which matters most under exactly the
        load that makes rate limiting necessary.
        """
        async with self._lock:
            return await asyncio.to_thread(work, self._connection)

    def close(self) -> None:
        self._connection.close()


def transaction(connection: sqlite3.Connection, work: Callable[[sqlite3.Cursor], T]) -> T:
    """Run `work` inside an immediate (write-locked) transaction.

    `BEGIN IMMEDIATE` rather than a plain `BEGIN`: the admission check reads then
    writes, and on a *second connection* a deferred transaction would read a
    stale snapshot and then fail to upgrade. See the module docstring.
    """
    cursor = connection.cursor()
    cursor.execute("BEGIN IMMEDIATE")
    try:
        result = work(cursor)
    except BaseException:
        cursor.execute("ROLLBACK")
        raise
    cursor.execute("COMMIT")
    return result


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]
