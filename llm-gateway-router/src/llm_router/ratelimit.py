"""Token-aware sliding window rate limiter.

Why a *sliding* window
----------------------
A fixed window ("50,000 per calendar minute") lets a tenant spend the full quota
at 11:59:59 and the full quota again at 12:00:00 - 100,000 tokens in two seconds,
which is exactly the burst the limit exists to prevent. This keeps individual
usage events with timestamps and sums the trailing 60 seconds, so the constraint
holds at every instant rather than at minute boundaries.

Reserve, then settle
--------------------
Token cost is only known *after* a call completes, but admission has to be
decided *before* it starts. So each request reserves an estimate up front
(prompt estimate + the `max_tokens` ceiling it could produce), and the
reservation is corrected to the provider's reported usage once the response
arrives. Reserving nothing and settling afterwards would let unlimited
concurrent requests through the check, which is the failure mode a naive
"count what you used" limiter has under exactly the load that matters.

A request that never completes releases its reservation, so a failed call does
not hold quota hostage for the rest of the window.
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from llm_router.store import Store, transaction

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 60.0
DEFAULT_TOKEN_LIMIT = 50_000


@dataclass(frozen=True)
class Reservation:
    """Quota held for one in-flight request."""

    request_id: str
    tenant: str
    estimated_tokens: int


@dataclass(frozen=True)
class LimitDecision:
    """The outcome of an admission check."""

    allowed: bool
    used_tokens: int
    limit: int
    reservation: Reservation | None = None
    retry_after_seconds: float | None = None

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used_tokens)


class RateLimiter:
    """A per-tenant sliding window over token spend, persisted in SQLite."""

    def __init__(
        self,
        store: Store,
        limit: int = DEFAULT_TOKEN_LIMIT,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        clock: Callable[[], float] = time.time,
    ):
        self._store = store
        self._limit = limit
        self._window = window_seconds
        # Injectable so tests can advance time instead of sleeping through a
        # 60-second window.
        self._clock = clock

    @property
    def limit(self) -> int:
        return self._limit

    def _evict(self, cursor: sqlite3.Cursor, now: float) -> int:
        """Drop events that have fallen out of the window.

        Called on every admission check rather than on a timer: the work is
        proportional to what actually expired, it keeps the table from growing
        without bound, and it means the summed total never includes stale rows
        even if the process has been idle for hours.
        """
        cursor.execute("DELETE FROM token_usage WHERE created_at <= ?", (now - self._window,))
        return cursor.rowcount

    def _used(self, cursor: sqlite3.Cursor, tenant: str, now: float) -> int:
        cursor.execute(
            "SELECT COALESCE(SUM(tokens), 0) AS total FROM token_usage "
            "WHERE tenant = ? AND created_at > ?",
            (tenant, now - self._window),
        )
        return int(cursor.fetchone()["total"])

    def _retry_after(self, cursor: sqlite3.Cursor, tenant: str, now: float, needed: int) -> float:
        """Seconds until enough quota frees up for `needed` tokens.

        Walks the window oldest-first, accumulating what each expiry returns,
        and reports when the running total covers the shortfall. That is a real
        answer rather than the usual "just retry in 60 seconds", so a client
        backing off correctly waits the minimum.
        """
        cursor.execute(
            "SELECT tokens, created_at FROM token_usage "
            "WHERE tenant = ? AND created_at > ? ORDER BY created_at ASC",
            (tenant, now - self._window),
        )
        rows = cursor.fetchall()

        used = sum(int(row["tokens"]) for row in rows)
        shortfall = used + needed - self._limit
        if shortfall <= 0:
            return 0.0

        freed = 0
        for row in rows:
            freed += int(row["tokens"])
            if freed >= shortfall:
                # This row leaves the window one full window after it was written.
                return max(0.0, (float(row["created_at"]) + self._window) - now)

        # Even emptying the window would not fit the request; it can never
        # succeed at this limit.
        return self._window

    async def check_and_reserve(self, tenant: str, estimated_tokens: int) -> LimitDecision:
        """Admit or reject one request, atomically reserving quota if admitted."""
        now = self._clock()
        request_id = str(uuid.uuid4())

        def work(connection: sqlite3.Connection) -> LimitDecision:
            def body(cursor: sqlite3.Cursor) -> LimitDecision:
                evicted = self._evict(cursor, now)
                if evicted:
                    logger.debug("Evicted %d expired usage row(s)", evicted)

                used = self._used(cursor, tenant, now)

                if used + estimated_tokens > self._limit:
                    return LimitDecision(
                        allowed=False,
                        used_tokens=used,
                        limit=self._limit,
                        retry_after_seconds=self._retry_after(cursor, tenant, now, estimated_tokens),
                    )

                cursor.execute(
                    "INSERT INTO token_usage (tenant, request_id, tokens, created_at, state) "
                    "VALUES (?, ?, ?, ?, 'reserved')",
                    (tenant, request_id, estimated_tokens, now),
                )
                return LimitDecision(
                    allowed=True,
                    used_tokens=used + estimated_tokens,
                    limit=self._limit,
                    reservation=Reservation(request_id, tenant, estimated_tokens),
                )

            return transaction(connection, body)

        return await self._store.execute(work)

    async def settle(self, reservation: Reservation, actual_tokens: int) -> None:
        """Correct a reservation to what the request actually cost.

        The original timestamp is kept, so the event ages out of the window from
        when the request *started*. Re-stamping it to completion time would let a
        slow request occupy quota for longer than the window.
        """
        def work(connection: sqlite3.Connection) -> None:
            def body(cursor: sqlite3.Cursor) -> None:
                cursor.execute(
                    "UPDATE token_usage SET tokens = ?, state = 'settled' WHERE request_id = ?",
                    (actual_tokens, reservation.request_id),
                )

            transaction(connection, body)

        await self._store.execute(work)
        logger.debug(
            "Settled %s: reserved %d, actual %d", reservation.request_id, reservation.estimated_tokens, actual_tokens
        )

    async def release(self, reservation: Reservation) -> None:
        """Give quota back for a request that never produced usage."""
        def work(connection: sqlite3.Connection) -> None:
            def body(cursor: sqlite3.Cursor) -> None:
                cursor.execute("DELETE FROM token_usage WHERE request_id = ?", (reservation.request_id,))

            transaction(connection, body)

        await self._store.execute(work)
        logger.debug("Released reservation %s", reservation.request_id)

    async def usage(self, tenant: str) -> int:
        """Tokens spent by `tenant` inside the current window."""
        now = self._clock()

        def work(connection: sqlite3.Connection) -> int:
            cursor = connection.cursor()
            try:
                return self._used(cursor, tenant, now)
            finally:
                cursor.close()

        return await self._store.execute(work)

    async def row_count(self) -> int:
        """Total retained events. Tests assert this stays bounded."""
        def work(connection: sqlite3.Connection) -> int:
            cursor = connection.cursor()
            try:
                cursor.execute("SELECT COUNT(*) AS count FROM token_usage")
                return int(cursor.fetchone()["count"])
            finally:
                cursor.close()

        return await self._store.execute(work)
