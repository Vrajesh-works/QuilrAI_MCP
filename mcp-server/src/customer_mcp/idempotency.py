"""Replay protection for the one tool that moves money.

What it guards
--------------
Without dedupe, three byte-identical `trigger_refund` calls issue three refunds
and move $150 off a real balance, producing three distinct refund ids. That is
not a theoretical retry: this server drops in-flight requests when stdin closes,
so a host that restarts mid-call cannot tell whether the refund landed, and its
only options are to retry - double-paying - or not to retry and under-pay.

Why SQLite and not a dict
-------------------------
The failure mode this exists to stop is *a retry after the process died*. An
in-memory set is empty exactly when it is needed. The ledger is therefore a real
file, and the uniqueness guarantee is a `UNIQUE` constraint enforced by the
database rather than a check-then-act in Python, which races.

The whole operation - the replay lookup, the refund, and recording the result -
runs inside one `BEGIN IMMEDIATE` transaction. `IMMEDIATE` takes the write lock
up front, so two concurrent identical requests serialise: the first commits a
refund, the second finds the committed record and returns it verbatim without
touching the balance. `BEGIN DEFERRED` would let both read, both decide the key
is new, and one would fail at commit with `SQLITE_BUSY_SNAPSHOT` - which
`busy_timeout` does not retry.

Operation identity
------------------
An explicit `idempotency_key` wins when the caller supplies one, which is the
contract a well-behaved client should use. When it is absent the key is derived
from `(customer_id, amount, reason)` inside a replay window, because a host
retrying a lost response resends exactly those three fields and has no key to
offer. The window is what keeps that from being permanent: two deliberately
identical refunds more than `replay_window_seconds` apart both succeed, and
inside the window a caller who genuinely means it must say so with a distinct
`idempotency_key`. Silently double-paying is the worse default.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = "data/customer_mcp.sqlite"
DEFAULT_REPLAY_WINDOW_SECONDS = 86_400.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS refund_operations (
    operation_key TEXT PRIMARY KEY,
    customer_id   TEXT NOT NULL,
    fingerprint   TEXT NOT NULL,
    response      TEXT NOT NULL,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_refund_operations_created_at
    ON refund_operations (created_at);
"""


@dataclass(frozen=True)
class IdempotencyConfig:
    database_path: str = DEFAULT_DB_PATH
    replay_window_seconds: float = DEFAULT_REPLAY_WINDOW_SECONDS

    @classmethod
    def from_env(cls) -> IdempotencyConfig:
        return cls(
            database_path=os.environ.get("CUSTOMER_MCP_DB", DEFAULT_DB_PATH),
            replay_window_seconds=float(
                os.environ.get("CUSTOMER_MCP_REPLAY_WINDOW_SECONDS", DEFAULT_REPLAY_WINDOW_SECONDS)
            ),
        )


def fingerprint(customer_id: str, amount: float, reason: str) -> str:
    """A stable identity for one logical refund.

    `amount` is rendered at cent precision so 10.0 and 10.00 are the same
    operation - they are the same payment, and treating them as distinct would
    let a client defeat replay protection by reformatting a number.
    """
    material = json.dumps(
        {"customer_id": customer_id, "amount": f"{amount:.2f}", "reason": reason},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class RefundLedger:
    """The persistent record of which refunds have already been issued."""

    def __init__(self, config: IdempotencyConfig | None = None, clock: Callable[[], float] = time.time):
        self._config = config or IdempotencyConfig.from_env()
        self._clock = clock
        # Serialises writers inside this process. `BEGIN IMMEDIATE` serialises
        # them across processes; both are needed because a single sqlite3
        # connection is not safe to use from several threads at once.
        self._lock = threading.Lock()

        path = self._config.database_path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.executescript(_SCHEMA)

    def close(self) -> None:
        self._connection.close()

    @contextmanager
    def _immediate(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def operation_key(self, customer_id: str, amount: float, reason: str, supplied: str | None) -> str:
        if supplied:
            # Namespaced by customer so one tenant's key cannot collide with -
            # or be used to read back - another's refund.
            return f"key:{customer_id}:{supplied}"
        return f"auto:{fingerprint(customer_id, amount, reason)}"

    def run_once(
        self,
        *,
        customer_id: str,
        amount: float,
        reason: str,
        idempotency_key: str | None,
        issue: Callable[[], dict[str, Any]],
    ) -> tuple[dict[str, Any], bool]:
        """Issue the refund, or return the one already issued for this operation.

        Returns `(payload, replayed)`. `replayed` is True when no money moved
        because a matching record already existed.

        `issue` is called at most once, inside the transaction, and only after
        the replay check has come up empty. If it raises - an unknown customer,
        a frozen account, an amount over the balance - the transaction rolls
        back and nothing is recorded, so the caller may legitimately retry once
        the underlying refusal is resolved.
        """
        key = self.operation_key(customer_id, amount, reason, idempotency_key)
        now = self._clock()
        horizon = now - self._config.replay_window_seconds

        with self._immediate() as connection:
            # Housekeeping first, inside the same lock, so an expired record can
            # never be seen as a live one by a concurrent caller.
            connection.execute("DELETE FROM refund_operations WHERE created_at <= ?", (horizon,))
            row = connection.execute(
                "SELECT response FROM refund_operations WHERE operation_key = ?", (key,)
            ).fetchone()
            if row is not None:
                return json.loads(row["response"]), True

            payload = issue()
            connection.execute(
                "INSERT INTO refund_operations "
                "(operation_key, customer_id, fingerprint, response, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    key,
                    customer_id,
                    fingerprint(customer_id, amount, reason),
                    json.dumps(payload, ensure_ascii=False),
                    now,
                ),
            )
            return payload, False

    def clear(self) -> None:
        """Drop every record. For tests only."""
        with self._immediate() as connection:
            connection.execute("DELETE FROM refund_operations")
