"""What `BEGIN IMMEDIATE` actually buys, verified by forcing the interleaving.

These are deliberately synchronous and thread-driven. The async tests in
`test_concurrency.py` cannot reach this: `asyncio.to_thread` dispatches fast
enough that two admission checks on separate connections finish one after the
other, so the dangerous window never opens. Proving the transaction mode matters
means holding one transaction open while another runs.

Swap `BEGIN IMMEDIATE` for `BEGIN DEFERRED` in `store.py` and both tests here
fail - which is what makes them worth having.
"""

from __future__ import annotations

import sqlite3
import threading
import time

from llm_router.store import Store, transaction

INSERT = (
    "INSERT INTO token_usage (tenant, request_id, tokens, created_at, state) "
    "VALUES ('t', ?, ?, 1000.0, 'reserved')"
)
SUM = "SELECT COALESCE(SUM(tokens), 0) AS total FROM token_usage"


def _run_overlapping(db_path, hold_seconds: float = 0.3):
    """Run two write transactions on separate connections, forced to overlap.

    Returns (errors, what B read). Accesses `_connection` directly because the
    point is to control transaction boundaries, which the async API deliberately
    hides.
    """
    first, second = Store(db_path), Store(db_path)
    started = threading.Event()
    errors: list[tuple[str, BaseException]] = []
    observed: dict[str, int] = {}

    def worker_first() -> None:
        def body(cursor: sqlite3.Cursor) -> None:
            cursor.execute(INSERT, ("first", 10_000))
            started.set()
            # Hold the transaction open so the second one is forced to contend.
            time.sleep(hold_seconds)

        try:
            transaction(first._connection, body)
        except BaseException as exc:  # noqa: BLE001 - recorded, then asserted on
            errors.append(("first", exc))

    def worker_second() -> None:
        started.wait(timeout=5)

        def body(cursor: sqlite3.Cursor) -> None:
            cursor.execute(SUM)
            observed["second"] = int(cursor.fetchone()["total"])
            cursor.execute(INSERT, ("second", 10_000))

        try:
            transaction(second._connection, body)
        except BaseException as exc:  # noqa: BLE001
            errors.append(("second", exc))

    threads = [threading.Thread(target=worker_first), threading.Thread(target=worker_second)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        return errors, observed
    finally:
        first.close()
        second.close()


def test_overlapping_writers_do_not_raise(db_path):
    """The second writer waits its turn instead of erroring.

    Under `BEGIN DEFERRED` it instead fails with SQLITE_BUSY_SNAPSHOT
    ("database is locked") when it tries to upgrade its read to a write - and
    `busy_timeout` does not retry a snapshot conflict, so the gateway would
    return a 500 rather than a rate-limit decision.
    """
    errors, _ = _run_overlapping(db_path)

    assert errors == [], f"a writer failed under contention: {errors}"


def test_the_second_writer_sees_the_first_writers_commit(db_path):
    """No stale snapshot: the read reflects everything already committed.

    This is the property the admission check depends on. Reading a total from
    before a concurrent reservation is what lets two requests be admitted
    against the same quota.
    """
    _, observed = _run_overlapping(db_path)

    assert observed["second"] == 10_000, (
        f"second transaction read {observed.get('second')} instead of 10,000 - it saw a stale snapshot"
    )
