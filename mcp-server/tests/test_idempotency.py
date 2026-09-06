"""Refund replay protection: sequential, concurrent, restart, and after loss.

The audit replayed one byte-identical refund three times and moved $150 off a
real balance, producing three distinct refund ids. Every test here counts the
number of times money actually moved, rather than checking a response shape - a
duplicate payout with a tidy-looking response is exactly the failure.
"""

from __future__ import annotations

import sqlite3
import threading

import pytest
from customer_mcp import store
from customer_mcp.idempotency import IdempotencyConfig, RefundLedger, fingerprint

REASON = "Duplicate charge on the April invoice."


@pytest.fixture
def ledger_path(tmp_path) -> str:
    return str(tmp_path / "refunds.sqlite")


def _counting_issue(counter: list[int]):
    def issue() -> dict:
        counter.append(1)
        return {"refund_id": f"RFND-{len(counter):04d}", "amount": 50.0}

    return issue


# --------------------------------------------------------------------------
# Sequential replay
# --------------------------------------------------------------------------


def test_three_identical_requests_move_money_once(ledger_path):
    ledger = RefundLedger(IdempotencyConfig(database_path=ledger_path))
    calls: list[int] = []
    results = [
        ledger.run_once(
            customer_id="CUST-00042",
            amount=50.0,
            reason=REASON,
            idempotency_key=None,
            issue=_counting_issue(calls),
        )
        for _ in range(3)
    ]

    assert len(calls) == 1, f"money moved {len(calls)} times"
    assert [replayed for _, replayed in results] == [False, True, True]
    assert {payload["refund_id"] for payload, _ in results} == {"RFND-0001"}


def test_the_end_to_end_tool_path_does_not_double_pay():
    """Through `store.create_refund`, against a real balance, as the audit did."""
    before = store.get_customer("CUST-00042")["refundable_balance"]
    payloads = [store.create_refund("CUST-00042", 50.0, REASON) for _ in range(3)]
    after = store.get_customer("CUST-00042")["refundable_balance"]

    assert before - after == pytest.approx(50.0), f"balance moved {before - after}, expected 50.00"
    assert len({payload["refund_id"] for payload in payloads}) == 1
    assert [payload["replayed"] for payload in payloads] == [False, True, True]
    assert len(store._store.refunds) == 1


def test_a_different_amount_is_a_different_refund():
    """Replay protection must not swallow a genuinely different request."""
    first = store.create_refund("CUST-00042", 50.0, REASON)
    second = store.create_refund("CUST-00042", 60.0, REASON)
    assert first["refund_id"] != second["refund_id"]
    assert second["replayed"] is False
    assert store.get_customer("CUST-00042")["refundable_balance"] == pytest.approx(320.0 - 110.0)


def test_a_different_reason_or_customer_is_a_different_refund():
    store.create_refund("CUST-00042", 10.0, REASON)
    other_reason = store.create_refund("CUST-00042", 10.0, "A completely different justification.")
    assert other_reason["replayed"] is False
    assert store.get_customer("CUST-00042")["refundable_balance"] == pytest.approx(300.0)


def test_an_explicit_key_lets_a_caller_repeat_an_identical_refund_on_purpose():
    """Two deliberate identical refunds are legitimate; the client says so with
    distinct keys. Without this the replay window would make them impossible."""
    first = store.create_refund("CUST-00042", 25.0, REASON, "ticket-1")
    second = store.create_refund("CUST-00042", 25.0, REASON, "ticket-2")
    assert first["refund_id"] != second["refund_id"]
    assert second["replayed"] is False
    assert store.get_customer("CUST-00042")["refundable_balance"] == pytest.approx(270.0)


def test_the_same_explicit_key_replays():
    first = store.create_refund("CUST-00042", 25.0, REASON, "ticket-1")
    second = store.create_refund("CUST-00042", 99.0, "A different reason entirely.", "ticket-1")
    assert second["refund_id"] == first["refund_id"], "the key is the identity, not the body"
    assert second["replayed"] is True
    assert store.get_customer("CUST-00042")["refundable_balance"] == pytest.approx(295.0)


def test_a_key_is_scoped_to_its_customer():
    """One customer's key must not replay - or read back - another's refund.

    `CUST-01337` is frozen, so reaching the balance logic at all is the proof:
    an unscoped key would have short-circuited to `CUST-00042`'s stored response
    and handed one customer's refund id to a request about another.
    """
    store.create_refund("CUST-00042", 25.0, REASON, "shared-key")
    with pytest.raises(store.DomainError) as refusal:
        store.create_refund("CUST-01337", 25.0, REASON, "shared-key")
    assert refusal.value.code == "account_not_active"


def test_ledger_keys_for_two_customers_differ():
    ledger = store.refund_ledger()
    assert ledger.operation_key("CUST-00042", 1.0, "r", "k") != ledger.operation_key(
        "CUST-01337", 1.0, "r", "k"
    )


# --------------------------------------------------------------------------
# Retry after the response was lost
# --------------------------------------------------------------------------


def test_a_retry_after_a_lost_response_returns_the_original(ledger_path):
    """The refund committed, the client never saw the answer, it retries."""
    ledger = RefundLedger(IdempotencyConfig(database_path=ledger_path))
    calls: list[int] = []
    committed, _ = ledger.run_once(
        customer_id="CUST-00042", amount=50.0, reason=REASON, idempotency_key=None,
        issue=_counting_issue(calls),
    )
    # ... response lost in transit; the client knows nothing and resends.
    replayed, was_replay = ledger.run_once(
        customer_id="CUST-00042", amount=50.0, reason=REASON, idempotency_key=None,
        issue=_counting_issue(calls),
    )
    assert was_replay is True
    assert replayed == committed, "a retry must be able to learn the original outcome"
    assert len(calls) == 1


# --------------------------------------------------------------------------
# Restart
# --------------------------------------------------------------------------


def test_the_record_survives_a_restart(ledger_path):
    """The failure this exists for. An in-memory set is empty exactly here."""
    calls: list[int] = []
    first_process = RefundLedger(IdempotencyConfig(database_path=ledger_path))
    original, _ = first_process.run_once(
        customer_id="CUST-00042", amount=50.0, reason=REASON, idempotency_key=None,
        issue=_counting_issue(calls),
    )
    first_process.close()

    second_process = RefundLedger(IdempotencyConfig(database_path=ledger_path))
    replayed, was_replay = second_process.run_once(
        customer_id="CUST-00042", amount=50.0, reason=REASON, idempotency_key=None,
        issue=_counting_issue(calls),
    )
    second_process.close()

    assert was_replay is True
    assert replayed == original
    assert len(calls) == 1, "the refund was issued again after restart"


def test_the_ledger_is_a_real_file_on_disk(ledger_path):
    ledger = RefundLedger(IdempotencyConfig(database_path=ledger_path))
    ledger.run_once(
        customer_id="CUST-00042", amount=50.0, reason=REASON, idempotency_key=None,
        issue=lambda: {"refund_id": "RFND-X"},
    )
    ledger.close()

    with sqlite3.connect(ledger_path) as inspection:
        rows = inspection.execute("SELECT operation_key, customer_id FROM refund_operations").fetchall()
    assert len(rows) == 1
    assert rows[0][1] == "CUST-00042"


# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------


def test_twenty_simultaneous_identical_requests_move_money_once(ledger_path):
    ledger = RefundLedger(IdempotencyConfig(database_path=ledger_path))
    calls: list[int] = []
    lock = threading.Lock()
    outcomes: list[tuple[dict, bool]] = []
    barrier = threading.Barrier(20)

    def issue() -> dict:
        with lock:
            calls.append(1)
            return {"refund_id": f"RFND-{len(calls):04d}"}

    def worker() -> None:
        barrier.wait()
        result = ledger.run_once(
            customer_id="CUST-00042", amount=50.0, reason=REASON, idempotency_key=None, issue=issue
        )
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(outcomes) == 20, "some worker never finished"
    assert len(calls) == 1, f"money moved {len(calls)} times under concurrency"
    assert sum(1 for _, replayed in outcomes if not replayed) == 1
    assert len({payload["refund_id"] for payload, _ in outcomes}) == 1


def test_separate_connections_on_one_file_still_move_money_once(ledger_path):
    """Stands in for several server processes sharing the volume.

    Each thread gets its own `RefundLedger`, so the in-process lock provides no
    protection and only `BEGIN IMMEDIATE` plus the primary key are left. This is
    the test that would fail if the transaction mode were relaxed to DEFERRED.
    """
    calls: list[int] = []
    lock = threading.Lock()
    errors: list[BaseException] = []
    replays: list[bool] = []
    barrier = threading.Barrier(8)

    def issue() -> dict:
        with lock:
            calls.append(1)
            return {"refund_id": f"RFND-{len(calls):04d}"}

    def worker() -> None:
        connection = RefundLedger(IdempotencyConfig(database_path=ledger_path))
        try:
            barrier.wait()
            _, replayed = connection.run_once(
                customer_id="CUST-00042", amount=50.0, reason=REASON, idempotency_key=None, issue=issue
            )
            with lock:
                replays.append(replayed)
        except BaseException as exc:  # noqa: BLE001 - recorded and asserted below
            with lock:
                errors.append(exc)
        finally:
            connection.close()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == [], f"concurrent access raised: {errors!r}"
    assert len(calls) == 1, f"money moved {len(calls)} times across connections"
    assert replays.count(False) == 1


# --------------------------------------------------------------------------
# Database-level guarantees
# --------------------------------------------------------------------------


def test_the_uniqueness_constraint_is_enforced_by_the_database(ledger_path):
    """Not by a check-then-act in Python, which races."""
    ledger = RefundLedger(IdempotencyConfig(database_path=ledger_path))
    ledger.run_once(
        customer_id="CUST-00042", amount=50.0, reason=REASON, idempotency_key="k1",
        issue=lambda: {"refund_id": "RFND-A"},
    )
    with pytest.raises(sqlite3.IntegrityError):
        ledger._connection.execute(
            "INSERT INTO refund_operations "
            "(operation_key, customer_id, fingerprint, response, created_at) VALUES (?,?,?,?,?)",
            ("key:CUST-00042:k1", "CUST-00042", "x", "{}", 0.0),
        )
    ledger.close()


def test_a_refused_refund_is_not_recorded_so_it_can_be_retried(ledger_path):
    """A frozen account today may be active tomorrow; the retry must work."""
    ledger = RefundLedger(IdempotencyConfig(database_path=ledger_path))

    def refuse() -> dict:
        raise store.DomainError("account_not_active", "nope")

    with pytest.raises(store.DomainError):
        ledger.run_once(
            customer_id="CUST-01337", amount=50.0, reason=REASON, idempotency_key=None, issue=refuse
        )

    payload, replayed = ledger.run_once(
        customer_id="CUST-01337", amount=50.0, reason=REASON, idempotency_key=None,
        issue=lambda: {"refund_id": "RFND-LATER"},
    )
    assert replayed is False
    assert payload["refund_id"] == "RFND-LATER"
    ledger.close()


def test_a_domain_refusal_through_the_tool_path_is_not_cached():
    with pytest.raises(store.DomainError):
        store.create_refund("CUST-01337", 10.0, REASON)
    with pytest.raises(store.DomainError):
        store.create_refund("CUST-01337", 10.0, REASON)


def test_records_outside_the_replay_window_stop_replaying(ledger_path):
    """Two deliberate identical refunds far enough apart are both honoured."""
    now = [1_000_000.0]
    ledger = RefundLedger(
        IdempotencyConfig(database_path=ledger_path, replay_window_seconds=60.0),
        clock=lambda: now[0],
    )
    calls: list[int] = []
    ledger.run_once(
        customer_id="CUST-00042", amount=50.0, reason=REASON, idempotency_key=None,
        issue=_counting_issue(calls),
    )
    now[0] += 30
    _, replayed = ledger.run_once(
        customer_id="CUST-00042", amount=50.0, reason=REASON, idempotency_key=None,
        issue=_counting_issue(calls),
    )
    assert replayed is True, "still inside the window"

    now[0] += 100
    _, replayed = ledger.run_once(
        customer_id="CUST-00042", amount=50.0, reason=REASON, idempotency_key=None,
        issue=_counting_issue(calls),
    )
    assert replayed is False, "past the window, this is a new refund"
    assert len(calls) == 2
    ledger.close()


def test_amount_formatting_cannot_defeat_the_fingerprint():
    """10.0 and 10.00 are the same payment."""
    assert fingerprint("CUST-1", 10.0, "r") == fingerprint("CUST-1", 10.00, "r")
    assert fingerprint("CUST-1", 10.0, "r") != fingerprint("CUST-1", 10.01, "r")
    assert fingerprint("CUST-1", 10.0, "r") != fingerprint("CUST-2", 10.0, "r")
