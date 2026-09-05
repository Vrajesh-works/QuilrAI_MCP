"""In-memory customer and refund data.

Deliberately not a database. The interesting behaviour of this server is at the
protocol and validation layer, so the data layer stays a dependency-free fixture
that makes every domain path reachable in a test. Swapping in a real store means
replacing this module; nothing above it depends on the storage choice.

State is per-process and resets on restart. `reset_store()` exists so tests
start from a known ledger.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


class DomainError(Exception):
    """A valid request whose business outcome is a refusal.

    Surfaced to the caller as `isError: true` with an explanatory message, not
    as a JSON-RPC error - the model can act on "insufficient refundable balance"
    but can do nothing useful with a transport failure.
    """

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class Customer:
    customer_id: str
    name: str
    email: str
    status: str  # "active" | "frozen" | "closed"
    lifetime_spend: float
    refundable_balance: float
    signed_up: str


@dataclass
class Refund:
    refund_id: str
    customer_id: str
    amount: float
    reason: str
    created_at: str


@dataclass
class _Store:
    customers: dict[str, Customer]
    refunds: dict[str, Refund] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _seed() -> dict[str, Customer]:
    """Fixtures picked so each refusal path has a customer that triggers it."""
    rows = [
        # Ordinary customer with room to refund.
        Customer("CUST-00042", "Ada Lovelace", "ada@example.com", "active", 1_240.00, 320.00, "2023-04-11"),
        # Nothing left to refund - exercises the balance check.
        Customer("CUST-00007", "Grace Hopper", "grace@example.com", "active", 89.99, 0.00, "2024-01-30"),
        # Frozen - exercises the account-status check before any balance math.
        Customer("CUST-01337", "Alan Turing", "alan@example.com", "frozen", 4_500.00, 1_200.00, "2022-09-02"),
        # Closed account, still readable but not refundable.
        Customer("CUST-99999", "Katherine Johnson", "katherine@example.com", "closed", 15.50, 0.00, "2021-06-18"),
    ]
    return {row.customer_id: row for row in rows}


_store = _Store(customers=_seed())


def reset_store() -> None:
    """Restore seed data and clear the refund ledger. For tests."""
    global _store
    _store = _Store(customers=_seed())


def get_customer(customer_id: str) -> dict[str, Any]:
    """Look up one customer.

    Raises:
        DomainError: no customer with that id.
    """
    with _store.lock:
        customer = _store.customers.get(customer_id)
    if customer is None:
        raise DomainError(
            "customer_not_found",
            f"No customer exists with id {customer_id}.",
            {"customer_id": customer_id},
        )
    return {
        "customer_id": customer.customer_id,
        "name": customer.name,
        "email": customer.email,
        "status": customer.status,
        "lifetime_spend": round(customer.lifetime_spend, 2),
        "refundable_balance": round(customer.refundable_balance, 2),
        "signed_up": customer.signed_up,
    }


def create_refund(customer_id: str, amount: float, reason: str) -> dict[str, Any]:
    """Issue a refund and decrement the customer's refundable balance.

    Raises:
        DomainError: unknown customer, non-active account, or an amount above
            the remaining refundable balance.
    """
    with _store.lock:
        customer = _store.customers.get(customer_id)
        if customer is None:
            raise DomainError(
                "customer_not_found",
                f"No customer exists with id {customer_id}.",
                {"customer_id": customer_id},
            )
        if customer.status != "active":
            raise DomainError(
                "account_not_active",
                f"Cannot refund customer {customer_id}: account status is {customer.status!r}.",
                {"customer_id": customer_id, "status": customer.status},
            )
        if amount > customer.refundable_balance:
            raise DomainError(
                "insufficient_refundable_balance",
                (
                    f"Refund of {amount:.2f} exceeds the refundable balance of "
                    f"{customer.refundable_balance:.2f} for {customer_id}."
                ),
                {
                    "customer_id": customer_id,
                    "requested": round(amount, 2),
                    "refundable_balance": round(customer.refundable_balance, 2),
                },
            )

        refund = Refund(
            refund_id=f"RFND-{uuid.uuid4().hex[:12].upper()}",
            customer_id=customer_id,
            amount=round(amount, 2),
            reason=reason,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        _store.refunds[refund.refund_id] = refund
        remaining = round(customer.refundable_balance - amount, 2)
        # Customer is frozen; replace the row rather than mutating it.
        _store.customers[customer_id] = Customer(
            customer_id=customer.customer_id,
            name=customer.name,
            email=customer.email,
            status=customer.status,
            lifetime_spend=customer.lifetime_spend,
            refundable_balance=remaining,
            signed_up=customer.signed_up,
        )

    return {
        "refund_id": refund.refund_id,
        "customer_id": refund.customer_id,
        "amount": refund.amount,
        "reason": refund.reason,
        "created_at": refund.created_at,
        "remaining_refundable_balance": remaining,
        "status": "issued",
    }
