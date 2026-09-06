"""Input schemas for the two tools.

These models are the single source of truth: `tools/list` advertises the JSON
Schema generated from them, and `tools/call` validates against the same models.
There is no hand-written schema that can drift from the runtime check.
"""

from __future__ import annotations

import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Exactly five digits. Anchored, so "cust-1", "CUST-123456" and " CUST-00001"
# are all rejected rather than partially matched.
CUSTOMER_ID_PATTERN = r"^CUST-\d{5}$"

MAX_REFUND_AMOUNT = 100_000.00
MIN_REASON_LENGTH = 10

# An upper bound as well as a lower one. Without it a 5 MB `reason` is accepted,
# serialised twice into the response (once as `TextContent`, once as
# `structuredContent`) for 10 MB out and seconds of single-threaded CPU, and
# then retained in the refund ledger forever - a memory-exhaustion path that
# needs no privileges. The customer id is protected by its anchored pattern; the
# same discipline belongs on the field next to it.
#
# 2,000 characters is roughly a page of prose. It is far more than any real
# refund justification and far less than anything that costs the server
# noticeable work.
MAX_REASON_LENGTH = 2_000

MAX_IDEMPOTENCY_KEY_LENGTH = 128

# Characters that occupy no visual space: control (Cc), format (Cf, which is
# where U+200B/U+200C/U+200D/U+FEFF live), and the three whitespace separator
# categories. `str.strip()` removes ASCII whitespace and nothing else, so on its
# own it would let ten zero-width spaces count as a "meaningful" reason for
# moving money.
#
# Only these are discounted. Letters, digits, punctuation, symbols and emoji all
# count, in every script - the check must not quietly become "must be English".
_INVISIBLE_CATEGORIES = frozenset({"Cc", "Cf", "Zs", "Zl", "Zp"})


def visible_length(value: str) -> int:
    """Characters that a human would actually see."""
    return sum(1 for character in value if unicodedata.category(character) not in _INVISIBLE_CATEGORIES)


class StrictModel(BaseModel):
    """Base config shared by every tool input.

    `extra="forbid"` matters as much as the field rules: a caller that sends
    `custmer_id` should get a loud rejection, not a silently ignored field and a
    confusing "missing required field" alongside it.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class GetCustomerRecordInput(StrictModel):
    customer_id: str = Field(
        pattern=CUSTOMER_ID_PATTERN,
        description="Customer identifier, formatted as CUST-XXXXX (exactly five digits).",
        examples=["CUST-00042"],
    )


class TriggerRefundInput(StrictModel):
    customer_id: str = Field(
        pattern=CUSTOMER_ID_PATTERN,
        description="Customer identifier, formatted as CUST-XXXXX (exactly five digits).",
        examples=["CUST-00042"],
    )
    amount: float = Field(
        gt=0,
        le=MAX_REFUND_AMOUNT,
        # Without this, pydantic accepts inf/NaN for a float field. The bounds
        # above happen to reject them too, but only incidentally - every
        # comparison against NaN is False, so it is the `le` check failing
        # closed rather than an actual finiteness test. Say what we mean.
        allow_inf_nan=False,
        description=(
            "Refund amount in USD. Must be strictly positive, finite, at most "
            f"{MAX_REFUND_AMOUNT:,.2f}, and no more than 2 decimal places."
        ),
        examples=[49.99],
    )
    reason: str = Field(
        max_length=MAX_REASON_LENGTH,
        description=(
            f"Why the refund is being issued. At least {MIN_REASON_LENGTH} visible "
            f"characters (whitespace, control and zero-width characters do not "
            f"count), and at most {MAX_REASON_LENGTH} characters in total."
        ),
        examples=["Duplicate charge on the customer's April invoice."],
    )
    idempotency_key: str | None = Field(
        default=None,
        max_length=MAX_IDEMPOTENCY_KEY_LENGTH,
        description=(
            "Optional caller-supplied identifier for this refund. Retrying with the "
            "same key returns the original refund instead of issuing a second one. "
            "When omitted, a request repeating the same customer, amount and reason "
            "is treated as a retry of the same refund rather than a new one."
        ),
        examples=["refund-2026-09-06-a41f"],
    )

    @field_validator("amount", mode="before")
    @classmethod
    def _reject_non_numeric(cls, value: Any) -> Any:
        """Reject the types pydantic would otherwise happily coerce to a float.

        `True` is an `int` in Python, so without this it validates as 1.0 and a
        boolean silently becomes a one-dollar refund. Strings are refused too:
        "49.99" from a client is a schema violation worth surfacing, not
        something to quietly parse.
        """
        if isinstance(value, bool):
            raise ValueError("must be a number, not a boolean")
        if isinstance(value, str):
            raise ValueError("must be a JSON number, not a string")
        return value

    @field_validator("amount", mode="after")
    @classmethod
    def _two_decimals(cls, value: float) -> float:
        """Reject sub-cent precision: 10.001 is not a dollar amount."""
        try:
            exponent = Decimal(str(value)).as_tuple().exponent
        except InvalidOperation:  # pragma: no cover - unreachable for finite floats
            raise ValueError("must be a decimal number") from None
        if isinstance(exponent, int) and -exponent > 2:
            raise ValueError("must have at most 2 decimal places (currency precision)")
        return value

    @field_validator("reason", mode="after")
    @classmethod
    def _meaningful_reason(cls, value: str) -> str:
        """Length is measured in *visible* characters, so padding is not a reason.

        A plain `min_length=10` accepts twelve spaces. Measuring after
        `str.strip()` covers that for ASCII and nothing else: ten U+200B
        zero-width spaces would still pass, putting a semantically empty
        justification on a money-moving action in the permanent ledger.

        The rule is therefore stated over Unicode categories rather than over a
        list of remembered characters, so U+00A0, U+3000, U+2028, U+FEFF and the
        zero-width joiners are all covered by the same sentence. Text in any
        script counts normally: `visible_length` discounts only characters that
        render as nothing.

        The stripped value is what gets returned and stored, so the ledger never
        holds padded whitespace.
        """
        stripped = value.strip()
        visible = visible_length(stripped)
        if visible < MIN_REASON_LENGTH:
            raise ValueError(
                f"must be at least {MIN_REASON_LENGTH} visible characters "
                f"(got {visible}; whitespace, control and zero-width characters do not count)"
            )
        return stripped

    @field_validator("idempotency_key", mode="after")
    @classmethod
    def _usable_idempotency_key(cls, value: str | None) -> str | None:
        """An empty or invisible key is a client bug, and silently treating it as
        "no key" would turn an intended replay guard into no guard at all."""
        if value is None:
            return None
        if visible_length(value.strip()) == 0:
            raise ValueError("must contain at least one visible character when provided")
        return value.strip()
