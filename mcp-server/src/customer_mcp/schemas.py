"""Input schemas for the two tools.

These models are the single source of truth: `tools/list` advertises the JSON
Schema generated from them, and `tools/call` validates against the same models.
There is no hand-written schema that can drift from the runtime check.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Exactly five digits. Anchored, so "cust-1", "CUST-123456" and " CUST-00001"
# are all rejected rather than partially matched.
CUSTOMER_ID_PATTERN = r"^CUST-\d{5}$"

MAX_REFUND_AMOUNT = 100_000.00
MIN_REASON_LENGTH = 10


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
        description=(
            f"Why the refund is being issued. At least {MIN_REASON_LENGTH} "
            "characters after leading/trailing whitespace is removed."
        ),
        examples=["Duplicate charge on the customer's April invoice."],
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
        """Length is measured after stripping, so 12 spaces is not a reason.

        A plain `min_length=10` would accept it. The stripped value is what gets
        returned and stored, so the ledger never holds padded whitespace.
        """
        stripped = value.strip()
        if len(stripped) < MIN_REASON_LENGTH:
            raise ValueError(
                f"must be at least {MIN_REASON_LENGTH} characters of actual text "
                f"(got {len(stripped)} after trimming whitespace)"
            )
        return stripped
