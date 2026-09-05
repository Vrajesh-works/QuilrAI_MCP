"""Every malformed input must come back as JSON-RPC -32602, not a result.

Driven through a real client session so these assert the *wire* behaviour, not
just that Pydantic raises.
"""

from __future__ import annotations

import pytest
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_PARAMS, METHOD_NOT_FOUND

from conftest import connected_session

VALID_REASON = "Duplicate charge on the April invoice."

# (label, tool, arguments, field expected to be blamed)
BAD_INPUTS = [
    # --- customer_id format ---
    ("lowercase prefix", "get_customer_record", {"customer_id": "cust-00001"}, "customer_id"),
    ("too few digits", "get_customer_record", {"customer_id": "CUST-0001"}, "customer_id"),
    ("too many digits", "get_customer_record", {"customer_id": "CUST-000001"}, "customer_id"),
    ("non-digit body", "get_customer_record", {"customer_id": "CUST-ABCDE"}, "customer_id"),
    ("leading space", "get_customer_record", {"customer_id": " CUST-00001"}, "customer_id"),
    ("trailing newline", "get_customer_record", {"customer_id": "CUST-00001\n"}, "customer_id"),
    ("missing prefix", "get_customer_record", {"customer_id": "00001"}, "customer_id"),
    ("empty string", "get_customer_record", {"customer_id": ""}, "customer_id"),
    ("null", "get_customer_record", {"customer_id": None}, "customer_id"),
    ("integer not string", "get_customer_record", {"customer_id": 42}, "customer_id"),
    ("missing field", "get_customer_record", {}, "customer_id"),
    ("unexpected extra field", "get_customer_record", {"customer_id": "CUST-00042", "admin": True}, "admin"),
    # --- refund amount ---
    ("zero amount", "trigger_refund", {"customer_id": "CUST-00042", "amount": 0, "reason": VALID_REASON}, "amount"),
    ("negative amount", "trigger_refund", {"customer_id": "CUST-00042", "amount": -5.0, "reason": VALID_REASON}, "amount"),
    ("NaN amount", "trigger_refund", {"customer_id": "CUST-00042", "amount": float("nan"), "reason": VALID_REASON}, "amount"),
    ("infinite amount", "trigger_refund", {"customer_id": "CUST-00042", "amount": float("inf"), "reason": VALID_REASON}, "amount"),
    ("above ceiling", "trigger_refund", {"customer_id": "CUST-00042", "amount": 100_000.01, "reason": VALID_REASON}, "amount"),
    ("boolean amount", "trigger_refund", {"customer_id": "CUST-00042", "amount": True, "reason": VALID_REASON}, "amount"),
    ("string amount", "trigger_refund", {"customer_id": "CUST-00042", "amount": "10.00", "reason": VALID_REASON}, "amount"),
    ("sub-cent precision", "trigger_refund", {"customer_id": "CUST-00042", "amount": 10.001, "reason": VALID_REASON}, "amount"),
    ("missing amount", "trigger_refund", {"customer_id": "CUST-00042", "reason": VALID_REASON}, "amount"),
    # --- refund reason ---
    ("reason too short", "trigger_refund", {"customer_id": "CUST-00042", "amount": 5.0, "reason": "too short"}, "reason"),
    ("reason all whitespace", "trigger_refund", {"customer_id": "CUST-00042", "amount": 5.0, "reason": " " * 12}, "reason"),
    ("reason empty", "trigger_refund", {"customer_id": "CUST-00042", "amount": 5.0, "reason": ""}, "reason"),
    ("reason not a string", "trigger_refund", {"customer_id": "CUST-00042", "amount": 5.0, "reason": 12345678901}, "reason"),
    ("missing reason", "trigger_refund", {"customer_id": "CUST-00042", "amount": 5.0}, "reason"),
]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool", "arguments", "blamed_field"),
    [pytest.param(t, a, f, id=label) for label, t, a, f in BAD_INPUTS],
)
async def test_invalid_input_is_a_jsonrpc_error(tool: str, arguments: dict, blamed_field: str) -> None:
    async with connected_session() as session:
        with pytest.raises(MCPError) as caught:
            await session.call_tool(tool, arguments)

    error = caught.value
    assert error.code == INVALID_PARAMS, f"expected -32602, got {error.code}"
    # The payload must name the offending field, or the caller cannot fix it.
    blamed = {issue["field"] for issue in error.data["issues"]}
    assert blamed_field in blamed, f"expected {blamed_field!r} to be blamed, got {blamed}"


@pytest.mark.anyio
async def test_unknown_tool_is_method_not_found() -> None:
    async with connected_session() as session:
        with pytest.raises(MCPError) as caught:
            await session.call_tool("admin_drop_database", {})

    assert caught.value.code == METHOD_NOT_FOUND
    assert "get_customer_record" in caught.value.data["available_tools"]


@pytest.mark.anyio
async def test_error_payload_does_not_echo_caller_input() -> None:
    """Rejections must not reflect the submitted values back out.

    Arguments can hold real customer data; error payloads propagate into client
    logs and model context, so they carry field names and reasons only.
    """
    secret = "CUST-DEADBEEF-secret-value"
    async with connected_session() as session:
        with pytest.raises(MCPError) as caught:
            await session.call_tool("get_customer_record", {"customer_id": secret})

    serialized = repr(caught.value.data) + caught.value.message
    assert secret not in serialized


@pytest.mark.anyio
async def test_valid_input_at_the_boundaries_is_accepted() -> None:
    """The rules must not be so tight that legitimate calls fail."""
    async with connected_session() as session:
        # Integer amount: JSON has no float/int distinction, so 100 is valid.
        result = await session.call_tool(
            "trigger_refund",
            {"customer_id": "CUST-00042", "amount": 100, "reason": "x" * 10},
        )
        assert result.is_error is False
        assert result.structured_content["amount"] == 100.0

        # Exactly at the minimum reason length, and two decimal places.
        result = await session.call_tool(
            "trigger_refund",
            {"customer_id": "CUST-00042", "amount": 0.01, "reason": "1234567890"},
        )
        assert result.is_error is False


@pytest.mark.anyio
async def test_reason_is_stored_stripped() -> None:
    async with connected_session() as session:
        result = await session.call_tool(
            "trigger_refund",
            {"customer_id": "CUST-00042", "amount": 1.0, "reason": "   padded but long enough   "},
        )
    assert result.structured_content["reason"] == "padded but long enough"
