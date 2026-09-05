"""Tool discovery, happy paths, and the domain-refusal channel."""

from __future__ import annotations

import pytest

from conftest import connected_session
from customer_mcp.schemas import GetCustomerRecordInput, TriggerRefundInput

VALID_REASON = "Duplicate charge on the April invoice."


@pytest.mark.anyio
async def test_list_tools_advertises_both_tools_with_generated_schemas() -> None:
    async with connected_session() as session:
        result = await session.list_tools()

    tools = {tool.name: tool for tool in result.tools}
    assert set(tools) == {"get_customer_record", "trigger_refund"}

    # The advertised schema must be the one that is actually enforced, or a
    # client can build a call that validates locally and fails on the server.
    assert tools["get_customer_record"].input_schema == GetCustomerRecordInput.model_json_schema()
    assert tools["trigger_refund"].input_schema == TriggerRefundInput.model_json_schema()

    # Unknown fields are rejected, and clients should be told so up front.
    assert tools["trigger_refund"].input_schema["additionalProperties"] is False

    # A refund moves money; the hint is what lets a client gate on confirmation.
    assert tools["trigger_refund"].annotations.destructive_hint is True
    assert tools["get_customer_record"].annotations.read_only_hint is True


@pytest.mark.anyio
async def test_get_customer_record_returns_the_record() -> None:
    async with connected_session() as session:
        result = await session.call_tool("get_customer_record", {"customer_id": "CUST-00042"})

    assert result.is_error is False
    record = result.structured_content
    assert record["customer_id"] == "CUST-00042"
    assert record["name"] == "Ada Lovelace"
    assert record["refundable_balance"] == 320.00


@pytest.mark.anyio
async def test_trigger_refund_succeeds_and_decrements_balance() -> None:
    async with connected_session() as session:
        result = await session.call_tool(
            "trigger_refund",
            {"customer_id": "CUST-00042", "amount": 120.00, "reason": VALID_REASON},
        )
        assert result.is_error is False
        refund = result.structured_content
        assert refund["status"] == "issued"
        assert refund["amount"] == 120.00
        assert refund["remaining_refundable_balance"] == 200.00
        assert refund["refund_id"].startswith("RFND-")

        # The decrement must be visible to a subsequent read.
        record = await session.call_tool("get_customer_record", {"customer_id": "CUST-00042"})
        assert record.structured_content["refundable_balance"] == 200.00


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("label", "tool", "arguments", "expected_code"),
    [
        (
            "unknown customer on read",
            "get_customer_record",
            {"customer_id": "CUST-55555"},
            "customer_not_found",
        ),
        (
            "unknown customer on refund",
            "trigger_refund",
            {"customer_id": "CUST-55555", "amount": 10.0, "reason": VALID_REASON},
            "customer_not_found",
        ),
        (
            "frozen account",
            "trigger_refund",
            {"customer_id": "CUST-01337", "amount": 10.0, "reason": VALID_REASON},
            "account_not_active",
        ),
        (
            "closed account",
            "trigger_refund",
            {"customer_id": "CUST-99999", "amount": 10.0, "reason": VALID_REASON},
            "account_not_active",
        ),
        (
            "amount exceeds refundable balance",
            "trigger_refund",
            {"customer_id": "CUST-00007", "amount": 10.0, "reason": VALID_REASON},
            "insufficient_refundable_balance",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
async def test_domain_failures_return_iserror_not_a_jsonrpc_error(
    label: str, tool: str, arguments: dict, expected_code: str
) -> None:
    """A valid call with a negative answer is a *result*, not a protocol error.

    The model needs to read "insufficient balance" and adjust; a JSON-RPC error
    would surface as a transport failure it cannot reason about.
    """
    async with connected_session() as session:
        result = await session.call_tool(tool, arguments)

    assert result.is_error is True
    assert result.structured_content["error"] == expected_code
    assert result.content[0].text  # human-readable explanation is present


@pytest.mark.anyio
async def test_refund_exactly_at_the_balance_is_allowed() -> None:
    """The balance check is `>`, not `>=` - draining the balance is legitimate."""
    async with connected_session() as session:
        result = await session.call_tool(
            "trigger_refund",
            {"customer_id": "CUST-00042", "amount": 320.00, "reason": VALID_REASON},
        )

    assert result.is_error is False
    assert result.structured_content["remaining_refundable_balance"] == 0.0


@pytest.mark.anyio
async def test_balance_cannot_be_overdrawn_by_repeated_refunds() -> None:
    async with connected_session() as session:
        first = await session.call_tool(
            "trigger_refund",
            {"customer_id": "CUST-00042", "amount": 300.00, "reason": VALID_REASON},
        )
        assert first.is_error is False

        second = await session.call_tool(
            "trigger_refund",
            {"customer_id": "CUST-00042", "amount": 100.00, "reason": VALID_REASON},
        )

    assert second.is_error is True
    assert second.structured_content["error"] == "insufficient_refundable_balance"
    assert second.structured_content["refundable_balance"] == 20.00
