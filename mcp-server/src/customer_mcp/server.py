"""The MCP server: tool declarations, validation, and dispatch.

Built on the low-level `Server` rather than `FastMCP` on purpose. FastMCP turns
a handler exception into a `CallToolResult` with `isError: true` - a successful
JSON-RPC *response*. Invalid input must instead be rejected with standard
JSON-RPC *error codes*, which means raising `MCPError` and letting the dispatcher
serialise it. The low-level server is what gives that control.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import mcp_types as types
from mcp.server.lowlevel import Server
from pydantic import BaseModel, ValidationError

from customer_mcp import errors, store
from customer_mcp.schemas import GetCustomerRecordInput, TriggerRefundInput

logger = logging.getLogger(__name__)

SERVER_NAME = "customer-mcp"
SERVER_VERSION = "0.1.0"


def _text_result(payload: dict[str, Any], *, is_error: bool = False) -> types.CallToolResult:
    """Return JSON as text content, and as structured content where supported.

    The text block keeps older clients working; `structuredContent` lets modern
    ones consume the result without re-parsing a string.
    """
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload, indent=2))],
        structuredContent=payload,
        isError=is_error,
    )


async def _get_customer_record(args: GetCustomerRecordInput) -> dict[str, Any]:
    return store.get_customer(args.customer_id)


async def _trigger_refund(args: TriggerRefundInput) -> dict[str, Any]:
    return store.create_refund(args.customer_id, args.amount, args.reason)


class _ToolSpec(BaseModel):
    """Ties a tool's advertised schema to the model that enforces it."""

    model_config = {"arbitrary_types_allowed": True}

    name: str
    title: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[Any], Awaitable[dict[str, Any]]]
    read_only: bool
    destructive: bool


TOOLS: dict[str, _ToolSpec] = {
    "get_customer_record": _ToolSpec(
        name="get_customer_record",
        title="Get customer record",
        description=(
            "Fetch the billing record for a single customer: name, email, account "
            "status, lifetime spend and remaining refundable balance. Read-only."
        ),
        input_model=GetCustomerRecordInput,
        handler=_get_customer_record,
        read_only=True,
        destructive=False,
    ),
    "trigger_refund": _ToolSpec(
        name="trigger_refund",
        title="Trigger a refund",
        description=(
            "Issue a refund against a customer's refundable balance. Fails if the "
            "customer does not exist, the account is not active, or the amount "
            "exceeds the remaining refundable balance. This moves money - confirm "
            "the amount with the user before calling it."
        ),
        input_model=TriggerRefundInput,
        handler=_trigger_refund,
        read_only=False,
        destructive=True,
    ),
}


async def handle_list_tools(
    ctx: Any, params: types.PaginatedRequestParams | None
) -> types.ListToolsResult:
    """Advertise both tools, with schemas generated from the Pydantic models."""
    logger.debug("tools/list requested")
    return types.ListToolsResult(
        tools=[
            types.Tool(
                name=spec.name,
                title=spec.title,
                description=spec.description,
                inputSchema=spec.input_model.model_json_schema(),
                annotations=types.ToolAnnotations(
                    title=spec.title,
                    readOnlyHint=spec.read_only,
                    destructiveHint=spec.destructive,
                    idempotentHint=spec.read_only,
                    openWorldHint=False,
                ),
            )
            for spec in TOOLS.values()
        ]
    )


async def handle_call_tool(
    ctx: Any, params: types.CallToolRequestParams
) -> types.CallToolResult:
    """Validate, dispatch, and map every failure onto the right channel."""
    spec = TOOLS.get(params.name)
    if spec is None:
        # Protocol-level: this tool will never exist, so it is a JSON-RPC error.
        raise errors.unknown_tool(params.name, list(TOOLS))

    try:
        args = spec.input_model.model_validate(params.arguments or {})
    except ValidationError as exc:
        # Protocol-level: the caller broke the advertised contract.
        raise errors.invalid_params(spec.name, exc) from None

    logger.info("Executing tool %r", spec.name)
    try:
        payload = await spec.handler(args)
    except store.DomainError as exc:
        # Business-level: a valid call with a negative answer. The model should
        # see this and reason about it, so it comes back as a result.
        logger.info("Tool %r refused: %s", spec.name, exc.code)
        return _text_result(
            {"error": exc.code, "message": exc.message, **exc.details},
            is_error=True,
        )
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all boundary
        raise errors.internal_error(spec.name, exc) from None

    return _text_result(payload)


def build_server() -> Server:
    """Construct the server. Kept separate from `__main__` so tests can drive it
    over in-memory streams without spawning a process."""
    return Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        title="Customer Billing MCP",
        instructions=(
            "Look up customer billing records and issue refunds. Customer ids are "
            "formatted CUST-XXXXX. Always read the record before issuing a refund "
            "so the amount can be checked against the refundable balance."
        ),
        on_list_tools=handle_list_tools,
        on_call_tool=handle_call_tool,
    )
