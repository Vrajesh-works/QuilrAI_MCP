"""A mock downstream MCP server, speaking JSON-RPC over HTTP POST.

Deliberately trusting: it performs no authorization of its own. That is what
makes it useful for testing the gateway - if a privileged call reaches this
server, it executes, so any test asserting a block is really asserting the
gateway blocked it rather than that nothing happened to work.

It records every call it receives, which the gateway tests assert against to
prove blocked calls never arrive.
"""

from __future__ import annotations

from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

TOOLS = [
    {
        "name": "search_docs",
        "description": "Search the internal documentation.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_ticket",
        "description": "Fetch a support ticket by id.",
        "inputSchema": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}},
            "required": ["ticket_id"],
        },
    },
    {
        "name": "admin_reset_key",
        "description": "Rotate a tenant's API key. Admin only.",
        "inputSchema": {
            "type": "object",
            "properties": {"tenant": {"type": "string"}},
            "required": ["tenant"],
        },
    },
    {
        "name": "admin_delete_tenant",
        "description": "Permanently delete a tenant. Admin only.",
        "inputSchema": {
            "type": "object",
            "properties": {"tenant": {"type": "string"}},
            "required": ["tenant"],
        },
    },
]

# Every message this server has been asked to handle. Tests read it to assert
# that blocked calls never got here.
received: list[dict[str, Any]] = []


def reset_received() -> None:
    received.clear()


def _handle_message(message: dict[str, Any]) -> dict[str, Any] | None:
    received.append(message)

    method = message.get("method")
    message_id = message.get("id")
    is_notification = "id" not in message

    def ok(result: Any) -> dict[str, Any] | None:
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": message_id, "result": result}

    def fail(code: int, text: str) -> dict[str, Any] | None:
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": text}}

    if method == "initialize":
        return ok(
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "mock-downstream", "version": "0.1.0"},
            }
        )

    if method == "tools/list":
        return ok({"tools": TOOLS})

    if method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        known = {tool["name"] for tool in TOOLS}
        if name not in known:
            return fail(-32601, f"Unknown tool: {name!r}")
        return ok(
            {
                "content": [{"type": "text", "text": f"executed {name} with {arguments}"}],
                "isError": False,
            }
        )

    if method == "ping":
        return ok({})

    return fail(-32601, f"Method not found: {method!r}")


async def handle_post(request: Request) -> Response:
    document = await request.json()

    if isinstance(document, list):
        responses = [response for item in document if (response := _handle_message(item)) is not None]
        if not responses:
            return Response(status_code=204)
        return JSONResponse(responses)

    response = _handle_message(document)
    if response is None:
        return Response(status_code=204)
    return JSONResponse(response)


async def handle_received(request: Request) -> Response:
    """Every message this server has been asked to handle.

    A demo/test affordance, not part of the MCP surface - it is what lets a
    caller prove the negative, that a blocked tool call never arrived here.
    A real downstream would not expose its request log.
    """
    return JSONResponse(received)


async def handle_get(request: Request) -> Response:
    """Stands in for the server->client SSE stream of MCP's HTTP transport."""
    return Response(
        content="event: message\ndata: {}\n\n",
        media_type="text/event-stream",
        headers={"mcp-session-id": "mock-session"},
    )


async def handle_delete(request: Request) -> Response:
    return Response(status_code=204)


def create_app() -> Starlette:
    return Starlette(
        routes=[
            Route("/mcp", handle_post, methods=["POST"]),
            Route("/_debug/received", handle_received, methods=["GET"]),
            Route("/mcp", handle_get, methods=["GET"]),
            Route("/mcp", handle_delete, methods=["DELETE"]),
        ]
    )


app = create_app()
