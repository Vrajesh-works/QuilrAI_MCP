"""Walk the gateway through the scenarios that matter, printing each exchange.

    uv run python scripts/demo.py

Runs entirely in process - no ports bound - and reports, for every request,
whether the downstream server was contacted. That last column is the point:
a blocked call must show `downstream: not contacted`.
"""

from __future__ import annotations

import json

import anyio
import httpx

from mcp_gateway.app import create_app
from mcp_gateway.config import Config
from mock_downstream import app as mock

ADMIN = "admin-token-abc123"
VIEWER = "viewer-token-xyz789"

SCENARIOS: list[tuple[str, str | None, object]] = [
    (
        "viewer lists tools -> forwarded transparently, admin tools visible",
        VIEWER,
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    ),
    (
        "viewer calls an ordinary tool -> forwarded",
        VIEWER,
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "search_docs", "arguments": {"query": "vpn"}}},
    ),
    (
        "viewer calls admin_reset_key -> BLOCKED with -32001",
        VIEWER,
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "admin_reset_key", "arguments": {"tenant": "acme"}}},
    ),
    (
        "admin calls admin_reset_key -> allowed",
        ADMIN,
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "admin_reset_key", "arguments": {"tenant": "acme"}}},
    ),
    (
        "viewer hides an admin call inside a batch -> only that element is blocked",
        VIEWER,
        [
            {"jsonrpc": "2.0", "id": 5, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "admin_delete_tenant", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "get_ticket", "arguments": {"ticket_id": "T-1"}}},
        ],
    ),
    (
        "viewer tries casing to dodge the prefix -> still blocked",
        VIEWER,
        {"jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {"name": "ADMIN_reset_key", "arguments": {}}},
    ),
    (
        "viewer sends a non-string tool name -> fails closed with -32602",
        VIEWER,
        {"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": ["admin_reset_key"]}},
    ),
    ("no credentials -> 401 before anything is parsed", None, {"jsonrpc": "2.0", "id": 10, "method": "tools/list"}),
]


def summarize(payload) -> str:
    items = payload if isinstance(payload, list) else [payload]
    parts = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if "error" in item:
            parts.append(f"id={item.get('id')} ERROR {item['error']['code']} {item['error']['message']}")
        else:
            parts.append(f"id={item.get('id')} ok")
    return "; ".join(parts) or "(no content)"


async def main() -> None:
    config = Config(
        downstream_url="http://downstream.test/mcp", request_timeout_seconds=5.0, filter_tools_list=False
    )
    app = create_app(config)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mock.create_app())) as downstream:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
        ) as client:
            async with app.router.lifespan_context(app):
                app.state.http_client = downstream

                for label, token, payload in SCENARIOS:
                    mock.reset_received()
                    headers = {"Authorization": f"Bearer {token}"} if token else {}

                    print("=" * 78)
                    print(f"  {label}")
                    print("=" * 78)
                    print(f"--> {json.dumps(payload)[:150]}")

                    response = await client.post("/mcp", json=payload, headers=headers)

                    body = response.json() if response.content else None
                    print(f"<-- HTTP {response.status_code}  {summarize(body) if body else '(204 No Content)'}")

                    forwarded = [
                        message.get("params", {}).get("name") or message.get("method")
                        for message in mock.received
                    ]
                    print(f"    downstream: {forwarded if forwarded else 'not contacted'}")
                    print()


if __name__ == "__main__":
    anyio.run(main)
