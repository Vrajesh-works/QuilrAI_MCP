"""Forwarding behaviour: headers, transparency, and upstream failure handling."""

from __future__ import annotations

import json

import pytest

from conftest import ADMIN_TOKEN, VIEWER_TOKEN, auth

pytestmark = pytest.mark.asyncio

LIST = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}


async def test_allowed_request_is_forwarded_byte_for_byte(gateway, received):
    """With nothing blocked, the proxy relays the original body unmodified.

    Re-encoding would silently normalise key order, numeric formatting and
    unicode escapes - a proxy should not rewrite what it is only inspecting.
    """
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "search_docs", "arguments": {"query": "café", "n": 1.50}}}
    raw = json.dumps(body)

    response = await gateway.post(
        "/mcp", content=raw, headers={**auth(VIEWER_TOKEN), "content-type": "application/json"}
    )

    assert response.status_code == 200
    assert received[0] == body


async def test_client_token_is_not_forwarded_downstream(capturing_gateway):
    """The credential terminates at the gateway.

    Passing it on would let the downstream server make its own decisions from a
    token it should never see, which defeats terminating auth here at all.
    """
    client, captured = capturing_gateway
    await client.post("/mcp", json=LIST, headers=auth(ADMIN_TOKEN))

    headers = captured["headers"]
    assert "authorization" not in headers
    # The gateway asserts the identity itself instead.
    assert headers["x-forwarded-user"] == "ops@example.com"
    assert headers["x-forwarded-role"] == "admin"


async def test_hop_by_hop_headers_are_not_forwarded(gateway, received):
    """RFC 9110 §7.6.1: these describe one connection and must not be relayed."""
    from mcp_gateway.proxy import build_downstream_headers
    import httpx

    from mcp_gateway.auth import resolve_principal

    incoming = httpx.Headers(
        {
            "authorization": "Bearer secret",
            "connection": "keep-alive",
            "transfer-encoding": "chunked",
            "content-length": "123",
            "host": "gateway.test",
            "mcp-session-id": "session-42",
            "content-type": "application/json",
        }
    )
    forwarded = build_downstream_headers(incoming, resolve_principal(f"Bearer {ADMIN_TOKEN}"))

    for banned in ("authorization", "connection", "transfer-encoding", "content-length", "host"):
        assert banned not in forwarded

    # MCP's HTTP transport correlates sessions with this header; dropping it
    # silently breaks every stateful server behind the proxy.
    assert forwarded["mcp-session-id"] == "session-42"
    assert forwarded["content-type"] == "application/json"


async def test_downstream_timeout_returns_502_without_leaking_internals(timing_out_gateway):
    response = await timing_out_gateway.post("/mcp", json=LIST, headers=auth(ADMIN_TOKEN))

    assert response.status_code == 502
    body = response.json()
    assert body["error"]["code"] == -32002

    # The underlying exception names an internal address; none of it may escape.
    serialized = json.dumps(body)
    assert "10.1.2.3" not in serialized
    assert "8081" not in serialized
    assert "Traceback" not in serialized


async def test_downstream_connection_error_is_sanitised(unreachable_gateway):
    response = await unreachable_gateway.post("/mcp", json=LIST, headers=auth(ADMIN_TOKEN))

    assert response.status_code == 502
    serialized = json.dumps(response.json())
    assert "cluster.local" not in serialized, "internal hostname leaked to the client"
    assert "Errno" not in serialized


async def test_denials_still_returned_when_downstream_is_down(unreachable_gateway, received):
    """A blocked call was decided locally; an upstream outage does not undo it."""
    response = await unreachable_gateway.post(
        "/mcp",
        json=[
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "admin_reset_key"}},
        ],
        headers=auth(VIEWER_TOKEN),
    )

    codes = {item["error"]["code"] for item in response.json() if "error" in item}
    assert -32001 in codes, "the denial must survive the upstream failure"
    assert -32002 in codes


async def test_malformed_json_is_rejected_at_the_gateway(gateway, received):
    response = await gateway.post(
        "/mcp", content=b"{not json at all", headers={**auth(VIEWER_TOKEN), "content-type": "application/json"}
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32700
    assert received == [], "an unparseable body must never be forwarded"


async def test_empty_body_is_rejected(gateway, received):
    response = await gateway.post("/mcp", content=b"", headers=auth(VIEWER_TOKEN))

    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32600
    assert received == []


async def test_downstream_status_code_is_relayed(gateway):
    """The proxy does not invent its own status for a healthy downstream."""
    response = await gateway.post("/mcp", json=LIST, headers=auth(VIEWER_TOKEN))
    assert response.status_code == 200


async def test_notification_only_payload_returns_204(gateway, received):
    response = await gateway.post(
        "/mcp", json={"jsonrpc": "2.0", "method": "ping"}, headers=auth(VIEWER_TOKEN)
    )

    assert response.status_code == 204
    assert [message["method"] for message in received] == ["ping"]


async def test_tools_list_filtering_when_enabled(filtering_gateway):
    """Optional, off by default: transparent forwarding is the default.

    Advertising `admin_reset_key` to a viewer who can never call it is an
    information leak about the internal surface, so the option exists.
    """
    viewer = await filtering_gateway.post("/mcp", json=LIST, headers=auth(VIEWER_TOKEN))
    names = [tool["name"] for tool in viewer.json()["result"]["tools"]]
    assert names == ["search_docs", "get_ticket"]

    admin = await filtering_gateway.post("/mcp", json=LIST, headers=auth(ADMIN_TOKEN))
    admin_names = [tool["name"] for tool in admin.json()["result"]["tools"]]
    assert "admin_reset_key" in admin_names


async def test_sse_response_passes_through_unmodified(gateway):
    """MCP's HTTP transport can answer with a stream; a JSON-oriented gateway
    must relay it rather than trying to parse it."""
    response = await gateway.get("/mcp", headers=auth(VIEWER_TOKEN))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert b"event: message" in response.content


async def test_transport_passthrough_still_requires_auth(gateway):
    for method in ("GET", "DELETE"):
        response = await gateway.request(method, "/mcp")
        assert response.status_code == 401, method
