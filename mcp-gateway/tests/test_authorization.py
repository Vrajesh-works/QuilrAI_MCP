"""The core requirement: admin_* tools require the admin role.

The assertion that matters in most of these is not just the -32001 response but
that `received` stays empty. A gateway that forwards first and filters the
response has already let the side effect happen.
"""

from __future__ import annotations

import pytest

from conftest import ADMIN_TOKEN, VIEWER_TOKEN, auth

pytestmark = pytest.mark.asyncio

UNAUTHORIZED = -32001


def call(tool: str, id: int = 1, **arguments) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": id,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }


async def test_viewer_calling_admin_tool_is_rejected(gateway, received):
    response = await gateway.post("/mcp", json=call("admin_reset_key", tenant="acme"), headers=auth(VIEWER_TOKEN))

    body = response.json()
    assert body["error"]["code"] == UNAUTHORIZED
    assert body["error"]["message"] == "Unauthorized Tool Call"
    assert body["id"] == 1, "the rejection must echo the request id so the client can correlate it"
    assert received == [], "the blocked call must never reach the downstream server"


async def test_admin_calling_admin_tool_is_forwarded(gateway, received):
    response = await gateway.post("/mcp", json=call("admin_reset_key", tenant="acme"), headers=auth(ADMIN_TOKEN))

    assert response.status_code == 200
    assert response.json()["result"]["isError"] is False
    assert [message["params"]["name"] for message in received] == ["admin_reset_key"]


async def test_viewer_calling_ordinary_tool_is_forwarded(gateway, received):
    response = await gateway.post("/mcp", json=call("search_docs", query="onboarding"), headers=auth(VIEWER_TOKEN))

    assert response.status_code == 200
    assert "result" in response.json()
    assert len(received) == 1


async def test_tools_list_is_forwarded_transparently_for_every_role(gateway, received):
    """tools/list passes straight through and is not filtered by default."""
    for token in (VIEWER_TOKEN, ADMIN_TOKEN):
        response = await gateway.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, headers=auth(token)
        )
        names = [tool["name"] for tool in response.json()["result"]["tools"]]
        assert "admin_reset_key" in names

    assert len(received) == 2


@pytest.mark.parametrize(
    "tool",
    [
        "admin_reset_key",
        "admin_delete_tenant",
        "admin_",
        "admin_anything_at_all",
    ],
)
async def test_every_admin_prefixed_tool_is_gated(gateway, received, tool):
    response = await gateway.post("/mcp", json=call(tool), headers=auth(VIEWER_TOKEN))

    assert response.json()["error"]["code"] == UNAUTHORIZED
    assert received == []


@pytest.mark.parametrize(
    ("tool", "why"),
    [
        ("ADMIN_reset_key", "uppercase"),
        ("Admin_reset_key", "capitalised"),
        ("aDmIn_reset_key", "mixed case"),
        (" admin_reset_key", "leading space"),
        ("\tadmin_reset_key", "leading tab"),
        ("ａdmin_reset_key", "fullwidth 'a' (U+FF41), NFKC-folds to ASCII"),
        ("ADMIN_ＲＥＳＥＴ", "fullwidth body"),
    ],
)
async def test_prefix_check_cannot_be_dodged_by_casing_or_unicode(gateway, received, tool, why):
    """The prefix match normalises before comparing, so these do not slip past.

    Fail-closed by design: this would also gate a genuinely unprivileged tool
    named `Admin_helper`. A false denial is visible and fixable; a false allow
    is a silent privilege escalation.
    """
    response = await gateway.post("/mcp", json=call(tool), headers=auth(VIEWER_TOKEN))

    assert response.json()["error"]["code"] == UNAUTHORIZED, f"bypass via {why}"
    assert received == [], f"bypass via {why} reached downstream"


async def test_non_admin_tools_are_not_over_blocked(gateway, received):
    """The fail-closed normalisation must not swallow ordinary tool names."""
    for tool in ("search_docs", "get_ticket", "administrate_nothing", "my_admin_tool"):
        response = await gateway.post("/mcp", json=call(tool), headers=auth(VIEWER_TOKEN))
        assert "error" not in response.json() or response.json()["error"]["code"] != UNAUTHORIZED, tool

    assert len(received) == 4


async def test_non_string_tool_name_is_rejected_not_forwarded(gateway, received):
    """`{"name": ["admin_reset_key"]}` must not slip past a str.startswith check."""
    response = await gateway.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": ["admin_reset_key"]}},
        headers=auth(VIEWER_TOKEN),
    )

    assert response.json()["error"]["code"] == -32602
    assert received == [], "an unreadable tool name must fail closed"


async def test_missing_params_on_tools_call_is_rejected(gateway, received):
    response = await gateway.post(
        "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/call"}, headers=auth(VIEWER_TOKEN)
    )

    assert response.json()["error"]["code"] == -32602
    assert received == []


async def test_blocked_notification_gets_no_response_body(gateway, received):
    """JSON-RPC §4.1: a notification never draws a response, even when refused."""
    response = await gateway.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "admin_reset_key"}},
        headers=auth(VIEWER_TOKEN),
    )

    assert response.status_code == 204
    assert response.content == b""
    assert received == []


async def test_other_methods_pass_through_untouched(gateway, received):
    """The policy governs tools/call only; the rest of MCP is not its business."""
    for method in ("initialize", "ping"):
        response = await gateway.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": method}, headers=auth(VIEWER_TOKEN)
        )
        assert response.status_code == 200

    assert [message["method"] for message in received] == ["initialize", "ping"]
