"""Batch payloads - the usual way a filter like this gets bypassed.

A gateway that reads `payload["method"]` sees nothing at all in
`[{tools/list}, {admin call}]`: the array has no "method" key, so a naive check
finds no admin tool and forwards the whole thing. Every element is inspected
individually here, and only the ones that pass are forwarded.
"""

from __future__ import annotations

import pytest

from conftest import ADMIN_TOKEN, VIEWER_TOKEN, auth

pytestmark = pytest.mark.asyncio

UNAUTHORIZED = -32001


def call(tool: str, id: int) -> dict:
    return {"jsonrpc": "2.0", "id": id, "method": "tools/call", "params": {"name": tool, "arguments": {}}}


def listing(id: int) -> dict:
    return {"jsonrpc": "2.0", "id": id, "method": "tools/list"}


async def test_admin_call_hidden_in_a_batch_is_still_blocked(gateway, received):
    response = await gateway.post(
        "/mcp",
        json=[listing(1), call("admin_reset_key", 2), call("search_docs", 3)],
        headers=auth(VIEWER_TOKEN),
    )

    by_id = {item["id"]: item for item in response.json()}
    assert by_id[2]["error"]["code"] == UNAUTHORIZED
    # The innocent members of the batch still get served.
    assert "result" in by_id[1]
    assert "result" in by_id[3]

    forwarded = [message.get("params", {}).get("name", message["method"]) for message in received]
    assert "admin_reset_key" not in forwarded
    assert forwarded == ["tools/list", "search_docs"]


async def test_batch_of_only_admin_calls_never_contacts_downstream(gateway, received):
    response = await gateway.post(
        "/mcp",
        json=[call("admin_reset_key", 1), call("admin_delete_tenant", 2)],
        headers=auth(VIEWER_TOKEN),
    )

    body = response.json()
    assert isinstance(body, list) and len(body) == 2
    assert all(item["error"]["code"] == UNAUTHORIZED for item in body)
    assert received == [], "downstream must not be contacted when the whole batch is blocked"


async def test_batch_response_stays_a_batch(gateway):
    """A batch request must draw a batch response, even when partly local."""
    response = await gateway.post("/mcp", json=[listing(1), call("admin_reset_key", 2)], headers=auth(VIEWER_TOKEN))
    assert isinstance(response.json(), list)

    response = await gateway.post("/mcp", json=[listing(1)], headers=auth(VIEWER_TOKEN))
    assert isinstance(response.json(), list), "a single-element batch is still a batch"


async def test_single_request_response_is_not_wrapped_in_an_array(gateway):
    response = await gateway.post("/mcp", json=listing(1), headers=auth(VIEWER_TOKEN))
    assert isinstance(response.json(), dict)


async def test_admin_batch_passes_entirely(gateway, received):
    response = await gateway.post(
        "/mcp", json=[call("admin_reset_key", 1), call("search_docs", 2)], headers=auth(ADMIN_TOKEN)
    )

    body = response.json()
    assert all("result" in item for item in body)
    assert len(received) == 2


async def test_every_id_is_accounted_for(gateway):
    """Each request in a batch draws exactly one response, blocked or not."""
    requests = [listing(1), call("admin_reset_key", 2), call("get_ticket", 3), call("admin_delete_tenant", 4)]
    response = await gateway.post("/mcp", json=requests, headers=auth(VIEWER_TOKEN))

    ids = sorted(item["id"] for item in response.json())
    assert ids == [1, 2, 3, 4]


async def test_blocked_notifications_in_a_batch_produce_no_entries(gateway, received):
    """Notifications draw no response; the requests around them still do."""
    response = await gateway.post(
        "/mcp",
        json=[
            {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "admin_reset_key"}},  # no id
            listing(7),
        ],
        headers=auth(VIEWER_TOKEN),
    )

    body = response.json()
    assert [item["id"] for item in body] == [7]
    assert [message["method"] for message in received] == ["tools/list"]


async def test_empty_batch_is_an_invalid_request(gateway, received):
    """JSON-RPC §6: an empty array is itself an Invalid Request."""
    response = await gateway.post("/mcp", json=[], headers=auth(VIEWER_TOKEN))

    assert response.json()["error"]["code"] == -32600
    assert received == []


async def test_one_malformed_member_rejects_the_whole_batch(gateway, received):
    """Fail closed: a batch we cannot fully parse is never partly forwarded."""
    response = await gateway.post(
        "/mcp",
        json=[listing(1), {"jsonrpc": "1.0", "id": 2, "method": "tools/list"}],
        headers=auth(VIEWER_TOKEN),
    )

    assert response.json()["error"]["code"] == -32600
    assert received == []
