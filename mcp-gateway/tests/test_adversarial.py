"""The gateway bypass harness that two prior audits wrote but never ran.

Both audits marked every gateway finding NOT VERIFIED because command execution
became unavailable before the probe could be invoked. Everything here is
therefore executed rather than reasoned about, and every authorization assertion
checks **what the downstream actually received**, not the response the client
got back. A gateway that forwards and then filters the response has already let
the side effect happen; only `received == []` can tell the two apart.
"""

from __future__ import annotations

import json

import pytest
from conftest import ADMIN_TOKEN, VIEWER_TOKEN, auth
from mcp_gateway import jsonrpc, policy

pytestmark = pytest.mark.asyncio

ADMIN_TOOL = "admin_reset_key"
SAFE_TOOL = "get_customer_record"


def call(tool: str, id: int = 1, method: str = "tools/call", **params) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": {"name": tool, "arguments": {}, **params},
    }


def errors_in(body) -> list[dict]:
    items = body if isinstance(body, list) else [body]
    return [item["error"] for item in items if isinstance(item, dict) and "error" in item]


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------

BAD_CREDENTIALS = [
    pytest.param({}, id="no header"),
    pytest.param({"Authorization": ""}, id="empty header"),
    pytest.param({"Authorization": "Bearer"}, id="scheme only"),
    pytest.param({"Authorization": "Bearer "}, id="empty token"),
    pytest.param({"Authorization": "Bearer    "}, id="whitespace token"),
    pytest.param({"Authorization": "Basic YWRtaW46YWRtaW4="}, id="wrong scheme"),
    pytest.param({"Authorization": "Token admin-token-abc123"}, id="wrong scheme, right token"),
    pytest.param({"Authorization": "admin-token-abc123"}, id="no scheme"),
    pytest.param({"Authorization": "Bearer wrong-token"}, id="unknown token"),
    pytest.param({"Authorization": "Bearer admin-token-abc12"}, id="token prefix"),
    pytest.param({"Authorization": "Bearer admin-token-abc1234"}, id="token with suffix"),
    pytest.param({"Authorization": "Bearer ADMIN-TOKEN-ABC123"}, id="token case changed"),
    pytest.param({"Authorization": "Bearer admin-token-abc123 extra"}, id="trailing junk"),
    # Non-ASCII cannot be expressed as a `str` header value here - httpx refuses
    # to encode one, and RFC 9110 field values are bytes. Starlette decodes them
    # as latin-1, so a high byte on the wire *does* reach `resolve_principal` as
    # a non-ASCII `str`. That is the shape that used to raise TypeError inside
    # `hmac.compare_digest` and turn a 401 into a 500 with a traceback.
    pytest.param({"Authorization": b"Bearer admin\xe9token-abc123"}, id="latin-1 high byte"),
    pytest.param({"Authorization": b"Bearer \xff\xfe\x00\x01"}, id="raw non-utf8 bytes"),
]


@pytest.mark.parametrize("headers", BAD_CREDENTIALS)
async def test_bad_credentials_are_rejected_and_never_forwarded(gateway, received, headers):
    response = await gateway.post("/mcp", json=call(SAFE_TOOL), headers=headers)
    assert response.status_code == 401, headers
    assert response.headers.get("www-authenticate", "").startswith("Bearer")
    assert received == [], "an unauthenticated request reached the downstream server"


@pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER", "BeArEr"])
async def test_the_bearer_scheme_is_case_insensitive_per_rfc_7235(gateway, received, scheme):
    response = await gateway.post(
        "/mcp", json=call(SAFE_TOOL), headers={"Authorization": f"{scheme} {VIEWER_TOKEN}"}
    )
    assert response.status_code == 200
    assert len(received) == 1


async def test_extra_space_between_scheme_and_token_is_accepted(gateway, received):
    """RFC 7235 §2.1 is `1*SP` between the scheme and the credentials, so this
    is a conformant request rather than a bypass attempt. Recorded here because
    it looks like one and someone will otherwise "fix" it into a rejection."""
    response = await gateway.post(
        "/mcp", json=call(SAFE_TOOL), headers={"Authorization": f"Bearer   {VIEWER_TOKEN}"}
    )
    assert response.status_code == 200
    assert len(received) == 1


@pytest.mark.parametrize(
    "token",
    [
        "admin‐token-abc123",  # U+2010 hyphen lookalike
        "admin−token-abc123",  # U+2212 minus sign
        "ａdmin-token-abc123",  # fullwidth a
        "admin-token-abc123​",  # trailing zero width
        "аdmin-token-abc123",  # Cyrillic а
    ],
)
def test_a_unicode_lookalike_token_is_rejected_not_a_crash(token):
    """Directly against `resolve_principal`, because these cannot be put on the
    wire as a `str` header. `hmac.compare_digest` raises TypeError for any
    non-ASCII `str`, so before the fix these produced an unhandled exception -
    a 500 with a traceback - instead of a clean authentication failure."""
    from mcp_gateway.auth import AuthError, resolve_principal

    with pytest.raises(AuthError):
        resolve_principal(f"Bearer {token}")


def test_a_valid_token_still_resolves():
    from mcp_gateway.auth import resolve_principal

    assert resolve_principal(f"Bearer {ADMIN_TOKEN}").role == "admin"
    assert resolve_principal(f"Bearer {VIEWER_TOKEN}").role == "viewer"


async def test_auth_failures_do_not_distinguish_unknown_token_from_wrong_role(gateway):
    """A token oracle would let an attacker confirm a guessed token."""
    unknown = await gateway.post("/mcp", json=call(SAFE_TOOL), headers=auth("no-such-token"))
    malformed = await gateway.post("/mcp", json=call(SAFE_TOOL), headers={"Authorization": "Bearer "})
    assert unknown.status_code == malformed.status_code == 401
    assert "role" not in unknown.text.lower()


async def test_duplicate_authorization_headers_do_not_grant_admin(gateway, received):
    """httpx joins repeated headers with ', ', which must not parse as a token."""
    request = gateway.build_request(
        "POST", "/mcp", json=call(ADMIN_TOOL), headers=[("Authorization", f"Bearer {VIEWER_TOKEN}")]
    )
    request.headers.setdefault("authorization", f"Bearer {ADMIN_TOKEN}")
    response = await gateway.send(request)
    assert response.status_code in (200, 401)
    assert not any(item.get("params", {}).get("name") == ADMIN_TOOL for item in received)


# --------------------------------------------------------------------------
# Authorization - the core property
# --------------------------------------------------------------------------


async def test_a_viewer_may_list_tools(gateway, received):
    response = await gateway.post(
        "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, headers=auth(VIEWER_TOKEN)
    )
    assert response.status_code == 200
    assert len(received) == 1


async def test_a_viewer_may_call_a_safe_tool(gateway, received):
    response = await gateway.post("/mcp", json=call(SAFE_TOOL), headers=auth(VIEWER_TOKEN))
    assert response.status_code == 200
    assert received[0]["params"]["name"] == SAFE_TOOL


async def test_a_viewer_calling_an_admin_tool_gets_32001_and_downstream_sees_nothing(gateway, received):
    response = await gateway.post("/mcp", json=call(ADMIN_TOOL), headers=auth(VIEWER_TOKEN))
    assert response.status_code == 200
    assert errors_in(response.json())[0]["code"] == jsonrpc.UNAUTHORIZED_TOOL_CALL
    assert errors_in(response.json())[0]["message"] == "Unauthorized Tool Call"
    assert received == [], "the privileged call reached the downstream server"


async def test_an_admin_may_call_an_admin_tool(gateway, received):
    response = await gateway.post("/mcp", json=call(ADMIN_TOOL), headers=auth(ADMIN_TOKEN))
    assert response.status_code == 200
    assert received[0]["params"]["name"] == ADMIN_TOOL


# --------------------------------------------------------------------------
# Tool-name bypasses
# --------------------------------------------------------------------------

TOOL_NAME_BYPASSES = [
    pytest.param("admin_reset_key", id="plain"),
    pytest.param("ADMIN_RESET_KEY", id="uppercase"),
    pytest.param("Admin_Reset_Key", id="mixed case"),
    pytest.param("aDmIn_reset_key", id="alternating case"),
    pytest.param(" admin_reset_key", id="leading space"),
    pytest.param("admin_reset_key ", id="trailing space"),
    pytest.param("\tadmin_reset_key", id="leading tab"),
    pytest.param("\nadmin_reset_key", id="leading newline"),
    pytest.param(" admin_reset_key", id="leading nbsp"),
    pytest.param("﻿admin_reset_key", id="leading BOM"),
    pytest.param("ａdmin_reset_key", id="fullwidth a"),
    pytest.param("ＡＤＭＩＮ＿ＲＥＳＥＴ", id="fullwidth uppercase"),
    pytest.param("ADMİN_reset", id="dotted capital I"),
    pytest.param("ADMIN_RESET_KEY​", id="trailing zero width"),
]


@pytest.mark.parametrize("name", TOOL_NAME_BYPASSES)
async def test_tool_name_variants_cannot_smuggle_a_privileged_call(gateway, received, name):
    response = await gateway.post("/mcp", json=call(name), headers=auth(VIEWER_TOKEN))
    assert response.status_code == 200
    assert errors_in(response.json())[0]["code"] == jsonrpc.UNAUTHORIZED_TOOL_CALL, name
    assert received == [], f"{name!r} reached the downstream server"


NON_STRING_NAMES = [
    pytest.param(["admin_reset_key"], id="array"),
    pytest.param({"toString": "admin_reset_key"}, id="object"),
    pytest.param(None, id="null"),
    pytest.param(42, id="number"),
    pytest.param(True, id="boolean"),
]


@pytest.mark.parametrize("name", NON_STRING_NAMES)
async def test_a_non_string_tool_name_fails_closed(gateway, received, name):
    """Without a readable name there is no way to know if the call is
    privileged, so it cannot be allowed."""
    response = await gateway.post("/mcp", json=call(name), headers=auth(VIEWER_TOKEN))
    assert errors_in(response.json())[0]["code"] == jsonrpc.INVALID_PARAMS
    assert received == []


async def test_a_missing_params_object_fails_closed(gateway, received):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call"}
    response = await gateway.post("/mcp", json=body, headers=auth(VIEWER_TOKEN))
    assert errors_in(response.json())[0]["code"] == jsonrpc.INVALID_PARAMS
    assert received == []


# --------------------------------------------------------------------------
# GW-2 - method-name bypasses
# --------------------------------------------------------------------------

METHOD_VARIANTS = ["TOOLS/CALL", "tools/Call", "Tools/Call", "tools/call ", " tools/call", "tools／call"]


@pytest.mark.parametrize("method", METHOD_VARIANTS)
async def test_method_casing_cannot_skip_the_tool_check(gateway, received, method):
    """These all used to bypass the policy entirely and be forwarded verbatim:
    tool names were normalised and the method next to them was compared with
    `!=`."""
    response = await gateway.post("/mcp", json=call(ADMIN_TOOL, method=method), headers=auth(VIEWER_TOKEN))
    codes = [error["code"] for error in errors_in(response.json())]
    assert codes and codes[0] in (jsonrpc.UNAUTHORIZED_TOOL_CALL, jsonrpc.METHOD_NOT_FOUND), method
    assert received == [], f"{method!r} reached the downstream server"


async def test_an_unknown_method_is_not_forwarded(gateway, received):
    body = {"jsonrpc": "2.0", "id": 1, "method": "admin/dangerous_thing"}
    response = await gateway.post("/mcp", json=body, headers=auth(VIEWER_TOKEN))
    assert errors_in(response.json())[0]["code"] == jsonrpc.METHOD_NOT_FOUND
    assert received == []


@pytest.mark.parametrize("method", sorted(policy.KNOWN_METHODS))
async def test_every_allowlisted_method_is_still_forwarded(gateway, received, method):
    """The allowlist must not break the protocol it is protecting."""
    body = {"jsonrpc": "2.0", "id": 1, "method": method}
    if method == "tools/call":
        # The one allowlisted method that carries a further check: without a
        # `params.name` the policy fails closed, which is correct and is covered
        # by `test_a_missing_params_object_fails_closed`.
        body["params"] = {"name": SAFE_TOOL, "arguments": {}}
    await gateway.post("/mcp", json=body, headers=auth(ADMIN_TOKEN))
    assert len(received) == 1, f"{method} was blocked"


# --------------------------------------------------------------------------
# GW-1 - request smuggling via duplicate keys
# --------------------------------------------------------------------------

DUPLICATE_KEY_BODIES = [
    pytest.param(
        '{"jsonrpc":"2.0","id":1,"method":"tools/call",'
        '"params":{"name":"admin_reset_key","name":"get_customer_record"}}',
        id="privileged name first",
    ),
    pytest.param(
        '{"jsonrpc":"2.0","id":1,"method":"tools/call",'
        '"params":{"name":"get_customer_record","name":"admin_reset_key"}}',
        id="privileged name second",
    ),
    pytest.param(
        '{"jsonrpc":"2.0","id":1,"method":"tools/list","method":"tools/call",'
        '"params":{"name":"admin_reset_key"}}',
        id="duplicate method",
    ),
    pytest.param(
        '{"jsonrpc":"2.0","id":1,"method":"tools/call",'
        '"params":{"name":"admin_reset_key"},"params":{"name":"get_customer_record"}}',
        id="duplicate params",
    ),
]


@pytest.mark.parametrize("body", DUPLICATE_KEY_BODIES)
async def test_duplicate_json_keys_are_rejected_rather_than_relayed(gateway, received, body):
    """The gateway authorizes on its own parse but relays the original bytes.
    Two parsers reading the same bytes differently is an authorization bypass
    waiting for the downstream to be reimplemented in another language."""
    response = await gateway.post(
        "/mcp", content=body.encode(), headers={**auth(VIEWER_TOKEN), "Content-Type": "application/json"}
    )
    assert response.status_code == 400, body
    assert errors_in(response.json())[0]["code"] == jsonrpc.INVALID_REQUEST
    assert received == [], "an ambiguous payload reached the downstream server"


async def test_duplicate_keys_are_rejected_for_admins_too(gateway, received):
    """The rule is about ambiguity, not about privilege."""
    body = '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"a","name":"b"}}'
    response = await gateway.post(
        "/mcp", content=body.encode(), headers={**auth(ADMIN_TOKEN), "Content-Type": "application/json"}
    )
    assert response.status_code == 400
    assert received == []


# --------------------------------------------------------------------------
# Batch smuggling
# --------------------------------------------------------------------------


async def test_an_admin_call_hidden_in_a_batch_is_blocked(gateway, received):
    batch = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        call(ADMIN_TOOL, id=2),
        {"jsonrpc": "2.0", "id": 3, "method": "ping"},
    ]
    response = await gateway.post("/mcp", json=batch, headers=auth(VIEWER_TOKEN))
    assert response.status_code == 200
    codes = [error["code"] for error in errors_in(response.json())]
    assert jsonrpc.UNAUTHORIZED_TOOL_CALL in codes
    forwarded = [item.get("params", {}).get("name") for item in received]
    assert ADMIN_TOOL not in forwarded, "the privileged element of the batch was forwarded"


async def test_a_batch_of_only_privileged_calls_never_reaches_downstream(gateway, received):
    batch = [call(ADMIN_TOOL, id=1), call("ADMIN_other", id=2)]
    response = await gateway.post("/mcp", json=batch, headers=auth(VIEWER_TOKEN))
    assert len(errors_in(response.json())) == 2
    assert received == []


async def test_one_malformed_batch_element_rejects_the_whole_batch(gateway, received):
    batch = [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, {"not": "a request"}]
    response = await gateway.post("/mcp", json=batch, headers=auth(VIEWER_TOKEN))
    assert response.status_code == 400
    assert received == []


async def test_an_empty_batch_is_an_invalid_request(gateway, received):
    response = await gateway.post("/mcp", json=[], headers=auth(VIEWER_TOKEN))
    assert response.status_code == 400
    assert errors_in(response.json())[0]["code"] == jsonrpc.INVALID_REQUEST
    assert received == []


async def test_a_blocked_notification_draws_no_response_body(gateway, received):
    """JSON-RPC §4.1 is unconditional: a notification gets no response, even
    when it is refused."""
    notification = {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": ADMIN_TOOL}}
    response = await gateway.post("/mcp", json=notification, headers=auth(VIEWER_TOKEN))
    assert response.status_code == 204
    assert response.content == b""
    assert received == []


# --------------------------------------------------------------------------
# Malformed JSON-RPC
# --------------------------------------------------------------------------

MALFORMED = [
    pytest.param(b"", 400, id="empty body"),
    pytest.param(b"   ", 400, id="whitespace only"),
    pytest.param(b"{ not json", 400, id="malformed json"),
    pytest.param(b'"a string"', 400, id="json string"),
    pytest.param(b"42", 400, id="json number"),
    pytest.param(b"null", 400, id="json null"),
    pytest.param(b"true", 400, id="json bool"),
    pytest.param(b'{"id":1,"method":"tools/list"}', 400, id="missing jsonrpc"),
    pytest.param(b'{"jsonrpc":"1.0","id":1,"method":"tools/list"}', 400, id="wrong version"),
    pytest.param(b'{"jsonrpc":2.0,"id":1,"method":"tools/list"}', 400, id="numeric version"),
    pytest.param(b'{"jsonrpc":"2.0","id":1}', 400, id="missing method"),
    pytest.param(b'{"jsonrpc":"2.0","id":1,"method":42}', 400, id="numeric method"),
    pytest.param(b'{"jsonrpc":"2.0","id":1,"method":""}', 400, id="empty method"),
    pytest.param(b'{"jsonrpc":"2.0","id":true,"method":"ping"}', 400, id="bool id"),
    pytest.param(b'{"jsonrpc":"2.0","id":{"a":1},"method":"ping"}', 400, id="object id"),
    pytest.param(b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":"x"}', 400, id="string params"),
    pytest.param(b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":42}', 400, id="numeric params"),
]


@pytest.mark.parametrize(("body", "status"), MALFORMED)
async def test_malformed_payloads_are_answered_and_never_forwarded(gateway, received, body, status):
    response = await gateway.post(
        "/mcp", content=body, headers={**auth(VIEWER_TOKEN), "Content-Type": "application/json"}
    )
    assert response.status_code == status, body
    assert response.content, "an identified request must not get an empty response"
    assert errors_in(response.json()), body
    assert received == [], body


# --------------------------------------------------------------------------
# Size cap
# --------------------------------------------------------------------------


async def test_an_oversized_body_is_refused_before_it_is_parsed(gateway, received, config):
    padding = "x" * (config.max_body_bytes + 1024)
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": SAFE_TOOL, "arguments": {"pad": padding}}})
    response = await gateway.post(
        "/mcp", content=body.encode(), headers={**auth(VIEWER_TOKEN), "Content-Type": "application/json"}
    )
    assert response.status_code == 413
    assert received == []


async def test_a_body_just_under_the_cap_is_accepted(gateway, received, config):
    padding = "x" * (config.max_body_bytes - 512)
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": SAFE_TOOL, "arguments": {"pad": padding}}})
    assert len(body) < config.max_body_bytes
    response = await gateway.post(
        "/mcp", content=body.encode(), headers={**auth(VIEWER_TOKEN), "Content-Type": "application/json"}
    )
    assert response.status_code == 200
    assert len(received) == 1


async def test_a_lying_content_length_does_not_bypass_the_cap(gateway, received, config):
    """The declared length is client-supplied; the streaming check is the real
    guard."""
    body = b'{"jsonrpc":"2.0","id":1,"method":"ping"}' + b" " * (config.max_body_bytes * 2)
    response = await gateway.post(
        "/mcp", content=body, headers={**auth(VIEWER_TOKEN), "Content-Type": "application/json"}
    )
    assert response.status_code == 413
    assert received == []


# --------------------------------------------------------------------------
# Header handling
# --------------------------------------------------------------------------


async def test_the_client_credential_never_reaches_downstream(capturing_gateway):
    client, captured = capturing_gateway
    await client.post("/mcp", json=call(SAFE_TOOL), headers=auth(ADMIN_TOKEN))
    assert "authorization" not in captured["headers"]


async def test_a_client_cannot_assert_its_own_forwarded_role(capturing_gateway):
    """SEC-6: the gateway's values won only because of dict-assignment order."""
    client, captured = capturing_gateway
    await client.post(
        "/mcp",
        json=call(SAFE_TOOL),
        headers={**auth(VIEWER_TOKEN), "X-Forwarded-Role": "admin", "X-Forwarded-User": "attacker"},
    )
    assert captured["headers"]["x-forwarded-role"] == "viewer"
    assert captured["headers"]["x-forwarded-user"] == "analyst@example.com"
    assert "admin" not in captured["headers"]["x-forwarded-role"]


# --------------------------------------------------------------------------
# SEC-4 - audit log injection
# --------------------------------------------------------------------------


async def test_a_tool_name_cannot_forge_an_audit_record(gateway, received, caplog):
    """The tool name is attacker-controlled and used to be interpolated raw into
    a newline-delimited log."""
    forged = 'safe\ndecision=ALLOW subject=attacker role=admin method=tools/call tool=admin_reset'
    with caplog.at_level("INFO", logger="mcp_gateway.audit"):
        await gateway.post("/mcp", json=call(forged), headers=auth(VIEWER_TOKEN))

    records = [record.getMessage() for record in caplog.records if record.name == "mcp_gateway.audit"]
    assert records, "the decision was not audited at all"
    for record in records:
        assert "\n" not in record, "a newline reached the audit log"
        assert "\r" not in record
        parsed = json.loads(record)
        assert parsed["role"] == "viewer", "the forged role was recorded"
        assert parsed["subject"] == "analyst@example.com"


INJECTION_PAYLOADS = [
    pytest.param("a\nb", id="newline"),
    pytest.param("a\rb", id="carriage return"),
    pytest.param("a\r\nb", id="crlf"),
    pytest.param("a\tb", id="tab"),
    pytest.param('a"b', id="double quote"),
    pytest.param("a\\b", id="backslash"),
    pytest.param("a b", id="U+2028 line separator"),
    pytest.param("a b", id="U+2029 paragraph separator"),
    pytest.param("a\x00b", id="null byte"),
    pytest.param("a\x1b[31mb", id="ansi escape"),
    pytest.param("a\x07b", id="bell"),
]


@pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
async def test_control_characters_in_a_tool_name_stay_on_one_audit_line(gateway, caplog, payload):
    with caplog.at_level("INFO", logger="mcp_gateway.audit"):
        await gateway.post("/mcp", json=call(payload), headers=auth(VIEWER_TOKEN))

    for record in caplog.records:
        if record.name != "mcp_gateway.audit":
            continue
        message = record.getMessage()
        assert message.count("\n") == 0, payload
        assert " " not in message and " " not in message
        assert json.loads(message)["tool"] == payload, "the value must still be recorded faithfully"


# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------


async def test_a_hundred_concurrent_privileged_calls_are_all_blocked(gateway, received):
    """A race in the decision path would show up as a leaked forward."""
    import asyncio

    responses = await asyncio.gather(
        *(gateway.post("/mcp", json=call(ADMIN_TOOL, id=i), headers=auth(VIEWER_TOKEN)) for i in range(100))
    )
    assert all(response.status_code == 200 for response in responses)
    assert all(errors_in(r.json())[0]["code"] == jsonrpc.UNAUTHORIZED_TOOL_CALL for r in responses)
    assert received == [], f"{len(received)} privileged calls leaked under concurrency"


async def test_mixed_concurrent_roles_do_not_cross_over(gateway, received):
    import asyncio

    await asyncio.gather(
        *(
            gateway.post(
                "/mcp",
                json=call(ADMIN_TOOL, id=i),
                headers=auth(ADMIN_TOKEN if i % 2 == 0 else VIEWER_TOKEN),
            )
            for i in range(60)
        )
    )
    assert len(received) == 30, f"expected 30 admin calls through, saw {len(received)}"
    assert all(item["params"]["name"] == ADMIN_TOOL for item in received)
