"""Wire-format parsing, tested directly.

Parsing is a security boundary here - the authorization decision is made from
the parsed result - so these cover the shapes that a policy check could be
tricked by, not just the happy path.
"""

from __future__ import annotations

import json

import pytest

from mcp_gateway.jsonrpc import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    PARSE_ERROR,
    InvalidPayload,
    error_response,
    parse_payload,
)


def encode(payload) -> bytes:
    return json.dumps(payload).encode()


def test_single_request_is_parsed():
    parsed = parse_payload(encode({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}))

    assert parsed.is_batch is False
    assert len(parsed.messages) == 1
    message = parsed.messages[0]
    assert message.method == "tools/list"
    assert message.id == 1
    assert message.is_request is True


def test_batch_is_parsed_into_separate_messages():
    parsed = parse_payload(
        encode([{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, {"jsonrpc": "2.0", "id": 2, "method": "ping"}])
    )

    assert parsed.is_batch is True
    assert [message.method for message in parsed.messages] == ["tools/list", "ping"]


def test_notification_is_distinguished_from_a_request():
    """The difference decides whether a blocked message draws a response."""
    notification = parse_payload(encode({"jsonrpc": "2.0", "method": "ping"})).messages[0]
    assert notification.is_request is False

    # An explicit null id is a request, per §4 - hence a key check, not
    # a truthiness test.
    explicit_null = parse_payload(encode({"jsonrpc": "2.0", "id": None, "method": "ping"})).messages[0]
    assert explicit_null.is_request is True


def test_tool_name_extraction():
    message = parse_payload(
        encode({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "admin_reset_key"}})
    ).messages[0]
    assert message.tool_name == "admin_reset_key"


@pytest.mark.parametrize(
    ("params", "why"),
    [
        ({"name": ["admin_reset_key"]}, "list instead of string"),
        ({"name": {"toString": "admin_reset_key"}}, "object instead of string"),
        ({"name": 42}, "number instead of string"),
        ({"name": None}, "explicit null"),
        ({}, "name absent"),
        ([{"name": "admin_reset_key"}], "params is an array"),
    ],
)
def test_tool_name_is_none_when_not_a_plain_string(params, why):
    """Anything that is not a string returns None so the policy can fail closed.

    A `str.startswith` against a non-string either raises or silently misses;
    both are worse than refusing to guess.
    """
    message = parse_payload(
        encode({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params})
    ).messages[0]
    assert message.tool_name is None, why


@pytest.mark.parametrize(
    ("body", "code", "why"),
    [
        (b"{not json", PARSE_ERROR, "malformed JSON"),
        (b"", INVALID_REQUEST, "empty body"),
        (b"   ", INVALID_REQUEST, "whitespace only"),
        (b"[]", INVALID_REQUEST, "empty batch (JSON-RPC §6)"),
        (b'"a string"', INVALID_REQUEST, "not an object"),
        (b"42", INVALID_REQUEST, "bare number"),
        (b"null", INVALID_REQUEST, "null document"),
    ],
)
def test_unusable_bodies_are_rejected(body, code, why):
    with pytest.raises(InvalidPayload) as caught:
        parse_payload(body)
    assert caught.value.code == code, why


@pytest.mark.parametrize(
    ("message", "why"),
    [
        ({"id": 1, "method": "tools/list"}, "missing jsonrpc version"),
        ({"jsonrpc": "1.0", "id": 1, "method": "tools/list"}, "wrong version"),
        ({"jsonrpc": "2.0", "id": 1}, "missing method"),
        ({"jsonrpc": "2.0", "id": 1, "method": ""}, "empty method"),
        ({"jsonrpc": "2.0", "id": 1, "method": 42}, "non-string method"),
        ({"jsonrpc": "2.0", "id": {"a": 1}, "method": "ping"}, "object id"),
        ({"jsonrpc": "2.0", "id": True, "method": "ping"}, "boolean id"),
    ],
)
def test_invalid_envelopes_are_rejected(message, why):
    with pytest.raises(InvalidPayload) as caught:
        parse_payload(encode(message))
    assert caught.value.code in (INVALID_REQUEST, INVALID_PARAMS), why


def test_non_object_params_is_invalid_params():
    with pytest.raises(InvalidPayload) as caught:
        parse_payload(encode({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": "nope"}))
    assert caught.value.code == INVALID_PARAMS


def test_rejection_preserves_the_id_where_it_can():
    """So the client can still correlate the failure with its call."""
    with pytest.raises(InvalidPayload) as caught:
        parse_payload(encode({"jsonrpc": "1.0", "id": 99, "method": "ping"}))
    assert caught.value.id == 99


def test_error_response_shape():
    response = error_response(-32001, "Unauthorized Tool Call", id=7, data={"tool": "admin_x"})

    assert response == {
        "jsonrpc": "2.0",
        "id": 7,
        "error": {"code": -32001, "message": "Unauthorized Tool Call", "data": {"tool": "admin_x"}},
    }


def test_error_response_omits_absent_data():
    assert "data" not in error_response(-32600, "Invalid Request")["error"]
