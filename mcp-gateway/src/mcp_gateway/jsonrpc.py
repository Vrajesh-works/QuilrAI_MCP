"""JSON-RPC 2.0 wire format: parsing, classification, error construction.

The gateway makes authorization decisions from the *parsed* payload, so parsing
is a security boundary rather than a convenience. Everything here fails closed:
anything that cannot be confidently understood is rejected rather than passed
downstream on the assumption it is probably harmless.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# Standard codes (JSON-RPC 2.0 §5.1).
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Implementation-defined server errors live in -32000..-32099.
UNAUTHORIZED_TOOL_CALL = -32001
UPSTREAM_UNAVAILABLE = -32002

JsonRpcId = str | int | float | None


@dataclass(frozen=True)
class RpcMessage:
    """One parsed JSON-RPC message.

    `is_request` distinguishes a call that expects a response from a
    notification that does not - the gateway must stay silent about the latter
    even when it blocks it.
    """

    method: str
    params: dict[str, Any] | list[Any] | None
    id: JsonRpcId
    is_request: bool
    raw: dict[str, Any]

    @property
    def tool_name(self) -> str | None:
        """`params.name` for a tools/call, or None if absent or not a string.

        Returning None for a non-string name matters: a caller sending
        `{"name": ["admin_reset_key"]}` must not slip past a `str.startswith`
        check by making the value the wrong type.
        """
        if not isinstance(self.params, dict):
            return None
        name = self.params.get("name")
        return name if isinstance(name, str) else None


class _DuplicateKey(ValueError):
    """A JSON object contained the same key twice."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Refuse any object with a repeated key.

    The gateway authorizes on the parsed payload but forwards the *original
    bytes* when nothing is blocked - deliberately, so it does not normalise key
    order or number formatting on a payload it is only inspecting. That leaves
    the authorization decision and the downstream execution reading the same
    bytes with two different parsers.

    For `{"name": "admin_reset", "name": "get_customer_record"}` CPython keeps
    the *last* key, so the gateway sees `get_customer_record` and allows it. A
    downstream parser that keeps the first - and several do, including some
    streaming and hand-rolled implementations - executes `admin_reset`. A
    CPython downstream is not vulnerable to that, which is precisely why the
    check belongs here: the divergence would appear silently, with no code
    change on this side, the day the downstream implementation is swapped.

    Rejecting the payload keeps the byte-relay design intact and costs five
    lines. No legitimate JSON-RPC client emits duplicate keys.
    """
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise _DuplicateKey(f"Duplicate key {key!r} in a JSON object; the request is ambiguous.")
        seen[key] = value
    return seen


class InvalidPayload(Exception):
    """The body is not a usable JSON-RPC payload. Carries the error to return."""

    def __init__(self, code: int, message: str, id: JsonRpcId = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.id = id


def error_response(code: int, message: str, id: JsonRpcId = None, data: Any = None) -> dict[str, Any]:
    """Build a JSON-RPC error object, echoing the request id.

    Echoing the id is what lets the client correlate the rejection with the call
    it made; a batch of ten with one blocked is otherwise unattributable.
    """
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": id, "error": error}


def _valid_id(value: Any) -> bool:
    """Per §4: an id is a string, a number, or null. Never a bool - `True` is an
    `int` in Python, so an unguarded isinstance check would accept it."""
    if isinstance(value, bool):
        return False
    return value is None or isinstance(value, (str, int, float))


def _parse_one(item: Any) -> RpcMessage:
    """Validate a single message object.

    Raises:
        InvalidPayload: the object is not a well-formed JSON-RPC request.
    """
    if not isinstance(item, dict):
        raise InvalidPayload(INVALID_REQUEST, "Request must be a JSON object.")

    # Recover the id before validating anything else, so the rejection can still
    # be correlated by the client.
    raw_id = item.get("id")
    message_id: JsonRpcId = raw_id if _valid_id(raw_id) else None

    if item.get("jsonrpc") != "2.0":
        raise InvalidPayload(INVALID_REQUEST, "Missing or unsupported 'jsonrpc' version; expected '2.0'.", message_id)

    method = item.get("method")
    if not isinstance(method, str) or not method:
        raise InvalidPayload(INVALID_REQUEST, "Request 'method' must be a non-empty string.", message_id)

    params = item.get("params")
    if params is not None and not isinstance(params, (dict, list)):
        raise InvalidPayload(INVALID_PARAMS, "Request 'params' must be an object or an array.", message_id)

    if "id" in item and not _valid_id(raw_id):
        raise InvalidPayload(INVALID_REQUEST, "Request 'id' must be a string, number, or null.", None)

    return RpcMessage(
        method=method,
        params=params,
        # A message with no "id" key at all is a notification. An explicit
        # null id is a request per the spec, hence the key check rather than
        # a truthiness test.
        id=message_id,
        is_request="id" in item,
        raw=item,
    )


@dataclass(frozen=True)
class ParsedPayload:
    """A parsed body, single or batch, kept together with its original shape."""

    messages: list[RpcMessage]
    is_batch: bool


def parse_payload(body: bytes) -> ParsedPayload:
    """Parse a request body into one or more messages.

    Batches are the reason this returns a list. A gateway that only inspects
    `payload["method"]` sees nothing in `[{...tools/list...}, {...admin call...}]`
    and forwards the whole array, which is the single most common way a filter
    like this is bypassed. Every element is parsed and checked individually.

    Raises:
        InvalidPayload: unparseable JSON, or a structurally invalid envelope.
    """
    if not body.strip():
        raise InvalidPayload(INVALID_REQUEST, "Empty request body.")

    try:
        document = json.loads(body, object_pairs_hook=_reject_duplicate_keys)
    except _DuplicateKey as exc:
        raise InvalidPayload(INVALID_REQUEST, str(exc)) from None
    except json.JSONDecodeError as exc:
        raise InvalidPayload(PARSE_ERROR, f"Invalid JSON: {exc.msg}.") from None

    if isinstance(document, list):
        if not document:
            # §6: an empty array is itself an Invalid Request.
            raise InvalidPayload(INVALID_REQUEST, "Batch must contain at least one request.")
        return ParsedPayload(messages=[_parse_one(item) for item in document], is_batch=True)

    return ParsedPayload(messages=[_parse_one(document)], is_batch=False)
