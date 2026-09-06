"""A stdio transport that answers malformed requests instead of dropping them.

Why this layer exists
---------------------
`mcp.server.stdio.stdio_server` validates each line with
`jsonrpc_message_adapter.validate_json`. On failure it forwards the *exception*
into the read stream and discards the line. By the time the low-level `Server`
sees anything the request id is gone, so no layer above the transport can answer
with the right id - and answering with `id: null` does not help a client that
correlates by id.

Left to the SDK alone, these shapes therefore draw no response at all - no
result, no error frame, nothing - and a client that sends one waits out its own
timeout with no information:

    missing 'method'       -> no response (client hangs)
    params is an array     -> no response
    jsonrpc version 1.0    -> no response
    jsonrpc key absent     -> no response
    id is a bool           -> no response
    method is a number     -> no response
    batch array of 1       -> no response

JSON-RPC 2.0 §5 requires a response to every request carrying an id, and §5.1
reserves `-32600` for exactly these shapes. The check has to run **before** the
SDK's parse, on the raw line, which is why this module exists rather than a
handler somewhere in `server.py`.

File descriptors
----------------
`stdio_server` claims fd 0 and fd 1 independently, and only when the
corresponding stream argument is None. This module passes an explicit `stdin`
and leaves `stdout` alone, so **the SDK still claims fd 1** - pointing it at
stderr and serving the wire from a private duplicate. That is the mechanism
behind the stdout-purity guarantee, and the test suite pins it against a process
that is actively printing to stdout.

Passing an explicit `stdin` gives up the fd 0 diversion, which pointed stdin at
the null device so a child process could not steal bytes off the wire. This
server spawns no children. The trade is recorded here rather than left to be
discovered.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import TextIOWrapper
from typing import Any

import anyio
import mcp_types as types
from mcp.server.stdio import stdio_server
from mcp.shared.message import SessionMessage

logger = logging.getLogger(__name__)

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INVALID_PARAMS = -32602

#: Rejections queued while `stdio_server` is still starting up. Sized far above
#: any realistic burst; if it ever filled, the reader would block rather than
#: drop a response, which is the safe direction.
_REJECTION_BUFFER = 256


def _recoverable_id(value: Any) -> int | str | None:
    """The id to echo, or None when the request did not supply a usable one.

    `isinstance(True, int)` is True in Python, so the bool check has to come
    first or `{"id": true}` echoes back as `1` and the client correlates the
    error with a completely different call.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int) or isinstance(value, str):
        return value
    return None


def _error(code: int, message: str, request_id: int | str | None) -> SessionMessage:
    return SessionMessage(
        types.JSONRPCError(
            jsonrpc="2.0",
            id=request_id,
            error=types.ErrorData(code=code, message=message),
        )
    )


def classify_line(line: str) -> SessionMessage | None | bool:
    """Decide what to do with one raw line from stdin.

    Returns:
        `True` to hand the line to the SDK unchanged, `None` to drop it
        silently, or a `SessionMessage` carrying the error response to send.

    Everything that is *well-formed at the envelope level* is passed straight
    through, so the SDK performs the real validation and this layer cannot
    change the meaning of a valid request. It only answers the shapes the SDK
    would otherwise discard.
    """
    if not line.strip():
        return None  # Blank line between frames; not a message.

    try:
        payload = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        # No id is recoverable from something that is not JSON at all. §5.1
        # allows `id: null` precisely for this case.
        return _error(PARSE_ERROR, "Parse error: the request is not valid JSON.", None)

    if isinstance(payload, list):
        # JSON-RPC allows batches; MCP's stdio framing is one message per line
        # and the SDK's adapter does not accept an array. Saying so is better
        # than silence.
        return _error(INVALID_REQUEST, "Batch requests are not supported over this transport.", None)

    if not isinstance(payload, dict):
        return _error(INVALID_REQUEST, "Invalid Request: a request must be a JSON object.", None)

    has_id = "id" in payload
    request_id = _recoverable_id(payload.get("id")) if has_id else None

    # A response to a request *this server* issued (sampling, elicitation, roots)
    # is not a request and must not be validated as one.
    if "method" not in payload and ("result" in payload or "error" in payload):
        return True

    # From here on the message claims to be a request or a notification. A
    # malformed *notification* still draws no response: §4.1 is unconditional,
    # and inventing a frame would desynchronise a client that is not expecting
    # one. Only requests - those carrying an id - get an error back.
    def reject(code: int, message: str, echo: int | str | None = ...) -> SessionMessage | None:
        if not has_id:
            logger.info("Dropping a malformed notification: %s", message)
            return None
        return _error(code, message, request_id if echo is ... else echo)

    if payload.get("jsonrpc") != "2.0":
        return reject(INVALID_REQUEST, "Invalid Request: 'jsonrpc' must be exactly \"2.0\".")

    method = payload.get("method")
    if not isinstance(method, str) or not method:
        return reject(INVALID_REQUEST, "Invalid Request: 'method' must be a non-empty string.")

    if "params" in payload and payload["params"] is not None and not isinstance(payload["params"], dict):
        return reject(INVALID_PARAMS, "Invalid params: 'params' must be an object.")

    if has_id and request_id is None:
        # The id was present but unusable - a bool, a float, an object. It
        # cannot be echoed, so the error goes back with `id: null` and the
        # message says why, rather than the request vanishing.
        return _error(
            INVALID_REQUEST,
            "Invalid Request: 'id' must be a string or an integer.",
            None,
        )

    return True


class _ValidatingStdin:
    """Wraps stdin, yielding only lines the SDK can parse.

    `stdio_server` consumes its `stdin` argument with `async for line in stdin`,
    so this only needs to be async-iterable. Rejections are pushed onto a queue
    that the caller drains into the write stream.
    """

    def __init__(self, source: Any, rejections: Any):
        self._source = source
        self._rejections = rejections

    async def __aiter__(self) -> AsyncIterator[str]:
        async for line in self._source:
            verdict = classify_line(line)
            if verdict is True:
                yield line
            elif verdict is not None:
                await self._rejections.send(verdict)


@asynccontextmanager
async def validating_stdio_server():
    """`stdio_server`, plus a JSON-RPC error for every malformed request.

    Yields the same `(read_stream, write_stream)` pair, so `Server.run` is
    called exactly as before.
    """
    send_rejection, receive_rejection = anyio.create_memory_object_stream[SessionMessage](
        max_buffer_size=_REJECTION_BUFFER
    )
    # Decoded the same way the SDK decodes it: UTF-8, replacing anything
    # undecodable rather than killing the transport on one bad byte.
    raw_stdin = anyio.wrap_file(TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace"))
    stdin = _ValidatingStdin(raw_stdin, send_rejection)

    async with stdio_server(stdin=stdin) as (read_stream, write_stream):

        async def pump_rejections() -> None:
            async with receive_rejection:
                async for rejection in receive_rejection:
                    await write_stream.send(rejection)

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(pump_rejections)
            try:
                yield read_stream, write_stream
            finally:
                task_group.cancel_scope.cancel()
