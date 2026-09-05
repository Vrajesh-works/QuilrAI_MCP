"""Mapping Python exceptions onto JSON-RPC errors.

Two channels, chosen deliberately (the README carries the full table):

  * A broken *contract* - malformed arguments, an unknown tool - is a JSON-RPC
    error. The caller sent something the server will never accept.
  * A broken *expectation* - no such customer, refund exceeds the balance - is a
    successful call returning `isError: true`, because the model should see the
    outcome and reason about it rather than get a transport-level failure.
"""

from __future__ import annotations

import logging

from mcp.shared.exceptions import MCPError
from mcp_types import INTERNAL_ERROR, INVALID_PARAMS, METHOD_NOT_FOUND
from pydantic import ValidationError

logger = logging.getLogger(__name__)


def invalid_params(tool_name: str, exc: ValidationError) -> MCPError:
    """Flatten a pydantic ValidationError into -32602 with per-field detail.

    The SDK maps an uncaught ValidationError to INVALID_PARAMS with an empty
    `data` field, which is spec-compliant but tells the caller nothing about
    *which* field was wrong. Catching it here and rebuilding the payload keeps
    the code correct and the message actionable.

    Field inputs are deliberately not echoed back: `loc`, `msg` and `type` say
    what is wrong without reflecting caller data (possibly a real customer id or
    free-text reason) into logs and error payloads downstream.
    """
    issues = [
        {
            "field": ".".join(str(part) for part in error["loc"]) or "(root)",
            "error": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]
    summary = "; ".join(f"{issue['field']}: {issue['error']}" for issue in issues)
    return MCPError(
        code=INVALID_PARAMS,
        message=f"Invalid arguments for tool {tool_name!r}: {summary}",
        data={"tool": tool_name, "issues": issues},
    )


def unknown_tool(tool_name: str, known: list[str]) -> MCPError:
    """-32601 for a tool that does not exist.

    `tools/call` itself is a valid method, but the named tool is not part of
    this server's surface, and METHOD_NOT_FOUND is the closest standard code.
    """
    return MCPError(
        code=METHOD_NOT_FOUND,
        message=f"Unknown tool: {tool_name!r}",
        data={"available_tools": sorted(known)},
    )


def internal_error(tool_name: str, exc: BaseException) -> MCPError:
    """-32603 with the details logged to stderr, never put on the wire.

    An unexpected exception can carry file paths, connection strings or fragments
    of another customer's data in its message. The client gets an opaque
    apology; the operator gets the traceback on stderr.
    """
    logger.exception("Unhandled error in tool %r: %s", tool_name, exc)
    return MCPError(
        code=INTERNAL_ERROR,
        message=f"Internal error while executing tool {tool_name!r}.",
        data={"tool": tool_name},
    )
