"""Method-level authorization: who may call what.

The rule: `tools/list` forwards transparently; `tools/call` on a
tool whose name starts with `admin_` requires the admin role, and is rejected
with -32001 *without* the downstream server being contacted.

Everything here is a pure function of (principal, message) so the decisions can
be tested directly, without HTTP.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from mcp_gateway.auth import Principal
from mcp_gateway.jsonrpc import (
    INVALID_PARAMS,
    UNAUTHORIZED_TOOL_CALL,
    RpcMessage,
)

ADMIN_TOOL_PREFIX = "admin_"
ADMIN_ROLE = "admin"


@dataclass(frozen=True)
class Denial:
    """A blocked message and the JSON-RPC error to return for it."""

    code: int
    message: str
    reason: str  # short machine-readable tag, for logs and metrics


def _normalize_tool_name(name: str) -> str:
    """Fold a tool name to the form the prefix check runs against.

    Three things happen here, each closing a bypass:

    * NFKC normalization, so the fullwidth `ａdmin_reset` (U+FF41...) cannot
      present as a different string to us than to a downstream server that
      normalizes, or to a human reading an audit log.
    * Whitespace stripping, so `" admin_reset"` cannot dodge a prefix match.
    * Case folding, so `Admin_reset` is treated as privileged.

    Case folding is a deliberate over-approximation: it would also catch a
    genuinely non-privileged tool named `Admin_helper`. For a security filter
    that is the right direction to be wrong in - a false denial is visible and
    fixable, a false allow is a silent privilege escalation.
    """
    return unicodedata.normalize("NFKC", name).strip().casefold()


def is_privileged_tool(name: str) -> bool:
    """Whether calling this tool requires the admin role."""
    return _normalize_tool_name(name).startswith(ADMIN_TOOL_PREFIX)


def evaluate(principal: Principal, message: RpcMessage) -> Denial | None:
    """Decide whether one message may be forwarded.

    Returns:
        None to allow, or a `Denial` describing why not.
    """
    if message.method != "tools/call":
        # tools/list and the rest of the protocol (initialize, ping, resources,
        # prompts) carry no privileged action, so they forward transparently.
        return None

    if not isinstance(message.params, dict):
        return Denial(
            code=INVALID_PARAMS,
            message="tools/call requires a params object containing 'name'.",
            reason="malformed_params",
        )

    tool_name = message.tool_name
    if tool_name is None:
        # Fail closed. Without a readable name there is no way to know whether
        # this call is privileged, so it cannot be allowed through.
        return Denial(
            code=INVALID_PARAMS,
            message="tools/call requires 'params.name' to be a string.",
            reason="missing_tool_name",
        )

    if is_privileged_tool(tool_name) and not principal.is_admin:
        return Denial(
            code=UNAUTHORIZED_TOOL_CALL,
            message="Unauthorized Tool Call",
            reason="insufficient_role",
        )

    return None
