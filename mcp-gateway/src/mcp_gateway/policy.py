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
    METHOD_NOT_FOUND,
    UNAUTHORIZED_TOOL_CALL,
    RpcMessage,
)

ADMIN_TOOL_PREFIX = "admin_"
ADMIN_ROLE = "admin"

TOOLS_CALL = "tools/call"

# The methods this gateway will forward. Anything else is answered locally with
# -32601 and the downstream server is never contacted.
#
# An allowlist rather than a denylist, for the same reason the tool check folds
# case: the failure mode of forgetting to add a method is a visible error, and
# the failure mode of forgetting to *block* one is a silent bypass. Adding a
# method as the protocol grows is a one-line config change
# (`MCP_GATEWAY_EXTRA_METHODS`), and the whole check can be turned off with
# `MCP_GATEWAY_METHOD_ALLOWLIST=off` for a deployment that would rather relay
# everything.
KNOWN_METHODS: frozenset[str] = frozenset(
    {
        "initialize",
        "ping",
        "completion/complete",
        "logging/setLevel",
        "prompts/list",
        "prompts/get",
        "resources/list",
        "resources/read",
        "resources/subscribe",
        "resources/unsubscribe",
        "resources/templates/list",
        "roots/list",
        "sampling/createMessage",
        "elicitation/create",
        "tools/list",
        TOOLS_CALL,
        "notifications/initialized",
        "notifications/cancelled",
        "notifications/progress",
        "notifications/message",
        "notifications/roots/list_changed",
        "notifications/resources/updated",
        "notifications/resources/list_changed",
        "notifications/tools/list_changed",
        "notifications/prompts/list_changed",
    }
)


@dataclass(frozen=True)
class Denial:
    """A blocked message and the JSON-RPC error to return for it."""

    code: int
    message: str
    reason: str  # short machine-readable tag, for logs and metrics


# Characters that render as nothing: control, format (zero-width joiners, the
# BOM), and the whitespace separator classes.
_INVISIBLE_CATEGORIES = frozenset({"Cc", "Cf", "Zs", "Zl", "Zp"})


def _normalize_tool_name(name: str) -> str:
    """Fold a tool name to the form the prefix check runs against.

    Four things happen here, each closing a class of bypass:

    * **NFKC normalization**, so the fullwidth `ａdmin_reset` (U+FF41...) cannot
      present as a different string to us than to a downstream server that
      normalizes, or to a human reading an audit log.
    * **Invisible characters removed anywhere in the string**, not just trimmed
      from the ends. `str.strip()` does not touch `"\\ufeffadmin_reset_key"` -
      the BOM is category `Cf`, not whitespace - and zero-width spaces and
      joiners behave the same way. Stripping the ends is not enough; the check
      is about what the name *is*, and an invisible character is not part of
      that.
    * **Case folding**, so `Admin_reset` is treated as privileged.
    * **Combining marks removed** after folding. `ADMİN_reset` (U+0130, the
      Turkish dotted capital I) case-folds to `i` + U+0307 COMBINING DOT ABOVE,
      which is not `i`, so a plain prefix match would miss it. Decomposing and
      dropping the marks collapses that whole family - `admın`, `ádmin`, and
      every other diacritic variant - onto `admin`.

    All of this is deliberate over-approximation: it also catches a genuinely
    non-privileged tool named `Admin_helper`, or `adm in_x`. For a security
    filter that is the right direction to be wrong in - a false denial is
    visible and fixable, a false allow is a silent privilege escalation.
    """
    folded = unicodedata.normalize("NFKC", name)
    folded = "".join(
        character
        for character in folded
        if unicodedata.category(character) not in _INVISIBLE_CATEGORIES
    )
    decomposed = unicodedata.normalize("NFD", folded.casefold())
    stripped = "".join(
        character for character in decomposed if unicodedata.category(character) != "Mn"
    )
    return unicodedata.normalize("NFC", stripped)


def is_privileged_tool(name: str) -> bool:
    """Whether calling this tool requires the admin role."""
    return _normalize_tool_name(name).startswith(ADMIN_TOOL_PREFIX)


def normalize_method(method: str) -> str:
    """Fold a method name the same way tool names are folded.

    Folding both sides removes an asymmetry that would otherwise matter: if tool
    names are NFKC-normalised, stripped and case-folded while the method beside
    them is compared with `!=`, then `TOOLS/CALL`, `tools/Call` and
    `"tools/call "` skip the tool check entirely. That is harmless against a
    downstream that dispatches by exact dict lookup - the MCP SDK simply 404s -
    but it is exploitable against any downstream that trims or lowercases, and
    a policy layer should not depend on the downstream's parsing quirks.
    """
    return unicodedata.normalize("NFKC", method).strip().casefold()


def evaluate(
    principal: Principal, message: RpcMessage, *, enforce_allowlist: bool = True
) -> Denial | None:
    """Decide whether one message may be forwarded.

    Returns:
        None to allow, or a `Denial` describing why not.
    """
    normalized = normalize_method(message.method)

    if enforce_allowlist and message.method not in KNOWN_METHODS:
        # Note this compares the *raw* method, so a method that only matches
        # after folding - `TOOLS/CALL`, `Tools/Call ` - is rejected rather than
        # quietly rewritten. If a client meant `tools/call` it can send it.
        return Denial(
            code=METHOD_NOT_FOUND,
            message="Unknown method.",
            reason="method_not_allowlisted",
        )

    if normalized != TOOLS_CALL:
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
