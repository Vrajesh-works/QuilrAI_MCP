"""Standardised gateway error payloads.

Every failure the client sees is built here, so there is exactly one place where
an internal detail could leak and exactly one place to audit. Upstream
exceptions, provider error bodies and stack traces are logged, never returned:
a connection error's message embeds the internal host and port, and a provider's
own error body can name internal model deployments or account identifiers.

The shape follows the Anthropic error convention so existing SDK error handling
keeps working through the gateway:

    {"type": "error", "error": {"type": "...", "message": "..."}, "gateway": {...}}
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Client-visible error types. Deliberately coarse: a finer taxonomy would start
# describing upstream internals.
RATE_LIMITED = "rate_limit_error"
UPSTREAM_UNAVAILABLE = "api_error"
INVALID_REQUEST = "invalid_request_error"
TIMEOUT = "timeout_error"

_SAFE_MESSAGES = {
    RATE_LIMITED: "Token rate limit exceeded for this API key.",
    UPSTREAM_UNAVAILABLE: "No model provider was able to serve this request.",
    INVALID_REQUEST: "The request could not be processed.",
    TIMEOUT: "The request timed out before any provider responded.",
}


def gateway_error(
    error_type: str,
    message: str | None = None,
    *,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the one error shape this gateway ever returns.

    Args:
        message: overrides the canned text. Must be a string this module's
            author wrote - never an upstream message or an exception's `str()`.
        detail: gateway-owned diagnostics (which providers were tried, retry
            timing). Safe by construction: assembled here, not copied from
            upstream.
    """
    payload: dict[str, Any] = {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message or _SAFE_MESSAGES.get(error_type, "The request could not be completed."),
        },
    }
    if detail:
        payload["gateway"] = detail
    return payload


def log_and_sanitise(context: str, exc: BaseException) -> None:
    """Record the real failure for operators; the caller returns a safe payload.

    Split out so the two halves cannot drift: whenever an exception is swallowed
    into a generic response, its detail still reaches the log.
    """
    logger.warning("%s: %s: %s", context, type(exc).__name__, exc, exc_info=logger.isEnabledFor(logging.DEBUG))
