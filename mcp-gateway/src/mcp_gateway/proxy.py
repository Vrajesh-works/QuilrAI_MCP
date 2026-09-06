"""Forwarding to the downstream MCP server.

Header handling is the fiddly part of writing a reverse proxy and the part most
often got wrong, so it is isolated here rather than inlined into the route.
"""

from __future__ import annotations

import logging

import httpx

from mcp_gateway.auth import Principal

logger = logging.getLogger(__name__)

# RFC 9110 §7.6.1: hop-by-hop headers describe a single connection and must not
# be forwarded. Content-Length is dropped because the body we send may differ in
# length from the one we received; httpx recomputes it.
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}

# The client's credential stops at the gateway. Forwarding it would let the
# downstream server make its own authorization decisions from a token it should
# never see, which defeats the point of terminating auth here.
#
# `x-forwarded-user` / `x-forwarded-role` are stripped explicitly rather than
# relying on the fact that the gateway assigns them *after* the comprehension
# below. That ordering does make the gateway's values win, but only as an
# emergent property: a later refactor moving the assignment would hand a client
# the ability to assert its own role downstream, with no failing test. Stating
# it as a rule makes the guarantee independent of statement order.
_STRIP_FROM_CLIENT = {"authorization", "x-forwarded-user", "x-forwarded-role", *_HOP_BY_HOP}

# Set by the downstream server; meaningful to the client, so pass them back.
_STRIP_FROM_DOWNSTREAM = _HOP_BY_HOP


def build_downstream_headers(incoming: httpx.Headers, principal: Principal) -> dict[str, str]:
    """Headers for the downstream request.

    The client's identity travels as `X-Forwarded-*` claims the gateway asserts,
    rather than as the original bearer token. `Mcp-Session-Id` is preserved
    because MCP's HTTP transport uses it to correlate a session, and a proxy
    that drops it silently breaks every stateful server behind it.
    """
    headers = {
        key: value for key, value in incoming.items() if key.lower() not in _STRIP_FROM_CLIENT
    }
    headers["x-forwarded-user"] = principal.subject
    headers["x-forwarded-role"] = principal.role
    return headers


def filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    """Headers to relay back to the client."""
    return {key: value for key, value in headers.items() if key.lower() not in _STRIP_FROM_DOWNSTREAM}


class DownstreamError(Exception):
    """The downstream server could not be reached or did not answer in time.

    Carries a short, safe summary. The original exception - which can name
    internal hostnames, ports and connection details - is logged, never
    returned.
    """

    def __init__(self, summary: str, cause: BaseException | None = None):
        super().__init__(summary)
        self.summary = summary
        self.cause = cause


async def forward(
    client: httpx.AsyncClient,
    url: str,
    body: bytes,
    headers: dict[str, str],
    timeout: float,
) -> httpx.Response:
    """POST a JSON-RPC body downstream and return the raw response.

    Raises:
        DownstreamError: timeout or transport failure, already sanitised.
    """
    try:
        return await client.post(url, content=body, headers=headers, timeout=timeout)
    except httpx.TimeoutException as exc:
        logger.warning("Downstream timed out after %ss: %r", timeout, exc)
        raise DownstreamError("The upstream MCP server did not respond in time.", exc) from None
    except httpx.HTTPError as exc:
        # Includes ConnectError, whose message embeds the internal address.
        logger.warning("Downstream request failed: %r", exc)
        raise DownstreamError("The upstream MCP server is unavailable.", exc) from None
