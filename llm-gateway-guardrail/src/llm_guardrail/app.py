"""The LLM gateway: proxy text generation, redacting PII on the way back.

Streaming is the interesting path. The response is relayed as a
`StreamingResponse` over an httpx streaming request, so bytes move
upstream-to-client continuously and neither side buffers the whole body.
"""

from __future__ import annotations

import hmac
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from llm_guardrail.config import Config
from llm_guardrail.redactor import StreamRedactor, redact_leaves
from llm_guardrail.stream import redact_sse_stream

logger = logging.getLogger(__name__)

# Hop-by-hop headers (RFC 9110 §7.6.1) plus lengths that no longer hold once the
# body has been rewritten - redaction changes the byte count.
_DROP_REQUEST_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "keep-alive", "upgrade"}
_DROP_RESPONSE_HEADERS = {"content-length", "content-encoding", "connection", "transfer-encoding", "keep-alive"}


def _forward_headers(request: Request) -> dict[str, str]:
    """Relay the caller's headers, including their provider credential.

    Unlike the MCP gateway, this proxy does not terminate authentication: it is
    a transparent path to the provider, and the caller's own API key is what
    authorises the upstream call.
    """
    return {key: value for key, value in request.headers.items() if key.lower() not in _DROP_REQUEST_HEADERS}


def _response_headers(upstream: httpx.Response) -> dict[str, str]:
    return {key: value for key, value in upstream.headers.items() if key.lower() not in _DROP_RESPONSE_HEADERS}


def _redact_message_body(payload: Any) -> Any:
    """Redact a non-streaming response body, failing closed on any shape.

    This path carries the same obligation as the SSE path and gets the same
    default. Matching a known shape first - Anthropic's ``content[]`` text
    blocks, say - and returning everything else untouched makes silence and
    safety the same value, which is the wrong failure direction for a security
    control. It relays, in full and with a 200:

    * ``choices[].message.content`` from any OpenAI-compatible upstream
    * ``content`` sent as a bare string rather than a list of blocks
    * ``tool_use`` blocks, whose ``input`` carries the identifiers being redacted
    * upstream error bodies, a common place for identifiers to appear

    So every string leaf is redacted regardless of shape. Over-redaction of a
    field like ``model`` or ``stop_reason`` is possible in principle and
    harmless in practice - none of the patterns match that vocabulary - and it
    is the direction a guardrail should be wrong in.
    """
    return redact_leaves(payload)


class _BodyTooLarge(Exception):
    """The request body exceeded the configured cap."""


async def _read_bounded_body(request: Request, limit: int) -> bytes:
    """Read the body, refusing to buffer more than `limit` bytes.

    `await request.body()` applies no cap, which combined with the absence of a
    concurrency limit is the cheapest available memory-exhaustion path. The
    declared `Content-Length` is checked first, but the streaming check is the
    one that matters: that header is client-supplied, and a chunked request does
    not carry one at all.
    """
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise _BodyTooLarge
    collected = bytearray()
    async for chunk in request.stream():
        collected.extend(chunk)
        if len(collected) > limit:
            raise _BodyTooLarge
    return bytes(collected)


def _authorized(request: Request, config: Config) -> bool:
    """Whether this caller may use the proxy.

    Returns True unconditionally when no tokens are configured - see
    `Config.describe_trust_boundary` for why that is a deliberate deployment
    mode rather than a missing check, and what is logged at startup about it.
    """
    if not config.requires_authentication:
        return True
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer":
        return False
    candidate = token.strip().encode("utf-8")
    # Constant-time against every entry, on bytes so a non-ASCII token is a
    # rejection rather than a TypeError.
    return any(hmac.compare_digest(candidate, known.encode("utf-8")) for known in config.tokens)


async def handle_messages(request: Request) -> Response:
    config: Config = request.app.state.config
    client: httpx.AsyncClient = request.app.state.http_client

    if not _authorized(request, config):
        return JSONResponse(
            {"type": "error", "error": {"type": "authentication_error", "message": "Invalid or missing bearer token."}},
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer realm="llm-guardrail"'},
        )

    try:
        body = await _read_bounded_body(request, config.max_body_bytes)
    except _BodyTooLarge:
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_request_error", "message": "Request body is too large."}},
            status_code=413,
        )
    try:
        wants_stream = json.loads(body).get("stream", True) if body else True
    except (json.JSONDecodeError, AttributeError):
        wants_stream = True

    timeout = httpx.Timeout(
        connect=config.request_timeout_seconds,
        read=config.read_timeout_seconds,
        write=config.request_timeout_seconds,
        pool=config.request_timeout_seconds,
    )
    upstream_request = client.build_request(
        "POST", config.upstream_url, content=body, headers=_forward_headers(request), timeout=timeout
    )

    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        logger.warning("Upstream request failed: %r", exc)
        return JSONResponse(
            {"type": "error", "error": {"type": "api_error", "message": "The upstream model provider is unavailable."}},
            status_code=502,
        )

    if not wants_stream or not upstream.headers.get("content-type", "").startswith("text/event-stream"):
        # Non-streaming, or an error body: read it, redact it, return it.
        try:
            raw = await upstream.aread()
        finally:
            await upstream.aclose()
        try:
            payload = _redact_message_body(json.loads(raw))
        except json.JSONDecodeError:
            return Response(content=raw, status_code=upstream.status_code, headers=_response_headers(upstream))
        return JSONResponse(payload, status_code=upstream.status_code, headers=_response_headers(upstream))

    redactor = StreamRedactor()

    async def relay():
        try:
            async for chunk in redact_sse_stream(upstream.aiter_bytes(), redactor):
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        relay(),
        status_code=upstream.status_code,
        headers=_response_headers(upstream),
        media_type="text/event-stream",
    )


async def handle_health(request: Request) -> Response:
    return JSONResponse({"status": "ok", "upstream": request.app.state.config.upstream_url})


def create_app(config: Config | None = None) -> Starlette:
    config = config or Config.from_env()

    @asynccontextmanager
    async def lifespan(app: Starlette):
        async with httpx.AsyncClient() as client:
            app.state.http_client = client
            app.state.config = config
            logger.info("Guardrail ready, proxying to %s", config.upstream_url)
            # The deployment posture goes in the log on every start, so "we
            # thought it was internal" is a checkable claim rather than an
            # assumption.
            logger.warning("Trust boundary: %s", config.describe_trust_boundary())
            yield

    return Starlette(
        routes=[
            Route("/v1/messages", handle_messages, methods=["POST"]),
            Route("/healthz", handle_health, methods=["GET"]),
        ],
        lifespan=lifespan,
    )
