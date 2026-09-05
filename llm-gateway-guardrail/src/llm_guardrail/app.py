"""The LLM gateway: proxy text generation, redacting PII on the way back.

Streaming is the interesting path. The response is relayed as a
`StreamingResponse` over an httpx streaming request, so bytes move
upstream-to-client continuously and neither side buffers the whole body.
"""

from __future__ import annotations

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
from llm_guardrail.redactor import StreamRedactor, redact_text
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
    """Redact the text blocks of a non-streaming Messages response."""
    if not isinstance(payload, dict) or not isinstance(payload.get("content"), list):
        return payload
    for block in payload["content"]:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            block["text"] = redact_text(block["text"])
    return payload


async def handle_messages(request: Request) -> Response:
    config: Config = request.app.state.config
    client: httpx.AsyncClient = request.app.state.http_client

    body = await request.body()
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
            yield

    return Starlette(
        routes=[
            Route("/v1/messages", handle_messages, methods=["POST"]),
            Route("/healthz", handle_health, methods=["GET"]),
        ],
        lifespan=lifespan,
    )
