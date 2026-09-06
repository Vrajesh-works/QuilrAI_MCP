"""HTTP surface for the router."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from llm_router.auth import AuthError, Tenant, extract_api_key, resolve_tenant
from llm_router.config import Config
from llm_router.errors import INVALID_REQUEST, gateway_error
from llm_router.ratelimit import RateLimiter
from llm_router.router import Router
from llm_router.store import Store

logger = logging.getLogger(__name__)

# Headers that must not be relayed upstream: the caller's credential is dropped
# here and replaced in `call_provider` by the gateway's own per-provider key
# (`Provider.api_key`), and lengths change when the body is rewritten with the
# provider's model name.
_DROP_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "keep-alive", "authorization", "x-api-key"}

# A Messages request is prose and tool definitions, not a file upload.
MAX_BODY_BYTES = 4_194_304


class _BodyTooLarge(Exception):
    """The request body exceeded the configured cap."""


async def _read_bounded_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise _BodyTooLarge
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise _BodyTooLarge
    return bytes(body)


def _authenticate(request: Request) -> Tenant:
    """Resolve the caller's API key to an opaque tenant id.

    Raises:
        AuthError: no key, or a key that is not recognised.
    """
    config: Config = request.app.state.config
    api_key = extract_api_key(request.headers.get("x-api-key"), request.headers.get("authorization"))
    return resolve_tenant(
        api_key, config.tenants, allow_unauthenticated=config.allow_unauthenticated_tenants
    )


def _unauthenticated(exc: AuthError) -> JSONResponse:
    return JSONResponse(gateway_error(INVALID_REQUEST, str(exc)), status_code=401)


async def handle_messages(request: Request) -> Response:
    router: Router = request.app.state.router

    try:
        tenant = _authenticate(request)
    except AuthError as exc:
        return _unauthenticated(exc)

    try:
        raw = await _read_bounded_body(request, MAX_BODY_BYTES)
    except _BodyTooLarge:
        return JSONResponse(
            gateway_error(INVALID_REQUEST, "Request body exceeds the configured limit."), status_code=413
        )

    try:
        body = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return JSONResponse(gateway_error(INVALID_REQUEST, "Request body must be valid JSON."), status_code=400)

    if not isinstance(body, dict) or not body.get("messages"):
        return JSONResponse(
            gateway_error(INVALID_REQUEST, "Request body must be an object containing 'messages'."),
            status_code=400,
        )

    forwarded = {key: value for key, value in request.headers.items() if key.lower() not in _DROP_HEADERS}
    result = await router.route(tenant.id, body, forwarded)

    if result.is_stream:
        # A streaming completion is relayed as a stream. Returning
        # `JSONResponse(result.body)` here is what produced HTTP 200 with an
        # empty object and threw the completion away.
        return StreamingResponse(
            result.stream,
            status_code=result.status_code,
            headers=result.headers,
            media_type=result.media_type or "text/event-stream",
        )

    return JSONResponse(result.body, status_code=result.status_code, headers=result.headers)


async def handle_usage(request: Request) -> Response:
    """Current window usage for the calling key. Useful for demos and clients."""
    try:
        tenant = _authenticate(request)
    except AuthError as exc:
        return _unauthenticated(exc)

    limiter: RateLimiter = request.app.state.limiter
    used = await limiter.usage(tenant.id)
    return JSONResponse({"used_tokens": used, "limit_tokens": limiter.limit, "remaining_tokens": max(0, limiter.limit - used)})


async def handle_health(request: Request) -> Response:
    router: Router = request.app.state.router
    return JSONResponse({"status": "ok", "providers": [provider.name for provider in router.providers]})


def create_app(config: Config | None = None) -> Starlette:
    config = config or Config.from_env()

    @asynccontextmanager
    async def lifespan(app: Starlette):
        # Before anything is served. A configuration that cannot work should
        # fail here, loudly and once, rather than per-request in production.
        config.validate()
        store = Store(config.database_path)
        limiter = RateLimiter(store, limit=config.token_limit, window_seconds=config.window_seconds)
        async with httpx.AsyncClient() as client:
            app.state.store = store
            app.state.limiter = limiter
            app.state.config = config
            app.state.router = Router(limiter, config.primary, config.fallbacks, client)
            logger.info(
                "Router ready: %d tokens/%.0fs, primary=%s fallbacks=%s db=%s",
                config.token_limit, config.window_seconds, config.primary.name,
                [provider.name for provider in config.fallbacks], config.database_path,
            )
            try:
                yield
            finally:
                store.close()

    return Starlette(
        routes=[
            Route("/v1/messages", handle_messages, methods=["POST"]),
            Route("/v1/usage", handle_usage, methods=["GET"]),
            Route("/healthz", handle_health, methods=["GET"]),
        ],
        lifespan=lifespan,
    )
