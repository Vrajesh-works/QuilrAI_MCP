"""HTTP surface for the router."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from llm_router.config import Config
from llm_router.errors import INVALID_REQUEST, gateway_error
from llm_router.ratelimit import RateLimiter
from llm_router.router import Router
from llm_router.store import Store

logger = logging.getLogger(__name__)

# Headers that must not be relayed upstream: the caller's credential is replaced
# by the gateway's own per-provider key, and lengths change when the body is
# rewritten with the provider's model name.
_DROP_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "keep-alive", "authorization", "x-api-key"}


def _tenant_of(request: Request) -> str | None:
    """The tenant API key identifying who to bill.

    Accepts either header the Anthropic SDKs send. Rate limiting is per key, so
    a request without one cannot be accounted for and is refused.
    """
    header = request.headers.get("x-api-key") or request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        header = header[7:]
    return header.strip() or None


async def handle_messages(request: Request) -> Response:
    router: Router = request.app.state.router

    tenant = _tenant_of(request)
    if tenant is None:
        return JSONResponse(
            gateway_error(INVALID_REQUEST, "An API key is required, via x-api-key or Authorization: Bearer."),
            status_code=401,
        )

    # INCOMPLETE: `stream: true` is accepted and forwarded, but usage arrives in
    # a terminal message_delta event rather than a JSON body, so `actual_tokens`
    # finds none and the request settles at its estimate instead of its real
    # cost. Correct accounting needs the reservation settled from the parsed
    # stream; llm-gateway-guardrail has the SSE parser this would build on.
    raw = await request.body()
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
    result = await router.route(tenant, body, forwarded)

    return JSONResponse(result.body, status_code=result.status_code, headers=result.headers)


async def handle_usage(request: Request) -> Response:
    """Current window usage for the calling key. Useful for demos and clients."""
    tenant = _tenant_of(request)
    if tenant is None:
        return JSONResponse(gateway_error(INVALID_REQUEST, "An API key is required."), status_code=401)

    limiter: RateLimiter = request.app.state.limiter
    used = await limiter.usage(tenant)
    return JSONResponse({"used_tokens": used, "limit_tokens": limiter.limit, "remaining_tokens": max(0, limiter.limit - used)})


async def handle_health(request: Request) -> Response:
    router: Router = request.app.state.router
    return JSONResponse({"status": "ok", "providers": [provider.name for provider in router.providers]})


def create_app(config: Config | None = None) -> Starlette:
    config = config or Config.from_env()

    @asynccontextmanager
    async def lifespan(app: Starlette):
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
