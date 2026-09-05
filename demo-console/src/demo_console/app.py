"""A browser console for driving the three gateways.

The services are middleware and have no user-facing surface of their own. This
serves one page that exercises them, so behaviour that a curl dump flattens -
redactions appearing partway through a stream, a role toggle flipping a call
from allowed to blocked - can actually be watched.

Requests to the gateways are proxied through this service so the browser stays
on a single origin; the gateways themselves need no CORS configuration.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

STATIC = Path(__file__).parent / "static"

# Demo credentials, matching the gateway's built-in table.
ROLE_TOKENS = {
    "viewer": "viewer-token-xyz789",
    "admin": "admin-token-abc123",
}


class Targets:
    """Where the upstream services live.

    Defaults are the localhost ports used when running with `uv`; compose
    overrides them with service names.
    """

    def __init__(self) -> None:
        self.mcp_gateway = os.environ.get("CONSOLE_MCP_GATEWAY", "http://127.0.0.1:8080")
        self.mcp_downstream = os.environ.get("CONSOLE_MCP_DOWNSTREAM", "http://127.0.0.1:8081")
        self.guardrail = os.environ.get("CONSOLE_GUARDRAIL", "http://127.0.0.1:8090")
        self.provider = os.environ.get("CONSOLE_PROVIDER", "http://127.0.0.1:8091")
        self.router = os.environ.get("CONSOLE_ROUTER", "http://127.0.0.1:8100")


async def index(request: Request) -> FileResponse:
    return FileResponse(STATIC / "index.html")


async def mcp_call(request: Request) -> Response:
    """Forward a JSON-RPC payload as the chosen role, and report what happened.

    The interesting part is not the response but `downstream_calls`: a blocked
    call must leave the downstream server's request log untouched. Reading that
    log is what turns "we returned -32001" into "the tool never ran".
    """
    targets: Targets = request.app.state.targets
    client: httpx.AsyncClient = request.app.state.client

    body = await request.json()
    role = body.get("role", "viewer")
    payload = body.get("payload")

    before = await _downstream_log(client, targets)
    headers = {"content-type": "application/json"}
    if role in ROLE_TOKENS:
        headers["authorization"] = f"Bearer {ROLE_TOKENS[role]}"

    try:
        response = await client.post(
            f"{targets.mcp_gateway}/mcp", json=payload, headers=headers, timeout=15
        )
    except httpx.HTTPError as exc:
        return JSONResponse({"error": f"gateway unreachable: {type(exc).__name__}"}, status_code=502)

    after = await _downstream_log(client, targets)

    try:
        parsed = response.json()
    except ValueError:
        parsed = {"raw": response.text}

    return JSONResponse(
        {
            "status": response.status_code,
            "body": parsed,
            "downstream_calls": [
                _describe(message) for message in after[len(before) :]
            ],
        }
    )


def _describe(message: Any) -> str:
    if not isinstance(message, dict):
        return "?"
    method = message.get("method", "?")
    params = message.get("params")
    if isinstance(params, dict) and isinstance(params.get("name"), str):
        return f"{method} -> {params['name']}"
    return str(method)


async def _downstream_log(client: httpx.AsyncClient, targets: Targets) -> list:
    """Snapshot of what the downstream MCP server has been asked to do."""
    try:
        response = await client.get(f"{targets.mcp_downstream}/_debug/received", timeout=5)
        return response.json()
    except (httpx.HTTPError, ValueError):
        # The console still works without it; the panel just cannot prove the
        # negative, so it says so rather than showing a misleading empty list.
        return []


async def guardrail_stream(request: Request) -> Response:
    """Relay one SSE stream, either through the guardrail or straight from the
    provider, so the page can show redacted and raw side by side."""
    targets: Targets = request.app.state.targets
    client: httpx.AsyncClient = request.app.state.client

    body = await request.json()
    target = targets.provider if body.get("raw") else targets.guardrail
    payload = {
        "stream": True,
        "text": body.get("text", ""),
        "chunk_size": int(body.get("chunk_size", 3)),
        "delay_ms": float(body.get("delay_ms", 25)),
    }

    async def relay():
        try:
            async with client.stream(
                "POST", f"{target}/v1/messages", json=payload, timeout=httpx.Timeout(60)
            ) as upstream:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
        except httpx.HTTPError as exc:
            error = json.dumps({"error": f"{type(exc).__name__}"})
            yield f"event: error\ndata: {error}\n\n".encode()

    return StreamingResponse(relay(), media_type="text/event-stream")


async def router_call(request: Request) -> Response:
    targets: Targets = request.app.state.targets
    client: httpx.AsyncClient = request.app.state.client

    body = await request.json()
    tenant = body.get("tenant", "sk-tenant-alpha")
    payload = {
        "messages": [{"role": "user", "content": body.get("prompt", "Summarise the incident report.")}],
        "max_tokens": int(body.get("max_tokens", 500)),
        "output_tokens": int(body.get("output_tokens", 100)),
    }
    behaviour = body.get("behaviour", "ok")
    if behaviour != "ok":
        payload["behaviour"] = behaviour
        # Only the primary misbehaves unless the caller asked for both, so the
        # failover path is the one being demonstrated.
        if not body.get("both_fail"):
            payload["fallback_behaviour"] = "ok"
            payload["fallback_delay_ms"] = 0

    try:
        response = await client.post(
            f"{targets.router}/v1/messages",
            json=payload,
            headers={"x-api-key": tenant, "content-type": "application/json"},
            timeout=30,
        )
    except httpx.HTTPError as exc:
        return JSONResponse({"error": f"router unreachable: {type(exc).__name__}"}, status_code=502)

    try:
        parsed = response.json()
    except ValueError:
        parsed = {"raw": response.text}

    return JSONResponse(
        {
            "status": response.status_code,
            "body": parsed,
            "provider": response.headers.get("x-gateway-provider"),
            "attempts": response.headers.get("x-gateway-attempts"),
            "retry_after": response.headers.get("retry-after"),
            "limit": response.headers.get("x-ratelimit-limit-tokens"),
            "remaining": response.headers.get("x-ratelimit-remaining-tokens"),
        }
    )


async def router_usage(request: Request) -> Response:
    targets: Targets = request.app.state.targets
    client: httpx.AsyncClient = request.app.state.client
    tenant = request.query_params.get("tenant", "sk-tenant-alpha")

    try:
        response = await client.get(
            f"{targets.router}/v1/usage", headers={"x-api-key": tenant}, timeout=10
        )
        return JSONResponse(response.json())
    except (httpx.HTTPError, ValueError) as exc:
        return JSONResponse({"error": type(exc).__name__}, status_code=502)


async def health(request: Request) -> Response:
    """Which upstreams are actually reachable, so the page can say so up front."""
    targets: Targets = request.app.state.targets
    client: httpx.AsyncClient = request.app.state.client

    async def probe(name: str, url: str) -> tuple[str, bool]:
        try:
            response = await client.get(url, timeout=3)
            return name, response.status_code < 500
        except httpx.HTTPError:
            return name, False

    checks = [
        await probe("mcp-gateway", f"{targets.mcp_gateway}/healthz"),
        await probe("guardrail", f"{targets.guardrail}/healthz"),
        await probe("router", f"{targets.router}/healthz"),
    ]
    return JSONResponse(dict(checks))


def create_app() -> Starlette:
    @asynccontextmanager
    async def lifespan(app: Starlette):
        async with httpx.AsyncClient() as client:
            app.state.client = client
            app.state.targets = Targets()
            yield

    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/health", health),
            Route("/api/mcp", mcp_call, methods=["POST"]),
            Route("/api/guardrail", guardrail_stream, methods=["POST"]),
            Route("/api/router", router_call, methods=["POST"]),
            Route("/api/router/usage", router_usage),
        ],
        lifespan=lifespan,
    )
