"""The gateway itself: authenticate, inspect, authorize, forward.

Request flow for POST:

    Bearer token -> Principal
      -> parse JSON-RPC payload (single or batch)
        -> evaluate every message against the policy
          -> blocked messages are answered here, downstream is never contacted
          -> allowed messages are forwarded
            -> downstream responses merged with local denials

The "downstream is never contacted" part is the security property worth
protecting: a gateway that forwards and then filters the *response* has already
let the side effect happen. `test_blocked_calls_never_reach_downstream` pins it.
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

from mcp_gateway import jsonrpc, policy, proxy
from mcp_gateway.auth import AuthError, Principal, resolve_principal
from mcp_gateway.config import Config

logger = logging.getLogger(__name__)
audit = logging.getLogger("mcp_gateway.audit")


def _json_rpc_response(payload: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status_code)


def _unauthenticated(detail: str) -> JSONResponse:
    """401 with a JSON-RPC error body.

    The HTTP status is what an HTTP client acts on; the JSON-RPC body is what an
    MCP client surfaces. Emitting both means neither layer has to guess.
    `WWW-Authenticate` is required by RFC 9110 for a 401.
    """
    return JSONResponse(
        jsonrpc.error_response(jsonrpc.INVALID_REQUEST, detail),
        status_code=401,
        headers={"WWW-Authenticate": 'Bearer realm="mcp-gateway"'},
    )


def _denial_response(message: jsonrpc.RpcMessage, denial: policy.Denial) -> dict[str, Any] | None:
    """The error object for a blocked message, or None for a notification.

    A notification gets no response even when refused - JSON-RPC §4.1 is
    unconditional about that, and inventing one would desynchronise a client
    that is not expecting a frame.
    """
    if not message.is_request:
        return None
    return jsonrpc.error_response(
        code=denial.code,
        message=denial.message,
        id=message.id,
        data={"method": message.method, "tool": message.tool_name},
    )


def _log_decision(principal: Principal, message: jsonrpc.RpcMessage, denial: policy.Denial | None) -> None:
    """Audit every decision.

    An access-control gateway that cannot say who called what, and whether it
    was allowed, is not auditable - and in practice that log is the thing
    incident response actually needs.
    """
    audit.info(
        "decision=%s subject=%s role=%s method=%s tool=%s%s",
        "DENY" if denial else "ALLOW",
        principal.subject,
        principal.role,
        message.method,
        message.tool_name or "-",
        f" reason={denial.reason}" if denial else "",
    )


def _filter_tools_list(payload: Any, principal: Principal) -> Any:
    """Strip admin-only tools from a tools/list result for non-admins.

    Off by default; see Config.filter_tools_list and the README.
    """
    if principal.is_admin or not isinstance(payload, dict):
        return payload
    result = payload.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
        return payload
    result["tools"] = [
        tool
        for tool in result["tools"]
        if not (isinstance(tool, dict) and isinstance(tool.get("name"), str) and policy.is_privileged_tool(tool["name"]))
    ]
    return payload


async def handle_mcp_post(request: Request) -> Response:
    config: Config = request.app.state.config
    client: httpx.AsyncClient = request.app.state.http_client

    try:
        principal = resolve_principal(request.headers.get("authorization"))
    except AuthError as exc:
        audit.info("decision=DENY subject=- role=- reason=authentication_failed detail=%s", exc)
        return _unauthenticated(str(exc))

    # A body we cannot understand is never forwarded.
    body = await request.body()
    try:
        parsed = jsonrpc.parse_payload(body)
    except jsonrpc.InvalidPayload as exc:
        return _json_rpc_response(jsonrpc.error_response(exc.code, exc.message, exc.id), status_code=400)

    # Every element of a batch is evaluated on its own; see parse_payload.
    allowed: list[jsonrpc.RpcMessage] = []
    denials: list[dict[str, Any]] = []
    for message in parsed.messages:
        denial = policy.evaluate(principal, message)
        _log_decision(principal, message, denial)
        if denial is None:
            allowed.append(message)
        else:
            response = _denial_response(message, denial)
            if response is not None:
                denials.append(response)

    # Nothing survived the policy: answer locally, contact nobody.
    if not allowed:
        if not denials:
            # Everything blocked was a notification, so there is nothing to say.
            return Response(status_code=204)
        return _json_rpc_response(denials if parsed.is_batch else denials[0])

    # With nothing blocked, relay the original bytes rather than re-encoding:
    # a proxy that only inspects should not normalise key order, number
    # formatting or unicode escapes on its way through.
    if len(allowed) == len(parsed.messages):
        forward_body = body
    else:
        payload = [message.raw for message in allowed]
        forward_body = json.dumps(payload if parsed.is_batch else payload[0]).encode()

    headers = proxy.build_downstream_headers(request.headers, principal)
    try:
        downstream = await proxy.forward(
            client, config.downstream_url, forward_body, headers, config.request_timeout_seconds
        )
    except proxy.DownstreamError as exc:
        upstream_error = jsonrpc.error_response(
            jsonrpc.UPSTREAM_UNAVAILABLE,
            exc.summary,
            id=allowed[0].id if allowed[0].is_request else None,
        )
        # Denials still stand even though the forward failed.
        payload = denials + [upstream_error] if parsed.is_batch else upstream_error
        return _json_rpc_response(payload, status_code=502)

    return _build_client_response(downstream, denials, parsed, principal, config)


def _build_client_response(
    downstream: httpx.Response,
    denials: list[dict[str, Any]],
    parsed: jsonrpc.ParsedPayload,
    principal: Principal,
    config: Config,
) -> Response:
    """Relay the downstream response, merging in any locally-generated denials."""
    headers = proxy.filter_response_headers(downstream.headers)
    content_type = downstream.headers.get("content-type", "")
    is_json = content_type.startswith("application/json")

    # Nothing to merge and nothing to rewrite: hand the bytes back untouched.
    # This is also what lets a streaming (text/event-stream) response pass
    # through a gateway that is otherwise JSON-oriented.
    if not denials and not (config.filter_tools_list and is_json):
        return Response(
            content=downstream.content,
            status_code=downstream.status_code,
            headers=headers,
            media_type=content_type or None,
        )

    if not is_json:
        # A merge was required but the response is not JSON we can splice into.
        # Return what we can rather than silently dropping the denials.
        logger.warning("Cannot merge denials into a %r response from downstream", content_type)
        return _json_rpc_response(denials, status_code=downstream.status_code)

    try:
        payload = downstream.json()
    except ValueError:
        logger.warning("Downstream sent malformed JSON while a merge was required")
        return _json_rpc_response(
            denials + [jsonrpc.error_response(jsonrpc.UPSTREAM_UNAVAILABLE, "Malformed response from the upstream MCP server.")],
            status_code=502,
        )

    if config.filter_tools_list:
        if isinstance(payload, list):
            payload = [_filter_tools_list(item, principal) for item in payload]
        else:
            payload = _filter_tools_list(payload, principal)

    if denials:
        downstream_items = payload if isinstance(payload, list) else [payload]
        # Batch responses may arrive in any order; clients correlate by id.
        payload = downstream_items + denials

    headers.pop("content-length", None)
    return _json_rpc_response(payload, status_code=downstream.status_code)


async def handle_mcp_passthrough(request: Request) -> Response:
    """Authenticated relay for GET (server->client stream) and DELETE (session end).

    Neither carries a JSON-RPC body to inspect, so the policy has nothing to act
    on; they are relayed so stateful MCP servers behind the gateway keep working.
    """
    config: Config = request.app.state.config
    client: httpx.AsyncClient = request.app.state.http_client

    try:
        principal = resolve_principal(request.headers.get("authorization"))
    except AuthError as exc:
        return _unauthenticated(str(exc))

    audit.info(
        "decision=ALLOW subject=%s role=%s method=%s (transport passthrough)",
        principal.subject, principal.role, request.method,
    )
    headers = proxy.build_downstream_headers(request.headers, principal)

    upstream = client.build_request(
        request.method, config.downstream_url, headers=headers, timeout=config.request_timeout_seconds
    )
    try:
        response = await client.send(upstream, stream=True)
    except httpx.HTTPError as exc:
        logger.warning("Passthrough %s failed: %r", request.method, exc)
        return _json_rpc_response(
            jsonrpc.error_response(jsonrpc.UPSTREAM_UNAVAILABLE, "The upstream MCP server is unavailable."),
            status_code=502,
        )

    async def stream():
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        finally:
            await response.aclose()

    return StreamingResponse(
        stream(),
        status_code=response.status_code,
        headers=proxy.filter_response_headers(response.headers),
        media_type=response.headers.get("content-type"),
    )


async def handle_health(request: Request) -> Response:
    return JSONResponse({"status": "ok", "downstream": request.app.state.config.downstream_url})


def create_app(config: Config | None = None) -> Starlette:
    config = config or Config.from_env()

    @asynccontextmanager
    async def lifespan(app: Starlette):
        # One pooled client for the process; creating one per request would
        # discard connection reuse and dominate the proxy's latency.
        async with httpx.AsyncClient() as client:
            app.state.http_client = client
            app.state.config = config
            logger.info("Gateway ready, forwarding to %s", config.downstream_url)
            yield

    return Starlette(
        routes=[
            Route("/mcp", handle_mcp_post, methods=["POST"]),
            Route("/mcp", handle_mcp_passthrough, methods=["GET", "DELETE"]),
            Route("/healthz", handle_health, methods=["GET"]),
        ],
        lifespan=lifespan,
    )
