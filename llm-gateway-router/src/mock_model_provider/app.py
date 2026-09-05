"""A mock model provider whose failure mode is controlled by the request.

Every branch the router has to handle - 429, slow, 500, malformed 4xx, healthy -
needs to be reproducible on demand, so the behaviour is a parameter rather than
something to be simulated with real load.

Knobs (read from the request body, all optional):
    behaviour:   "ok" | "rate_limited" | "slow" | "error" | "bad_request"
    delay_ms:    how long to sleep before answering (used by "slow")
    output_tokens / input_tokens: usage to report back
    omit_usage:  answer successfully but report no usage at all

Any knob can be overridden for one provider by prefixing it with that
provider's name: ``{"behaviour": "rate_limited", "fallback_behaviour": "ok"}``
makes the primary fail and the backup succeed. Without this the same body
reaches both providers and every failover test degenerates into "everything
failed".
"""

from __future__ import annotations

import asyncio

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

async def handle_messages(request: Request) -> JSONResponse:
    body = await request.json() if await request.body() else {}
    # Per-instance, not module-level: primary and fallback run as two separate
    # apps, and the tests need to tell which of them was actually called.
    request.app.state.received.append(body)

    name = request.app.state.name

    def knob(key: str, default):
        """Per-provider value if given, else the shared one, else the default."""
        return body.get(f"{name}_{key}", body.get(key, default))

    behaviour = knob("behaviour", "ok")
    delay_ms = float(knob("delay_ms", 0))

    if behaviour == "slow":
        # Long enough that the caller's timeout fires first. The point is to
        # exercise the router's timeout path, not this sleep.
        await asyncio.sleep(delay_ms / 1000 if delay_ms else 30)
    elif delay_ms:
        await asyncio.sleep(delay_ms / 1000)

    if behaviour == "rate_limited":
        return JSONResponse(
            {
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    # Deliberately leaky: names an internal deployment and quota
                    # detail. The gateway must not pass this through.
                    "message": "quota exhausted on deployment prod-eu-3 (tenant acct_9931, internal-llm.prod.svc)",
                },
            },
            status_code=429,
            headers={"retry-after": "42"},
        )

    if behaviour == "error":
        return JSONResponse(
            {
                "type": "error",
                "error": {"type": "api_error", "message": "internal failure at /srv/model/worker.py:412"},
            },
            status_code=500,
        )

    if behaviour == "bad_request":
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens exceeds model limit"}},
            status_code=400,
        )

    response = {
        "id": "msg_mock",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", "unknown"),
        "content": [{"type": "text", "text": f"Answered by {request.app.state.name}."}],
        "stop_reason": "end_turn",
    }
    if not knob("omit_usage", False):
        response["usage"] = {
            "input_tokens": int(knob("input_tokens", 50)),
            "output_tokens": int(knob("output_tokens", 100)),
        }
    return JSONResponse(response)


def create_app(name: str = "mock") -> Starlette:
    """Build one provider instance.

    `app.state.received` records every request it served.
    """
    app = Starlette(routes=[Route("/v1/messages", handle_messages, methods=["POST"])])
    app.state.name = name
    app.state.received = []
    return app


app = create_app()
