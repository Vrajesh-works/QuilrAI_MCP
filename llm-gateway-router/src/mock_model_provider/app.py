"""A mock model provider whose failure mode is controlled by the request.

Every branch the router has to handle - 429, slow, 500, malformed 4xx, healthy -
needs to be reproducible on demand, so the behaviour is a parameter rather than
something to be simulated with real load.

Knobs (read from the request body, all optional):
    behaviour:   "ok" | "rate_limited" | "slow" | "error" | "bad_request"
                 | "unauthorized" | "forbidden"
    delay_ms:    how long to sleep before answering (used by "slow")
    output_tokens / input_tokens: usage to report back
    omit_usage:  answer successfully but report no usage at all
    stream:      answer with `text/event-stream` in the Anthropic shape
    stream_chunks: how many text deltas to emit (default 3)
    stream_gap_ms: pause between deltas, so streaming can be shown to be
                 incremental rather than buffered

`unauthorized`/`forbidden` and `stream` exist because the router had defects
neither could be reached without them: a 401 from a provider was classified as
the *caller's* error and suppressed failover, and a streaming response was
parsed with `response.json()`, discarded, and reported as a 200 with an empty
body. Both were invisible to a suite whose mocks always returned JSON 200.

Any knob can be overridden for one provider by prefixing it with that
provider's name: ``{"behaviour": "rate_limited", "fallback_behaviour": "ok"}``
makes the primary fail and the backup succeed. Without this the same body
reaches both providers and every failover test degenerates into "everything
failed".
"""

from __future__ import annotations

import asyncio

import json

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route


def _stream_response(request: Request, body: dict, knob) -> StreamingResponse:
    """An Anthropic-shaped SSE stream, usage included in the terminal events."""
    name = request.app.state.name
    chunks = int(knob("stream_chunks", 3))
    gap_ms = float(knob("stream_gap_ms", 0))
    input_tokens = int(knob("input_tokens", 50))
    output_tokens = int(knob("output_tokens", 100))

    async def events():
        def frame(event: str, payload: dict) -> bytes:
            return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()

        yield frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_mock_stream",
                    "role": "assistant",
                    "model": body.get("model", "unknown"),
                    "content": [],
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            },
        )
        yield frame("content_block_start", {"type": "content_block_start", "index": 0,
                                            "content_block": {"type": "text", "text": ""}})
        for index in range(chunks):
            if gap_ms:
                await asyncio.sleep(gap_ms / 1000)
            yield frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": f"chunk-{index} from {name}. "},
                },
            )
        yield frame("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield frame(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
             "usage": {"output_tokens": output_tokens}},
        )
        yield frame("message_stop", {"type": "message_stop"})

    return StreamingResponse(events(), media_type="text/event-stream")

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

    if behaviour in ("unauthorized", "forbidden"):
        return JSONResponse(
            {
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    # Also deliberately leaky.
                    "message": "invalid x-api-key for org org_88213 (key sk-live-9f3a...b21, vault prod/eu)",
                },
            },
            status_code=401 if behaviour == "unauthorized" else 403,
        )

    if knob("stream", False):
        return _stream_response(request, body, knob)

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
