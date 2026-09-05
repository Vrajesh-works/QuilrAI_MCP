"""A mock LLM provider emitting Anthropic-shaped SSE.

Exists so the guardrail can be tested deterministically: the text, the chunk
boundaries and the per-token delay are all controlled by the request, which is
what makes "PII split across chunks" reproducible rather than a matter of luck.

Request knobs (all optional):
    text:          what to "generate"
    chunk_size:    characters per delta; 1 is the worst case for the redactor
    delay_ms:      pause between deltas, for TTFT measurement
    stream:        false returns a normal JSON response instead
"""

from __future__ import annotations

import asyncio
import json

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

DEFAULT_TEXT = (
    "Sure. You can reach Ada at ada.lovelace@example.com or on (555) 123-4567. "
    "Her SSN is 123-45-6789 and the card on file is 4111 1111 1111 1111. "
    "Order 1234567890123456 shipped Tuesday."
)


def _event(name: str, payload: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _chunks(text: str, size: int) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)] or [""]


async def handle_messages(request: Request) -> StreamingResponse | JSONResponse:
    body = await request.json() if await request.body() else {}
    text = body.get("text", DEFAULT_TEXT)
    chunk_size = int(body.get("chunk_size", 8))
    delay_ms = float(body.get("delay_ms", 0))

    if body.get("stream") is False:
        return JSONResponse(
            {
                "id": "msg_mock",
                "type": "message",
                "role": "assistant",
                "model": body.get("model", "claude-sonnet-5"),
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 12, "output_tokens": len(text) // 4},
            }
        )

    async def generate():
        yield _event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_mock",
                    "type": "message",
                    "role": "assistant",
                    "model": body.get("model", "claude-sonnet-5"),
                    "content": [],
                    "usage": {"input_tokens": 12, "output_tokens": 0},
                },
            },
        )
        yield _event(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        )

        for piece in _chunks(text, chunk_size):
            if delay_ms:
                await asyncio.sleep(delay_ms / 1000)
            yield _event(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}},
            )

        yield _event("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield _event(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": len(text) // 4}},
        )
        yield _event("message_stop", {"type": "message_stop"})

    return StreamingResponse(generate(), media_type="text/event-stream")


def create_app() -> Starlette:
    return Starlette(routes=[Route("/v1/messages", handle_messages, methods=["POST"])])


app = create_app()
