"""Show the guardrail redacting a live stream, chunk by chunk.

    uv run python scripts/demo.py

Prints what the provider emitted against what the client received, so the
boundary-splitting cases are visible: watch an email arrive as `ada@ex` +
`ample.com` and still come out as [REDACTED].
"""

from __future__ import annotations

import asyncio
import json
import time

from llm_guardrail.redactor import StreamRedactor
from llm_guardrail.stream import redact_sse_stream

LEAKY_TEXT = (
    "Sure. Ada's address is ada.lovelace@example.com and her direct line is "
    "(555) 123-4567. The SSN on file is 123-45-6789 and the card ending in 1111 "
    "is 4111 1111 1111 1111. Her internal key is sk-ant-abcdefghij1234567890. "
    "Order 1234567890123456 shipped Tuesday and is not sensitive."
)


def delta(text: str) -> bytes:
    payload = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}
    return f"event: content_block_delta\ndata: {json.dumps(payload)}\n\n".encode()


async def provider(text: str, chunk_size: int, delay_ms: float = 4):
    for index in range(0, len(text), chunk_size):
        await asyncio.sleep(delay_ms / 1000)
        yield delta(text[index : index + chunk_size])
    yield b"event: content_block_stop\ndata: {}\n\n"


def text_of(chunk: bytes) -> str:
    pieces = []
    for line in chunk.decode().splitlines():
        if line.startswith("data: "):
            try:
                payload = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("delta", {}).get("type") == "text_delta":
                pieces.append(payload["delta"]["text"])
    return "".join(pieces)


async def run(chunk_size: int) -> None:
    print("=" * 78)
    print(f"  Provider emitting {chunk_size} character(s) per delta")
    print("=" * 78)

    redactor = StreamRedactor()
    received: list[str] = []
    start = time.perf_counter()
    first: float | None = None

    async for chunk in redact_sse_stream(provider(LEAKY_TEXT, chunk_size), redactor):
        piece = text_of(chunk)
        if piece:
            if first is None:
                first = time.perf_counter() - start
            received.append(piece)

    total = time.perf_counter() - start
    output = "".join(received)

    print(f"client received:\n  {output}\n")
    print(f"redacted:      {redactor.stats.counts}")
    print(f"peak holdback: {redactor.stats.max_holdback_seen} chars (memory is bounded)")
    print(f"TTFT:          {first * 1000:.1f}ms of a {total * 1000:.1f}ms stream")

    leaked = [
        secret
        for secret in (
            "ada.lovelace@example.com", "123-45-6789", "4111 1111 1111 1111",
            "(555) 123-4567", "sk-ant-abcdefghij1234567890",
        )
        if secret in output
    ]
    print(f"leaked:        {leaked or 'nothing'}")
    # The non-Luhn order number is deliberately left alone.
    print(f"order number kept: {'1234567890123456' in output}")
    print()


async def main() -> None:
    # 1 char per delta is the worst case: every secret is split many times.
    for chunk_size in (1, 5, 40):
        await run(chunk_size)


if __name__ == "__main__":
    asyncio.run(main())
