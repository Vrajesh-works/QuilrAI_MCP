"""Latency and memory: the guardrail must not turn a stream into a batch.

A redactor that buffers the whole response would pass every correctness test in
this suite and still be useless, because it converts time-to-first-token into
time-to-*last*-token. These tests fail if that regression is ever introduced.

Timing assertions use wide relative margins - the claim is "streams incrementally"
rather than any absolute millisecond budget, so an overloaded machine slows both
sides of the comparison equally.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from conftest import request_body
from llm_guardrail.redactor import StreamRedactor
from llm_guardrail.sse import SSEParser
from llm_guardrail.stream import redact_sse_stream

pytestmark = pytest.mark.asyncio

CHUNKS = 40
DELAY_MS = 10


def _delta_event(text: str) -> bytes:
    payload = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}
    return f"event: content_block_delta\ndata: {json.dumps(payload)}\n\n".encode()


async def _slow_upstream(chunks: int, delay_ms: int = DELAY_MS):
    """An upstream that trickles deltas, like a model generating tokens."""
    for index in range(chunks):
        await asyncio.sleep(delay_ms / 1000)
        yield _delta_event(f"word{index} ")
    yield b"event: content_block_stop\ndata: {}\n\n"


async def _first_and_total_through_redactor(chunks: int) -> tuple[float, float]:
    """Seconds until the guardrail emits its first byte, and until it finishes."""
    start = time.perf_counter()
    first: float | None = None

    async for chunk in redact_sse_stream(_slow_upstream(chunks), StreamRedactor()):
        if first is None and b"text_delta" in chunk:
            first = time.perf_counter() - start

    return (first if first is not None else float("inf"), time.perf_counter() - start)


async def test_first_token_arrives_long_before_the_stream_ends():
    """The defining property: TTFT tracks the upstream, not the response length."""
    ttft, total = await _first_and_total_through_redactor(CHUNKS)

    assert total > 0.1, "the upstream should have been slow enough to measure"
    assert ttft < total / 3, (
        f"first token took {ttft:.3f}s of a {total:.3f}s stream - "
        "the guardrail looks like it is buffering the whole response"
    )


async def test_ttft_does_not_grow_with_response_length():
    """Doubling the response must not move the first token.

    This is what separates incremental redaction from buffer-then-redact: under
    buffering, TTFT scales with total length.
    """
    short_ttft, _ = await _first_and_total_through_redactor(10)
    long_ttft, long_total = await _first_and_total_through_redactor(CHUNKS * 2)

    assert long_ttft < short_ttft + (long_total / 4), (
        f"TTFT grew from {short_ttft:.3f}s to {long_ttft:.3f}s as the response got longer"
    )


async def test_the_deployed_http_path_streams_too(live_guardrail):
    """The same property over real sockets, not just the transformer.

    Worth its own test: the transformer streaming does not by itself prove the
    Starlette/httpx path around it does, and buffering anywhere in that chain
    would be just as damaging.
    """
    parser = SSEParser()
    start = time.perf_counter()
    first: float | None = None

    body = request_body("word " * CHUNKS, chunk_size=5, delay_ms=DELAY_MS)
    async with live_guardrail.stream("POST", "/v1/messages", json=body) as response:
        async for chunk in response.aiter_bytes():
            for event in parser.feed(chunk):
                payload = event.json()
                if (
                    first is None
                    and isinstance(payload, dict)
                    and payload.get("delta", {}).get("text")
                ):
                    first = time.perf_counter() - start
    total = time.perf_counter() - start

    assert first is not None, "no text delta ever arrived"
    assert total > 0.1
    assert first < total / 2, (
        f"first token took {first:.3f}s of a {total:.3f}s stream over real sockets"
    )


async def test_clean_text_is_never_held_back():
    """Prose with no PII should stream through with an empty holdback.

    If ordinary text were held, every response would carry the extra latency of
    the guardrail rather than only the ones containing something risky.
    """
    redactor = StreamRedactor()
    peak = 0
    for word in ("The ", "quick ", "brown ", "fox ", "jumps ", "over ", "the ", "lazy ", "dog. "):
        redactor.feed(word)
        peak = max(peak, len(redactor.pending))

    # Only the trailing partial word is ever held, and only until the next space.
    assert peak <= len("jumps ")


async def test_memory_is_constant_over_a_long_stream():
    """Peak holdback must be bounded by the pattern set, not the stream length."""
    redactor = StreamRedactor()
    for index in range(5_000):
        redactor.feed(f"sentence number {index} with nothing sensitive in it. ")
    redactor.flush()

    assert redactor.stats.characters_in > 200_000, "the stream should have been long"
    assert redactor.stats.max_holdback_seen < 64, (
        f"held up to {redactor.stats.max_holdback_seen} chars; memory is not constant"
    )


async def test_throughput_is_not_quadratic():
    """Cost per chunk must not grow with how much has already been streamed.

    Rescanning the whole response on each chunk is the classic way to get this
    wrong: correct output, quadratic time, and a stream that visibly stalls as it
    gets longer.
    """
    def elapsed_for(chunks: int) -> float:
        redactor = StreamRedactor()
        start = time.perf_counter()
        for index in range(chunks):
            redactor.feed(f"some ordinary words number {index}. ")
        redactor.flush()
        return time.perf_counter() - start

    small = elapsed_for(1_000)
    large = elapsed_for(8_000)

    # Linear would be ~8x. Quadratic would be ~64x. A generous ceiling still
    # separates the two decisively.
    assert large < small * 24, f"8x the input took {large / small:.1f}x the time"
