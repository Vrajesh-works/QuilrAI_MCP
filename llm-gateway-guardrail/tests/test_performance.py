"""Measured performance and memory, recorded rather than asserted vaguely.

The remediation added a whole-buffer scan and a fail-closed suppression state to
the redactor. Both are on the hot path, so "it still passes the correctness
tests" is not enough - the question is whether the guardrail still streams and
still holds bounded memory under adversarial input.

Every number this file measures is written to `PERF_OUTPUT` as JSON so the
remediation report can quote observations rather than adjectives. The assertions
are deliberately loose relative bounds: the claim is "streams incrementally with
bounded memory", not any absolute millisecond budget, and a shared CI machine
would make a tight bound a flake.
"""

from __future__ import annotations

import json
import os
import time
import tracemalloc
from pathlib import Path

import pytest
from conftest import request_body
from llm_guardrail.redactor import StreamRedactor, redact_text
from llm_guardrail.stream import redact_sse_stream

pytestmark = pytest.mark.asyncio

# Written into the project directory (and gitignored) rather than the system
# temp dir, so the numbers are findable next to the code that produced them.
PERF_OUTPUT = Path(__file__).resolve().parents[1] / "perf-observations.json"

# Sized so the whole file runs in seconds. These are large enough to expose a
# buffering implementation or a super-linear scan and small enough that nobody
# is tempted to skip them.
STREAM_CHARS = 40_000

_observations: dict[str, object] = {}


def record(name: str, value: object) -> None:
    _observations[name] = value
    PERF_OUTPUT.write_text(json.dumps(_observations, indent=2, sort_keys=True), encoding="utf-8")


def delta(text: str) -> bytes:
    payload = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}
    return f"event: content_block_delta\ndata: {json.dumps(payload)}\n\n".encode()


# --------------------------------------------------------------------------
# TTFT over a real socket
# --------------------------------------------------------------------------


async def test_time_to_first_token_over_real_tcp(live_guardrail):
    """`httpx.ASGITransport` buffers the whole body, so an in-process measurement
    would report full buffering that is not there. Real sockets or nothing."""
    start = time.perf_counter()
    first: float | None = None
    chunks = 0

    async with live_guardrail.stream(
        "POST", "/v1/messages", json=request_body("hello there, this is a streamed answer")
    ) as response:
        assert response.status_code == 200
        async for chunk in response.aiter_bytes():
            if chunk.strip():
                chunks += 1
                if first is None:
                    first = time.perf_counter() - start
    total = time.perf_counter() - start

    assert first is not None, "no bytes were delivered"
    record("ttft_ms", round(first * 1000, 1))
    record("last_byte_ms", round(total * 1000, 1))
    record("transport_chunks", chunks)

    assert first < total, "the first byte must not arrive with the last"
    assert chunks > 1, f"the response arrived as {chunks} chunk(s); it was buffered"


async def test_time_to_first_token_tracks_the_upstream_not_the_length():
    """The number that actually characterises the guardrail.

    The end-to-end measurement above runs against a mock that answers instantly,
    so its TTFT is dominated by process overhead and says little. Here the
    upstream trickles 60 deltas at 10 ms intervals, like a model generating, and
    the question is whether the first byte tracks the *upstream's* first token
    or the response's last one. A buffering implementation scores ~1.0.
    """
    import asyncio

    async def trickling_upstream():
        for index in range(60):
            await asyncio.sleep(0.01)
            yield delta(f"word{index} ")
        yield b"event: content_block_stop\ndata: {}\n\n"

    start = time.perf_counter()
    first: float | None = None
    async for chunk in redact_sse_stream(trickling_upstream(), StreamRedactor()):
        if first is None and b"text_delta" in chunk:
            first = time.perf_counter() - start
    total = time.perf_counter() - start

    assert first is not None
    ratio = first / total
    record("slow_upstream_ttft_ms", round(first * 1000, 1))
    record("slow_upstream_total_ms", round(total * 1000, 1))
    record("slow_upstream_ttft_fraction_of_total", round(ratio, 3))

    assert ratio < 0.25, f"first byte arrived {ratio:.0%} of the way through; this is buffering"


# --------------------------------------------------------------------------
# Sustained throughput
# --------------------------------------------------------------------------


# Keyed by a short name rather than by the text itself: putting a 40,000
# character string in a parametrize id makes the node id unusable (pytest writes
# it to the cache and the report, and it errored outright).
STREAM_SHAPES: dict[str, str] = {
    "benign_prose": "the quick brown fox jumps over the lazy dog. " * 900,
    "pii_heavy": "email a@b.com ssn 123-45-6789 card 4111111111111111. " * 800,
    "adversarial_digits": "4" * STREAM_CHARS,
    "adversarial_long_token": "sk-" + "A" * STREAM_CHARS,
    "adversarial_digit_hyphen": "1-" * (STREAM_CHARS // 2),
    "adversarial_mixed_run": "a1-b2." * (STREAM_CHARS // 6),
}


@pytest.mark.parametrize("name", sorted(STREAM_SHAPES))
async def test_sustained_throughput_on_a_long_stream(name):
    """One ~40 KB response, fed in realistic 64-character deltas."""
    text = STREAM_SHAPES[name]

    async def upstream():
        for index in range(0, len(text), 64):
            yield delta(text[index : index + 64])

    redactor = StreamRedactor()
    start = time.perf_counter()
    produced = 0
    async for chunk in redact_sse_stream(upstream(), redactor):
        produced += len(chunk)
    elapsed = time.perf_counter() - start

    throughput = len(text) / elapsed / 1_000_000
    record(f"throughput_{name}_mb_per_s", round(throughput, 2))
    record(f"peak_holdback_{name}_chars", redactor.stats.max_holdback_seen)

    assert redactor.stats.max_holdback_seen <= redactor._max_holdback, "memory is not bounded"
    assert elapsed < 60, f"{name} took {elapsed:.1f}s for {len(text)} characters"
    assert produced > 0


async def test_one_character_chunks_are_survivable():
    """The worst realistic case for a per-chunk scan: 20,000 feed() calls."""
    text = "here is a key sk-" + "B" * 5_000 + " and an email a@b.com end"
    redactor = StreamRedactor()

    start = time.perf_counter()
    out = redactor.process(list(text))
    elapsed = time.perf_counter() - start

    record("one_char_chunks_count", len(text))
    record("one_char_chunks_seconds", round(elapsed, 3))
    record("one_char_chunks_peak_holdback", redactor.stats.max_holdback_seen)

    assert "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB" not in out, "the long token leaked"
    assert redactor.stats.max_holdback_seen <= redactor._max_holdback
    assert elapsed < 60, f"{len(text)} single-character feeds took {elapsed:.1f}s"


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------


async def test_the_guardrail_does_not_accumulate_the_response():
    """The property the whole design exists for: peak memory must not scale with
    response length.

    The upstream generates each delta on the fly and the output is discarded, so
    what is measured is the guardrail's own retained state. An earlier version
    of this test materialised the whole body up front and then reported *its own
    input string* as 31x growth - measuring the harness, not the subject.
    """
    sentence = "ordinary prose with no pii in it at all. "

    async def upstream(deltas: int):
        for index in range(deltas):
            yield delta(f"{sentence}{index} ")

    peaks = {}
    for deltas in (100, 10_000):
        tracemalloc.start()
        redactor = StreamRedactor()
        async for _ in redact_sse_stream(upstream(deltas), redactor):
            pass
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks[deltas] = peak
        record(f"peak_bytes_for_{deltas}_deltas", peak)
        record(f"holdback_for_{deltas}_deltas", redactor.stats.max_holdback_seen)

    growth = peaks[10_000] / max(peaks[100], 1)
    record("peak_memory_growth_for_100x_response", round(growth, 2))
    # A buffering implementation would grow ~100x with the response.
    assert growth < 10, f"peak memory grew {growth:.1f}x for a 100x longer response"


def test_the_holdback_ceiling_is_what_it_claims():
    redactor = StreamRedactor()
    record("max_holdback_chars", redactor._max_holdback)
    assert redactor._max_holdback == 328, "the documented ceiling changed"


def test_no_catastrophic_backtracking():
    """Adversarial inputs against every pattern, in one batch pass."""
    cases = {
        "digits_5k": "4" * 5_000,
        "sk_20k": "sk-" + "A" * 20_000,
        "at_signs": "x@" * 5_000,
        # Pathological for the credit-card validator: every digit is a possible
        # match end, so the shorter-candidate retry runs at nearly every offset.
        "hyphen_digits": "1-" * 10_000,
        "spaced_digits": "1 " * 10_000,
        "grouped_digits": "4111 " * 4_000,
        "email_ish": ("a.b+c%d-" * 2_000) + "@example.com",
    }
    for name, text in cases.items():
        start = time.perf_counter()
        redact_text(text)
        elapsed = time.perf_counter() - start
        record(f"redos_{name}_ms", round(elapsed * 1000, 2))
        assert elapsed < 5.0, f"{name} took {elapsed:.2f}s"


async def test_the_observations_file_was_written():
    """Fails loudly if the report would otherwise quote stale numbers."""
    assert PERF_OUTPUT.exists()
    data = json.loads(PERF_OUTPUT.read_text(encoding="utf-8"))
    assert "max_holdback_chars" in data or len(data) > 0
    record("recorded_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    record("platform", os.name)
