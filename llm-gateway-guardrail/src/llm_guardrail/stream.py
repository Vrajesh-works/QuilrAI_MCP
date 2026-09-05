"""Applying the redactor to a live SSE stream from an LLM provider.

Events flow through untouched except for text deltas, whose payload is rewritten
in place. Two details decide whether this is correct:

* **A delta can emit less text than it carried** (some was held back) or more
  (a held partial resolved and flushed through). The event is re-serialised
  rather than patched, and a delta that redacts down to nothing is dropped
  instead of being sent as an empty text event.
* **The held tail must be released before the block ends.** A response finishing
  mid-partial-match would otherwise lose its last few characters, so the
  redactor is flushed at `content_block_stop` and again when the stream ends -
  whichever arrives first.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from llm_guardrail.redactor import StreamRedactor
from llm_guardrail.sse import SSEEvent, SSEParser

logger = logging.getLogger(__name__)

TEXT_DELTA_EVENT = "content_block_delta"
BLOCK_STOP_EVENT = "content_block_stop"


def _delta_text(payload: object) -> str | None:
    """The text carried by a content_block_delta, or None if it carries none.

    Thinking and input_json deltas share the event name but are not response
    text; rewriting them would corrupt the block.
    """
    if not isinstance(payload, dict):
        return None
    delta = payload.get("delta")
    if not isinstance(delta, dict) or delta.get("type") != "text_delta":
        return None
    text = delta.get("text")
    return text if isinstance(text, str) else None


def _rebuild(payload: dict, text: str) -> SSEEvent:
    updated = {**payload, "delta": {**payload["delta"], "text": text}}
    return SSEEvent(event=TEXT_DELTA_EVENT, data=json.dumps(updated, separators=(",", ":")))


def _flush_event(redactor: StreamRedactor, template: dict | None) -> SSEEvent | None:
    """Wrap whatever the redactor still holds into one final text delta."""
    remaining = redactor.flush()
    if not remaining:
        return None
    payload = template or {"type": TEXT_DELTA_EVENT, "index": 0, "delta": {"type": "text_delta", "text": ""}}
    return _rebuild(payload, remaining)


async def redact_sse_stream(
    upstream: AsyncIterator[bytes], redactor: StreamRedactor | None = None
) -> AsyncIterator[bytes]:
    """Transform an upstream SSE byte stream, redacting PII as it passes.

    Nothing is accumulated: each chunk is parsed, redacted and yielded as soon as
    it is safe to do so, so the client's TTFT is the upstream's TTFT plus a
    bounded regex scan.
    """
    redactor = redactor or StreamRedactor()
    parser = SSEParser()
    last_delta: dict | None = None

    async for chunk in upstream:
        for event in parser.feed(chunk):
            payload = event.json()
            text = _delta_text(payload)

            if text is None:
                # Not response text. Release anything held before a block ends,
                # so the tail is never stranded behind the stop event.
                if event.event == BLOCK_STOP_EVENT:
                    flushed = _flush_event(redactor, last_delta)
                    if flushed is not None:
                        yield flushed.encode()
                yield event.encode()
                continue

            last_delta = payload
            safe = redactor.feed(text)
            # An all-held or fully-redacted-to-nothing delta is dropped rather
            # than emitted empty; the text will arrive in a later event.
            if safe:
                yield _rebuild(payload, safe).encode()

    # The stream ended. Anything still parked in either buffer goes out now.
    for event in parser.flush():
        payload = event.json()
        text = _delta_text(payload)
        if text is None:
            yield event.encode()
        else:
            safe = redactor.feed(text)
            if safe:
                yield _rebuild(payload, safe).encode()

    final = _flush_event(redactor, last_delta)
    if final is not None:
        yield final.encode()

    if redactor.stats.total:
        logger.info(
            "Redacted %d item(s) from stream: %s (peak holdback %d chars)",
            redactor.stats.total,
            redactor.stats.counts,
            redactor.stats.max_holdback_seen,
        )
