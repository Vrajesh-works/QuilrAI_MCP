"""Token accounting.

Estimation before the call, reconciliation after it.
"""

from __future__ import annotations

import json
from typing import Any

# Characters per token. A rough average for English; the point of this constant
# is that it is *deliberately* rough and corrected on settle, not that it is
# accurate. A production gateway calls the provider's token-counting endpoint
# (or runs the tokenizer locally) instead - the seam is `estimate_input_tokens`.
CHARS_PER_TOKEN = 4

DEFAULT_MAX_TOKENS = 1_024

# Upper bound on the `max_tokens` component of a reservation. Well above any
# real completion and well below the point where the arithmetic stops meaning
# anything.
MAX_RESERVABLE_TOKENS = 1_000_000

# Charged when a request is admitted but the provider reports no usage, so a
# stream of such requests still consumes quota rather than being free.
MINIMUM_CHARGE = 1


def estimate_input_tokens(body: dict[str, Any]) -> int:
    """Estimate the prompt cost of a Messages-shaped request body."""
    characters = 0

    system = body.get("system")
    if isinstance(system, str):
        characters += len(system)
    elif isinstance(system, list):
        characters += len(json.dumps(system))

    for message in body.get("messages", []) or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            characters += len(content)
        elif content is not None:
            # Structured blocks: images, tool results. Serialise for a rough size.
            characters += len(json.dumps(content))

    # Tool definitions are part of the prompt and are easy to forget; they are
    # frequently the largest component of a real agent request.
    if body.get("tools"):
        characters += len(json.dumps(body["tools"]))

    return max(1, characters // CHARS_PER_TOKEN)


def reservation_size(body: dict[str, Any]) -> int:
    """Tokens to hold for a request before it runs.

    The prompt estimate plus `max_tokens`, because that is the most the request
    could possibly cost. Reserving only the prompt would let a tenant sit just
    under the limit and then generate an unbounded completion past it.
    """
    max_tokens = body.get("max_tokens")
    # `isinstance(True, int)` is True in Python, so `{"max_tokens": true}` used
    # to satisfy this check and reserve **1** token instead of the 1,024
    # default - a systematic under-reservation available to anyone who noticed.
    # The MCP server guards against exactly this bool/int confusion in
    # `schemas.py`; the router did not.
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        max_tokens = DEFAULT_MAX_TOKENS
    # Clamped upward too. An absurd `max_tokens` makes every request
    # self-reject, which is only self-inflicted - but a reservation larger than
    # the whole window is never a useful thing to compute.
    max_tokens = min(max_tokens, MAX_RESERVABLE_TOKENS)
    return estimate_input_tokens(body) + max_tokens


def actual_tokens(response_body: Any) -> int | None:
    """Real token spend from a provider response, or None if it did not say."""
    if not isinstance(response_body, dict):
        return None
    usage = response_body.get("usage")
    if not isinstance(usage, dict):
        return None

    total = 0
    found = False
    for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and value >= 0:
            total += value
            found = True

    if not found:
        return None
    return max(MINIMUM_CHARGE, total)


class StreamUsageCollector:
    """Accumulates token usage from an SSE stream as its bytes go past.

    Streaming responses do not carry a JSON body, so `actual_tokens` found
    nothing and every streamed request settled at its estimate. Anthropic
    reports input tokens on `message_start` and output tokens on the terminal
    `message_delta`, so both have to be picked up as they fly by - the stream is
    relayed to the client, never buffered, so there is no complete document to
    inspect afterwards.

    Deliberately forgiving: this is billing telemetry riding along on a relay,
    and a parse failure must never break the client's stream. Anything it
    cannot understand simply leaves the total where it was, and the router
    falls back to charging the estimate.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self.input_tokens = 0
        self.output_tokens = 0
        self.saw_usage = False

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk.decode("utf-8", errors="replace")
        # Keep only the trailing partial line; a `data:` line can straddle any
        # number of transport chunks.
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            self._consume(line.strip())

    def _consume(self, line: str) -> None:
        if not line.startswith("data:"):
            return
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            return
        try:
            event = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return
        if not isinstance(event, dict):
            return

        usage = event.get("usage")
        if not isinstance(usage, dict) and isinstance(event.get("message"), dict):
            usage = event["message"].get("usage")
        if not isinstance(usage, dict):
            return

        for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                self.input_tokens += value
                self.saw_usage = True
        output = usage.get("output_tokens")
        if isinstance(output, int) and not isinstance(output, bool) and output >= 0:
            # `message_delta` reports a running total, not an increment, so the
            # last value wins rather than being summed.
            self.output_tokens = output
            self.saw_usage = True

    @property
    def total(self) -> int | None:
        if not self.saw_usage:
            return None
        return max(MINIMUM_CHARGE, self.input_tokens + self.output_tokens)
