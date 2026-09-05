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
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        max_tokens = DEFAULT_MAX_TOKENS
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
