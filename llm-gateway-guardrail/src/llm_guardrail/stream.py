"""Applying the redactor to a live SSE stream from an LLM provider.

Fail-closed schema handling
---------------------------
Asking one question - ``is delta.type == "text_delta"?`` - and relaying
everything else byte-for-byte is the wrong failure direction for a security
control, because it makes *silence and safety the same value*. It would leak
through at least five channels: OpenAI's ``choices[].delta.content``,
Anthropic's ``thinking_delta``, ``input_json_delta`` (tool-call arguments,
which routinely carry exactly the identifiers being redacted), ``message_start``
content blocks, and any delta type invented after this file was written.

The operator's signal in that situation is silence: point ``LLM_UPSTREAM_URL``
at vLLM, Together, Groq, LiteLLM or Azure OpenAI and the proxy starts,
``/healthz`` reports ok, responses stream perfectly, and every SSN reaches the
client.

So every fragment falls into exactly one of these buckets, and there is no
default-relay path:

======================================  ==========================================
Channel                                 Policy
======================================  ==========================================
Anthropic ``text_delta``                REDACT, stateful across chunks
OpenAI ``choices[].delta.content``      REDACT, stateful across chunks
``input_json_delta.partial_json``       REDACT, stateful, its own buffer per block
``thinking_delta`` / ``signature``      DROP (configurable)
Any other JSON payload                  REDACT every string leaf, statelessly
Non-JSON data (``[DONE]``)              REDACT the raw string, statelessly
======================================  ==========================================

Thinking blocks default to DROP rather than REDACT because rewriting the text
invalidates the block's ``signature``, so a redacted thinking block is a
*corrupt* thinking block. Neither relaying nor rewriting is safe, so the block
does not go to the client at all. ``GUARDRAIL_THINKING_POLICY=redact`` is
available for deployments that prefer corrupted signatures to missing blocks.

The unknown-schema pass is stateless, and that is a real limitation rather than
an oversight: a value split across two *unrecognised* events is not caught,
because there is no way to know which string leaves form one logical stream.
It is strictly better than relaying, and the recognised channels - which is
where streamed text actually lives - are fully stateful.

Two details that decide correctness:

* **A delta can emit less text than it carried** (some was held back) or more
  (a held partial resolved and flushed through). The event is re-serialised
  rather than patched, and a delta that redacts down to nothing is dropped
  instead of being sent as an empty text event.
* **The held tail must be released before the block ends.** A response finishing
  mid-partial-match would otherwise lose its last few characters, so every live
  redactor is flushed at `content_block_stop` and again when the stream ends -
  whichever arrives first.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from llm_guardrail.redactor import StreamRedactor, redact_leaves, redact_text
from llm_guardrail.sse import SSEEvent, SSEParser

logger = logging.getLogger(__name__)

TEXT_DELTA_EVENT = "content_block_delta"
BLOCK_STOP_EVENT = "content_block_stop"

#: Key of the main response-text channel in the redactor registry. It maps to
#: the redactor the caller injected, so `stats` stays observable from outside.
MAIN_TEXT = ""

THINKING_DELTA_TYPES = frozenset({"thinking_delta", "signature_delta"})

#: Policy names, deliberately spelled out so configuration is greppable.
POLICY_REDACT = "redact"
POLICY_DROP = "drop"
POLICY_BLOCK = "block"
POLICY_PASS = "pass"

#: Written here, never taken from the exception. An upstream error message can
#: name internal hosts, deployments and account identifiers.
UPSTREAM_FAILED_MESSAGE = "The upstream model provider failed part-way through this response."


@dataclass(frozen=True)
class StreamPolicy:
    """What to do with fragments this gateway cannot redact statefully.

    Both default to the safe choice. `pass` exists so that an operator who
    genuinely needs verbatim relay has to say so, in configuration, on purpose,
    rather than getting it by accident from an unrecognised `delta.type`.
    """

    #: drop | redact | pass
    thinking: str = POLICY_DROP
    #: redact | block | pass
    unknown_schema: str = POLICY_REDACT

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> StreamPolicy:
        import os

        source = os.environ if environ is None else environ
        return cls(
            thinking=source.get("GUARDRAIL_THINKING_POLICY", POLICY_DROP).strip().lower(),
            unknown_schema=source.get("GUARDRAIL_ON_UNKNOWN_SCHEMA", POLICY_REDACT).strip().lower(),
        )


@dataclass
class _Channel:
    """One logical text stream inside the SSE stream.

    `rebuild` turns a replacement string back into a whole event, so the flush
    at the end of a block can re-emit a held tail in the shape it arrived in.
    """

    redactor: StreamRedactor
    rebuild: Callable[[str], SSEEvent] | None = None


@dataclass
class _Channels:
    """Registry of live per-channel redactors, keyed by logical stream."""

    main: StreamRedactor
    others: dict[str, _Channel] = field(default_factory=dict)
    rebuild_main: Callable[[str], SSEEvent] | None = None
    #: Set when any recognised, statefully-redacted text channel is seen. A
    #: stream that never sets this is the detector for a schema the gateway does
    #: not understand.
    saw_text: bool = False

    def get(self, key: str) -> _Channel:
        if key == MAIN_TEXT:
            return _Channel(self.main, self.rebuild_main)
        if key not in self.others:
            self.others[key] = _Channel(StreamRedactor())
        return self.others[key]

    def remember(self, key: str, rebuild: Callable[[str], SSEEvent]) -> None:
        if key == MAIN_TEXT:
            self.rebuild_main = rebuild
        else:
            self.others[key].rebuild = rebuild

    def flush(self) -> Iterator[SSEEvent]:
        """Release every held tail, in a deterministic order."""
        for key in [MAIN_TEXT, *sorted(self.others)]:
            channel = self.get(key)
            remaining = channel.redactor.flush()
            if not remaining:
                continue
            rebuild = channel.rebuild or _default_text_rebuild
            yield rebuild(remaining)

    @property
    def total_redactions(self) -> int:
        return self.main.stats.total + sum(c.redactor.stats.total for c in self.others.values())


def _default_text_rebuild(text: str) -> SSEEvent:
    payload = {"type": TEXT_DELTA_EVENT, "index": 0, "delta": {"type": "text_delta", "text": text}}
    return SSEEvent(event=TEXT_DELTA_EVENT, data=json.dumps(payload, separators=(",", ":")))


def _serialise(event_name: str | None, payload: Any) -> SSEEvent:
    return SSEEvent(event=event_name, data=json.dumps(payload, separators=(",", ":")))


def _replace_at(payload: Any, path: tuple[Any, ...], value: str) -> Any:
    """Copy `payload` with the leaf at `path` replaced. Never mutates the input."""
    if not path:
        return value
    head, rest = path[0], path[1:]
    if isinstance(head, int):
        items = list(payload)
        items[head] = _replace_at(items[head], rest, value)
        return items
    return {**payload, head: _replace_at(payload[head], rest, value)}


def _locate_texts(payload: Any) -> list[tuple[str, tuple[Any, ...], str]]:
    """Find **every** stateful text field in a delta event.

    Returns a list of ``(channel key, path to the leaf, text)``, empty if this
    payload carries no recognised streaming text. An empty list means "not a
    recognised text channel" and routes to the fail-closed path, not to verbatim
    relay.

    Every field is returned, not the first one found. `n > 1` is a standard
    OpenAI parameter, and a locator that stops at ``choices[0]`` leaves
    ``choices[1..]`` to be re-serialised verbatim - the one branch of this
    function that relays text without redacting it.

    Each channel gets its own key, and therefore its own holdback buffer:
    interleaving two logical streams through one redactor would corrupt both.
    """
    if not isinstance(payload, dict):
        return []

    found: list[tuple[str, tuple[Any, ...], str]] = []

    delta = payload.get("delta")
    if isinstance(delta, dict):
        kind = delta.get("type")
        index = payload.get("index", 0)
        if kind == "text_delta" and isinstance(delta.get("text"), str):
            # Index 0 is the main channel so the injected redactor - and the
            # stats the caller reads off it - covers the ordinary case.
            key = MAIN_TEXT if index in (0, None) else f"text:{index}"
            found.append((key, ("delta", "text"), delta["text"]))
        elif kind == "input_json_delta" and isinstance(delta.get("partial_json"), str):
            # Tool arguments, which routinely carry the identifiers being
            # redacted. A separate buffer per block, because this is a different
            # logical stream from the prose.
            found.append((f"json:{index}", ("delta", "partial_json"), delta["partial_json"]))

    # OpenAI-compatible: choices[i].delta.content. Anything that speaks this
    # shape - vLLM, Together, Groq, LiteLLM, Azure - lands here.
    choices = payload.get("choices")
    if isinstance(choices, list):
        for position, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue
            inner = choice.get("delta") or choice.get("message")
            if isinstance(inner, dict) and isinstance(inner.get("content"), str):
                field_name = "delta" if "delta" in choice else "message"
                # `not found` guards the main channel against being claimed
                # twice: a payload carrying both an Anthropic delta and an
                # OpenAI choice would otherwise share one holdback between two
                # unrelated streams.
                key = MAIN_TEXT if position == 0 and not found else f"openai:{position}"
                found.append((key, ("choices", position, field_name, "content"), inner["content"]))

    return found


def _is_thinking(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    delta = payload.get("delta")
    if isinstance(delta, dict) and delta.get("type") in THINKING_DELTA_TYPES:
        return True
    block = payload.get("content_block")
    return isinstance(block, dict) and block.get("type") == "thinking"


async def redact_sse_stream(
    upstream: AsyncIterator[bytes],
    redactor: StreamRedactor | None = None,
    policy: StreamPolicy | None = None,
) -> AsyncIterator[bytes]:
    """Transform an upstream SSE byte stream, redacting PII as it passes.

    Nothing is accumulated: each chunk is parsed, redacted and yielded as soon as
    it is safe to do so, so the client's TTFT is the upstream's TTFT plus a
    bounded regex scan.
    """
    policy = policy or StreamPolicy.from_env()
    channels = _Channels(main=redactor or StreamRedactor())
    parser = SSEParser()
    dropped = 0

    def handle(event: SSEEvent) -> Iterator[SSEEvent]:
        nonlocal dropped
        payload = event.json()

        if payload is None:
            # Not JSON at all - `data: [DONE]`, a provider sentinel, or a
            # malformed frame. Redact the raw string; a sentinel is unaffected.
            yield SSEEvent(event=event.event, data=redact_text(event.data))
            return

        if _is_thinking(payload):
            if policy.thinking == POLICY_DROP:
                dropped += 1
                return
            if policy.thinking == POLICY_REDACT:
                yield _serialise(event.event, redact_leaves(payload))
                return
            yield event  # POLICY_PASS, chosen explicitly in configuration.
            return

        located = _locate_texts(payload)
        if located:
            channels.saw_text = True
            updated = payload
            released_any = False
            for key, path, text in located:
                channel = channels.get(key)
                channels.remember(
                    key,
                    lambda safe, p=payload, q=path, e=event.event: _serialise(e, _replace_at(p, q, safe)),
                )
                safe = channel.redactor.feed(text)
                # Every located leaf is rewritten, including the ones that came
                # back empty, so no original text survives into the re-serialised
                # event. A leaf that is entirely held simply becomes "" here and
                # arrives in a later event.
                updated = _replace_at(updated, path, safe)
                released_any = released_any or bool(safe)
            # A delta that is entirely held, or redacted down to nothing, is
            # dropped rather than emitted empty.
            if released_any:
                yield _serialise(event.event, updated)
            return

        # Unrecognised shape. Release anything held before a block ends, so the
        # tail is never stranded behind the stop event.
        if event.event == BLOCK_STOP_EVENT or payload.get("type") == BLOCK_STOP_EVENT:
            yield from channels.flush()

        if policy.unknown_schema == POLICY_BLOCK:
            dropped += 1
            logger.warning("Blocked an SSE frame of unrecognised shape (event=%r)", event.event)
            return
        if policy.unknown_schema == POLICY_PASS:
            yield event
            return
        yield _serialise(event.event, redact_leaves(payload))

    try:
        async for chunk in upstream:
            for event in parser.feed(chunk):
                for out in handle(event):
                    yield out.encode()

        # The stream ended. Anything still parked in either buffer goes out now.
        for event in parser.flush():
            for out in handle(event):
                yield out.encode()
    except Exception as exc:
        # The status line went out with the first byte, so a mid-stream failure
        # cannot be signalled by status code - it has to be signalled in-band.
        # Letting the generator raise would abort the response and leave the
        # client with a truncation indistinguishable from a short completion,
        # and silent truncation read as a complete answer is worse than a
        # visible error.
        logger.warning("Upstream stream failed mid-response: %s: %s", type(exc).__name__, exc)
        # Release the held tail first - it is redacted by `flush`, so this does
        # not leak, and losing it would drop real response text on the floor.
        for out in channels.flush():
            yield out.encode()
        yield SSEEvent(
            event="error",
            data=json.dumps(
                {"type": "error", "error": {"type": "api_error", "message": UPSTREAM_FAILED_MESSAGE}},
                separators=(",", ":"),
            ),
        ).encode()
        return

    for out in channels.flush():
        yield out.encode()

    if not channels.saw_text:
        # The detector for this whole class of failure. A stream that produced
        # no redactable text field is either empty or spoke a schema this
        # gateway does not understand, and the second case is otherwise silent.
        logger.warning(
            "Stream yielded no recognised text channel; %d frame(s) went through the "
            "fail-closed path. If this is a supported provider, the guardrail is not "
            "redacting its response text and needs a locator for its schema.",
            dropped,
        )

    if channels.total_redactions:
        logger.info(
            "Redacted %d item(s) from stream: %s (peak holdback %d chars)",
            channels.total_redactions,
            channels.main.stats.counts,
            channels.main.stats.max_holdback_seen,
        )
