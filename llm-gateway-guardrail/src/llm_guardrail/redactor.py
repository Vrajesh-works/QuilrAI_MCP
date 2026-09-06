"""Incremental PII redaction over a text stream.

The difficulty
--------------
A stream delivers text in arbitrary pieces. A model emitting an SSN might send
``"...my ssn is 123-4"`` and then ``"5-6789..."``. Running a regex over each
chunk independently finds nothing in either, and the PII goes out intact:
matches straddle chunk boundaries.

Buffering the whole response and redacting it once is correct but defeats the
point of streaming: time-to-first-token becomes time-to-*last*-token, and memory
grows with response length.

The approach
------------
Keep a small **holdback**: the shortest suffix of pending text that could still
turn into a match, found with a genuine partial-match engine
(``regex``'s ``partial=True``). Everything before it can never be part of a
future match, so it is safe to emit immediately.

For ordinary prose the holdback is empty and text flows through untouched, so
TTFT is the upstream's TTFT plus a regex scan. Only when the tail starts to look
like PII does anything get held, and only until the next chunk resolves it.

    feed("Your ssn is 123-4")  -> emits "Your ssn is ", holds "123-4"
    feed("5-6789 ok")          -> emits "[REDACTED] ok", holds ""

Memory is bounded by `max_holdback` regardless of how long the stream runs.

Over-long candidates
--------------------
The holdback ceiling and the patterns disagree about length: a `sk-` key or an
email local part can be arbitrarily long, the ceiling is not. Two rules resolve
that, and both are needed:

* **Scan the whole buffer**, never a trailing window. A candidate longer than a
  window loses its own beginning and stops being recognisable - `sk-AAAA...`
  with the `sk-` scrolled out matches no pattern - so windowing would release
  the run one character at a time. The buffer is small, so a full scan is cheap.
* **Fail closed when the candidate outgrows the ceiling.** The buffer may
  legitimately exceed `max_holdback`, and at that point the redactor does not
  know what it is holding. It emits one `[REDACTED]` and enters a suppression
  state that drops the rest of the run, rather than releasing a prefix it
  cannot vouch for. Uncertainty must not turn into disclosure.

Bounding the regex quantifiers to `max_length` is not an alternative to either
rule: a bounded `{1,64}` local part means the surviving partial starts only 64
characters from the end, so *more* of the run is released, not less.

The deliberate trade, stated so it is not discovered in production: an unbroken
run of more than `max_holdback` token characters - a long base64 blob, a hash,
a JWT - is replaced by `[REDACTED]` while streaming, where the non-streaming
path would leave it alone. `max_holdback` is constructor-configurable for
deployments that need a different balance.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from llm_guardrail.patterns import (
    DEFAULT_PATTERNS,
    OVERLONG_RUN,
    REDACTION_PLACEHOLDER,
    Pattern,
)

logger = logging.getLogger(__name__)

#: Stats key for a candidate that outgrew the ceiling and was failed closed.
OVERFLOW_COUNTER = "overflow"


@dataclass
class RedactionStats:
    """What the redactor did, for logging and for the tests to assert on."""

    counts: dict[str, int] = field(default_factory=dict)
    characters_in: int = 0
    characters_out: int = 0
    max_holdback_seen: int = 0
    #: Characters dropped as the tail of an over-long suppressed run. A non-zero
    #: value means the fail-closed path was taken, and is what a production
    #: deployment should alert on.
    suppressed_characters: int = 0

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def record(self, name: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1


class StreamRedactor:
    """Redacts PII from a text stream without buffering the whole thing.

    Not thread-safe and not reusable across streams: one instance per response.
    """

    def __init__(
        self,
        patterns: Sequence[Pattern] = DEFAULT_PATTERNS,
        placeholder: str = REDACTION_PLACEHOLDER,
        max_holdback: int | None = None,
    ):
        self._patterns = tuple(patterns)
        self._placeholder = placeholder
        # A hard ceiling on retained text. Without it, a stream of digits looks
        # like a forever-growing partial credit card and the buffer grows without
        # bound - the exact failure the design is meant to avoid. Sized from the
        # longest pattern so it can never split a legitimate match.
        self._max_holdback = max_holdback or max(pattern.max_length for pattern in self._patterns) + 8
        self._buffer = ""
        # True while dropping the tail of a run that outgrew the ceiling and was
        # already replaced by a placeholder. Without it, only the held prefix is
        # protected and the remainder of the same secret streams out behind it.
        self._suppressing = False
        self.stats = RedactionStats()

    @property
    def pending(self) -> str:
        """Text currently held back. Exposed so tests can assert it stays bounded."""
        return self._buffer

    def _redact(self, text: str, *, protect_tail: bool) -> str:
        """Replace matches, honouring each pattern's validator.

        Args:
            protect_tail: skip a match that ends at the very end of `text`,
                because more input could still extend it. Set while streaming,
                cleared on flush when no more input can arrive.

        Patterns are applied in order and the result re-scanned by the next, so
        overlapping detectors settle deterministically.
        """
        for pattern in self._patterns:
            pieces: list[str] = []
            position = 0
            search_from = 0
            while search_from <= len(text):
                match = pattern.expression.search(text, search_from)
                if match is None:
                    break
                end = self._validated_end(pattern, text, match.start(), match.end())
                if end is None or (protect_tail and end == len(text)):
                    # Either the validator rejected every candidate at this
                    # position, or the match reaches the end of the buffer and
                    # could still grow. Resume one character along rather than
                    # skipping the whole span, so a rejected long candidate
                    # cannot hide a real match that starts inside it.
                    search_from = match.start() + 1
                    continue
                pieces.append(text[position : match.start()])
                pieces.append(self._placeholder)
                self.stats.record(pattern.name)
                position = end
                search_from = end if end > match.start() else match.start() + 1
            pieces.append(text[position:])
            text = "".join(pieces)
        return text

    @staticmethod
    def _validated_end(pattern: Pattern, text: str, start: int, end: int) -> int | None:
        """Longest candidate starting at `start` that the validator accepts.

        A greedy quantifier and a validator can disagree. In
        ``4111111111111111 123-45-6789`` the card pattern greedily swallows the
        separator and the first three SSN digits, giving a nineteen-digit
        candidate that fails Luhn. Discarding it and moving on would emit a
        perfectly valid sixteen-digit card in the clear next to a redacted SSN,
        so the end is backed off and re-anchored until the real card is found.

        The step is not "one character at a time". `match` with an `endpos`
        returns the *greedy* match within that limit, which is already the next
        plausible end - so each iteration jumps straight to the next candidate
        instead of testing positions the pattern could never end on. On a
        pathological run like ``"1-" * 10000`` (where every digit is a possible
        match end, so this loop runs at nearly every offset) that difference,
        together with the `validate_min_length` floor, is the difference between
        a measurable denial-of-service vector and a rounding error.
        """
        if pattern.validate is None:
            return end if pattern.accepts(text[start:end]) else None

        # `>=`, not `>`: a candidate of exactly `validate_min_length` characters
        # is the shortest the validator can accept, so it must still be tried.
        floor = start + max(pattern.validate_min_length, 1)
        while end >= floor:
            candidate = pattern.expression.match(text, start, end)
            if candidate is None:
                return None
            if pattern.accepts(candidate.group()):
                return candidate.end()
            # The greedy match ended here and was rejected, so the next
            # candidate must be strictly shorter.
            end = candidate.end() - 1
        return None

    def _holdback_index(self, text: str) -> int:
        """Index of the earliest point from which a match might still change.

        Two kinds of match are unsafe, and missing either one is a bug:

        * **Partial** - the pattern ran out of input mid-match, e.g. ``123-45``
          of an SSN. This is the case the design starts from.
        * **Complete, but ending at the very end of the buffer** - the less
          obvious one. ``ada.lovelace@example.co`` is a complete, valid email
          match; redact it the moment it appears and the ``m`` that arrives next
          chunk lands after the placeholder, giving ``[REDACTED]m``. A match
          touching the end of the buffer can still grow, so it is held until
          something follows it.

        Returns len(text) when nothing is pending, meaning all of it is safe.

        The **whole** buffer is scanned, not a trailing window. Windowing would
        look like a free optimisation - anything unsafe runs to the end of the
        buffer, so surely a match cannot start further back than one maximum
        pattern length - but that argument assumes the patterns respect
        `max_length`, and the unbounded ones do not. A candidate longer than the
        window loses its own beginning, stops being recognisable, and is
        released. This scan is bounded instead by the buffer, which `feed` keeps
        small.
        """
        earliest = len(text)
        for pattern in self._patterns:
            for match in pattern.expression.finditer(text, partial=True):
                if match.partial or match.end() == len(text):
                    # A zero-width partial at the very end (a lone `\b`) holds
                    # nothing back, which is what keeps prose flowing freely.
                    earliest = min(earliest, match.start())
        return earliest

    def _drop_suppressed_prefix(self, text: str) -> str:
        """Discard the leading part of `text` that continues a suppressed run.

        `_fail_closed` has already emitted a placeholder for the head of an
        over-long candidate. Everything still arriving from the same unbroken
        run belongs to that placeholder and must not be emitted. The run is
        known to consist only of `OVERLONG_RUN` characters (see the reasoning in
        `patterns.py`), so the first character outside that class ends it.
        """
        match = OVERLONG_RUN.match(text)
        consumed = match.end() if match else 0
        self.stats.suppressed_characters += consumed
        if consumed == len(text):
            return ""  # Still inside the run; the whole chunk is dropped.
        self._suppressing = False
        return text[consumed:]

    def _fail_closed(self) -> str:
        """Give up on an over-long candidate safely rather than release it.

        Reached when the held candidate has grown past `max_holdback`. The
        redactor cannot tell whether it is holding a credential or a base64
        blob, and the whole point of the component is that it must not guess in
        the direction of disclosure.
        """
        held = len(self._buffer)
        self.stats.record(OVERFLOW_COUNTER)
        self.stats.suppressed_characters += held
        self._buffer = ""
        self._suppressing = True
        logger.warning(
            "Candidate exceeded the %d-character holdback ceiling; emitted a placeholder "
            "and suppressing the remainder of the run (%d characters held). This is the "
            "fail-closed path for an over-long token.",
            self._max_holdback,
            held,
        )
        return self._placeholder

    def feed(self, text: str) -> str:
        """Absorb the next piece of the stream; return text safe to emit now.

        Redaction happens *before* the split, and this order is load-bearing.
        Splitting first lets one pattern's partial match land inside another
        pattern's completed match - a partial credit card starting in the middle
        of `(555) 123-4567` - and the split then cuts the phone number in half,
        so neither half matches and the whole thing is emitted in the clear.
        Redacting settled matches first removes the text those spurious partials
        were feeding on.
        """
        if not text:
            return ""

        self.stats.characters_in += len(text)

        if self._suppressing:
            text = self._drop_suppressed_prefix(text)
            if not text:
                return ""

        self._buffer = self._redact(self._buffer + text, protect_tail=True)

        split = self._holdback_index(self._buffer)
        emit, self._buffer = self._buffer[:split], self._buffer[split:]

        # Enforce the ceiling. Exceeding it means one candidate has stayed
        # unresolved for longer than any bounded pattern could be, so it is an
        # over-long email or API key run. It is *not* safe to release the oldest
        # held text - that text is the head of the candidate. Fail closed.
        if len(self._buffer) > self._max_holdback:
            emit += self._fail_closed()

        self.stats.max_holdback_seen = max(self.stats.max_holdback_seen, len(self._buffer))
        self.stats.characters_out += len(emit)
        return emit

    def flush(self) -> str:
        """End of stream: redact and release whatever is still held.

        Must be called, or a response ending mid-partial-match loses its tail.
        At this point no more input can arrive, so a partial match is just text.
        """
        remaining = self._redact(self._buffer, protect_tail=False)
        self._buffer = ""
        self._suppressing = False
        self.stats.characters_out += len(remaining)
        return remaining

    def process(self, chunks: Iterable[str]) -> str:
        """Convenience for tests: run a whole stream and return the output."""
        return "".join([*(self.feed(chunk) for chunk in chunks), self.flush()])


def redact_text(text: str, patterns: Sequence[Pattern] = DEFAULT_PATTERNS) -> str:
    """Non-streaming redaction, used as the oracle the stream must agree with."""
    return StreamRedactor(patterns).process([text])


def redact_leaves(value: Any) -> Any:
    """Statelessly redact every string in an arbitrary JSON structure.

    The fail-closed default for a payload whose shape is not known, used by both
    the SSE path and the non-streaming path. Rebuilding from the parsed
    structure rather than relaying the original bytes is deliberate: it
    guarantees no string leaf can slip past because the parser and the relay
    disagreed about what was in there.

    Statelessly, because there is no way to know which leaves of an
    unrecognised structure form one logical stream. A value split across two
    unrecognised events is therefore not caught; recognised channels get a
    stateful redactor instead.
    """
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_leaves(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_leaves(item) for key, item in value.items()}
    return value
