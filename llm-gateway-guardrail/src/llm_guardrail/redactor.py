"""Incremental PII redaction over a text stream.

The problem
-----------
A stream delivers text in arbitrary pieces. A model emitting an SSN might send
``"...my ssn is 123-4"`` and then ``"5-6789..."``. Running a regex over each
chunk independently finds nothing in either, and the PII goes out intact. This
is the whole difficulty here: matches straddle chunk boundaries.

The rejected fix
----------------
Buffer the entire response, redact once, then send. Correct, and it destroys the
point of streaming: time-to-first-token becomes time-to-*last*-token, and memory
grows with response length.

The approach here
-----------------
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
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from llm_guardrail.patterns import DEFAULT_PATTERNS, REDACTION_PLACEHOLDER, Pattern


@dataclass
class RedactionStats:
    """What the redactor did, for logging and for the tests to assert on."""

    counts: dict[str, int] = field(default_factory=dict)
    characters_in: int = 0
    characters_out: int = 0
    max_holdback_seen: int = 0

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
            for match in pattern.expression.finditer(text):
                if protect_tail and match.end() == len(text):
                    continue
                if not pattern.accepts(match.group()):
                    # Validator says this is not really PII (a non-Luhn order
                    # number, say). Leave the original text alone.
                    continue
                pieces.append(text[position : match.start()])
                pieces.append(self._placeholder)
                self.stats.record(pattern.name)
                position = match.end()
            pieces.append(text[position:])
            text = "".join(pieces)
        return text

    def _holdback_index(self, text: str) -> int:
        """Index of the earliest point from which a match might still change.

        Two kinds of match are unsafe, and missing either one is a bug:

        * **Partial** - the pattern ran out of input mid-match, e.g. ``123-45``
          of an SSN. Obvious, and the case the design starts from.
        * **Complete, but ending at the very end of the buffer** - far less
          obvious and the one that actually bit. ``ada.lovelace@example.co``
          is a complete, valid email match; redact it the moment it appears and
          the ``m`` that arrives next chunk lands after the placeholder, giving
          ``[REDACTED]m``. A match touching the end of the buffer can still
          grow, so it is held until something follows it.

        Returns len(text) when nothing is pending, meaning all of it is safe.
        Only the tail window is scanned - anything unsafe runs to the end of the
        buffer, so a match starting earlier than one maximum pattern length back
        cannot be one.
        """
        window_start = max(0, len(text) - self._max_holdback)
        window = text[window_start:]

        earliest = len(text)
        for pattern in self._patterns:
            for match in pattern.expression.finditer(window, partial=True):
                if match.partial or match.end() == len(window):
                    # A zero-width partial at the very end (a lone `\b`) holds
                    # nothing back, which is what keeps prose flowing freely.
                    earliest = min(earliest, window_start + match.start())
        return earliest

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
        self._buffer = self._redact(self._buffer + text, protect_tail=True)

        split = self._holdback_index(self._buffer)
        emit, self._buffer = self._buffer[:split], self._buffer[split:]

        # Enforce the ceiling. Exceeding it means the tail has looked like an
        # unfinished match for longer than any real match could be, so the oldest
        # held text cannot belong to a genuine one and is safe to release.
        if len(self._buffer) > self._max_holdback:
            overflow = len(self._buffer) - self._max_holdback
            emit += self._redact(self._buffer[:overflow], protect_tail=False)
            self._buffer = self._buffer[overflow:]

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
        self.stats.characters_out += len(remaining)
        return remaining

    def process(self, chunks: Iterable[str]) -> str:
        """Convenience for tests: run a whole stream and return the output."""
        return "".join([*(self.feed(chunk) for chunk in chunks), self.flush()])


def redact_text(text: str, patterns: Sequence[Pattern] = DEFAULT_PATTERNS) -> str:
    """Non-streaming redaction, used as the oracle the stream must agree with."""
    return StreamRedactor(patterns).process([text])
