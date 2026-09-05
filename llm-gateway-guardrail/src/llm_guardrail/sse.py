"""Incremental Server-Sent Events parsing and serialisation.

SSE has the same chunk-boundary problem as the redactor, one layer down: the
transport hands over arbitrary byte runs, and a single event - even a single
`data:` line - can be split across several of them. The parser is therefore fed
bytes and yields only whole events.

Only the parts of the SSE grammar (WHATWG HTML §9.2) that a streaming LLM API
actually uses are implemented: `event`, `data`, `id`, `retry`, comments, and
multi-line data joined with newlines.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SSEEvent:
    event: str | None
    data: str

    def json(self) -> Any | None:
        """Parsed data, or None if it is not JSON.

        A stream ends with the literal `data: [DONE]` on some providers, which
        must not blow up a JSON-oriented proxy.
        """
        try:
            return json.loads(self.data)
        except (json.JSONDecodeError, ValueError):
            return None

    def encode(self) -> bytes:
        lines = []
        if self.event:
            lines.append(f"event: {self.event}")
        # Each line of a multi-line payload needs its own `data:` prefix, or the
        # receiver reassembles it wrongly.
        for line in self.data.split("\n"):
            lines.append(f"data: {line}")
        return ("\n".join(lines) + "\n\n").encode("utf-8")


class SSEParser:
    """Feed it bytes, get whole events out.

    Holds at most one incomplete event, so memory does not grow with stream
    length.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._event: str | None = None
        self._data: list[str] = []

    def feed(self, chunk: bytes | str) -> Iterator[SSEEvent]:
        text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk
        self._buffer += text

        # Normalise the three line terminators SSE permits before splitting, so
        # a \r\n landing across two chunks cannot be read as an empty line and
        # dispatch an event early.
        while True:
            index = self._find_line_end()
            if index is None:
                break
            line, self._buffer = self._buffer[: index[0]], self._buffer[index[1] :]
            event = self._handle_line(line)
            if event is not None:
                yield event

    def _find_line_end(self) -> tuple[int, int] | None:
        """(end of line, start of next line), or None if no complete line yet."""
        position = self._buffer.find("\r\n")
        if position != -1:
            return position, position + 2
        for terminator in ("\n", "\r"):
            position = self._buffer.find(terminator)
            if position != -1:
                # A trailing lone \r might be the first half of a \r\n still in
                # flight; wait for more input rather than guess.
                if terminator == "\r" and position == len(self._buffer) - 1:
                    return None
                return position, position + 1
        return None

    def _handle_line(self, line: str) -> SSEEvent | None:
        if line == "":
            return self._dispatch()
        if line.startswith(":"):
            return None  # comment / keep-alive
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value

        if field == "event":
            self._event = value
        elif field == "data":
            self._data.append(value)
        # `id` and `retry` are consumed and not surfaced; nothing here needs them.
        return None

    def _dispatch(self) -> SSEEvent | None:
        if not self._data and self._event is None:
            return None
        event = SSEEvent(event=self._event, data="\n".join(self._data))
        self._event = None
        self._data = []
        return event

    def flush(self) -> Iterator[SSEEvent]:
        """Emit a final event if the stream ended without a trailing blank line."""
        if self._buffer:
            line, self._buffer = self._buffer, ""
            self._handle_line(line)
        event = self._dispatch()
        if event is not None:
            yield event
