"""SSE parsing, which has the chunk-boundary problem one layer below the redactor."""

from __future__ import annotations

import pytest

from llm_guardrail.sse import SSEEvent, SSEParser

RAW = (
    b"event: message_start\ndata: {\"type\":\"message_start\"}\n\n"
    b"event: content_block_delta\ndata: {\"delta\":{\"text\":\"hi\"}}\n\n"
    b"event: message_stop\ndata: {}\n\n"
)


def drain(parser: SSEParser, chunks) -> list[SSEEvent]:
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.flush())
    return events


def test_whole_stream_in_one_chunk():
    events = drain(SSEParser(), [RAW])
    assert [event.event for event in events] == ["message_start", "content_block_delta", "message_stop"]


@pytest.mark.parametrize("size", [1, 2, 3, 7, 13, 64, 4096])
def test_events_survive_every_byte_chunking(size):
    """An event - even a single data: line - can be split anywhere."""
    chunks = [RAW[index : index + size] for index in range(0, len(RAW), size)]
    events = drain(SSEParser(), chunks)

    assert [event.event for event in events] == ["message_start", "content_block_delta", "message_stop"]
    assert events[1].json()["delta"]["text"] == "hi"


@pytest.mark.parametrize("terminator", [b"\n", b"\r\n", b"\r"])
def test_all_three_line_terminators(terminator):
    raw = b"event: ping" + terminator + b"data: {}" + terminator + terminator
    events = drain(SSEParser(), [raw])
    assert [event.event for event in events] == ["ping"]


def test_crlf_split_across_chunks_does_not_dispatch_early():
    """A lone trailing \r might be the first half of a \r\n still in flight.

    Treating it as a line end would dispatch the event early and then read the
    \n as an empty line, emitting a spurious second event.
    """
    events = drain(SSEParser(), [b"event: ping\r", b"\ndata: {}\r\n\r\n"])
    assert [event.event for event in events] == ["ping"]


def test_comments_and_keepalives_are_ignored():
    events = drain(SSEParser(), [b": keep-alive\n\nevent: ping\ndata: {}\n\n"])
    assert [event.event for event in events] == ["ping"]


def test_multiline_data_is_joined_with_newlines():
    events = drain(SSEParser(), [b"data: line one\ndata: line two\n\n"])
    assert events[0].data == "line one\nline two"


def test_non_json_data_does_not_raise():
    """Some providers end a stream with a literal `data: [DONE]`."""
    events = drain(SSEParser(), [b"data: [DONE]\n\n"])
    assert events[0].json() is None


def test_optional_space_after_colon_is_stripped_once():
    events = drain(SSEParser(), [b"data:  two spaces\n\n"])
    assert events[0].data == " two spaces"


def test_stream_ending_without_trailing_blank_line_is_flushed():
    events = drain(SSEParser(), [b"event: ping\ndata: {}"])
    assert [event.event for event in events] == ["ping"]


def test_encode_round_trips():
    event = SSEEvent(event="content_block_delta", data='{"a":1}')
    assert drain(SSEParser(), [event.encode()])[0] == event


def test_encode_prefixes_every_line_of_multiline_data():
    encoded = SSEEvent(event="x", data="one\ntwo").encode()
    assert encoded == b"event: x\ndata: one\ndata: two\n\n"
