"""Every malformed request shape gets an answer, over a real stdio subprocess.

MCP-1: seven classes of malformed-but-identified request received no response at
all. The process survived and kept serving - which is why a survivability test
passed and this was graded correct - but a client that sent one waited out its
own timeout with nothing. Survivability and conformance are different
properties.

These drive `python -m customer_mcp` as an actual subprocess rather than the
in-memory session, because the defect lives in the transport and the in-memory
fixture does not use it.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time

import pytest
from customer_mcp.transport import classify_line

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 0,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "conformance-probe", "version": "1.0"},
    },
}

PROBE_ID = 911


class Server:
    """A real `python -m customer_mcp` subprocess, spoken to over pipes.

    stdout is drained on a background thread. That is not incidental: a plain
    `readline()` blocks forever when the server sends nothing, so a harness
    built on it *hangs on exactly the bug being tested* instead of reporting it.
    The queue makes "no response" an observable, timed result.
    """

    def __init__(self) -> None:
        environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self.process = subprocess.Popen(
            [sys.executable, "-m", "customer_mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=environment,
        )
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def _drain(self) -> None:
        for line in self.process.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def send_raw(self, line: str) -> None:
        self.process.stdin.write(line if line.endswith("\n") else line + "\n")
        self.process.stdin.flush()

    def send(self, payload) -> None:
        self.send_raw(json.dumps(payload))

    def read_frames(self, count: int, timeout: float = 10.0) -> list[dict]:
        """Read up to `count` JSON frames, or fewer if the server goes quiet."""
        frames: list[dict] = []
        deadline = time.monotonic() + timeout
        while len(frames) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                break
            if line is None:  # stdout closed
                break
            if line.strip():
                frames.append(json.loads(line))
        return frames

    def handshake(self) -> None:
        self.send(INITIALIZE)
        assert self.read_frames(1), "server never answered initialize"
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def close(self) -> None:
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - only on a hang
            self.process.kill()


@pytest.fixture
def server():
    instance = Server()
    try:
        instance.handshake()
        yield instance
    finally:
        instance.close()


# The exact seven shapes the audit found, plus the ones next to them.
MALFORMED_REQUESTS = [
    pytest.param({"jsonrpc": "2.0", "id": PROBE_ID}, id="missing method"),
    pytest.param({"jsonrpc": "2.0", "id": PROBE_ID, "method": "tools/list", "params": []}, id="array params"),
    pytest.param({"jsonrpc": "2.0", "id": PROBE_ID, "method": "tools/list", "params": "x"}, id="string params"),
    pytest.param({"jsonrpc": "2.0", "id": PROBE_ID, "method": "tools/list", "params": 42}, id="numeric params"),
    pytest.param({"jsonrpc": "1.0", "id": PROBE_ID, "method": "tools/list"}, id="jsonrpc 1.0"),
    pytest.param({"id": PROBE_ID, "method": "tools/list"}, id="no jsonrpc key"),
    pytest.param({"jsonrpc": 2.0, "id": PROBE_ID, "method": "tools/list"}, id="numeric jsonrpc"),
    pytest.param({"jsonrpc": "2.0", "id": PROBE_ID, "method": 42}, id="numeric method"),
    pytest.param({"jsonrpc": "2.0", "id": PROBE_ID, "method": ""}, id="empty method"),
    pytest.param({"jsonrpc": "2.0", "id": PROBE_ID, "method": None}, id="null method"),
]


@pytest.mark.parametrize("payload", MALFORMED_REQUESTS)
def test_a_malformed_request_carrying_an_id_is_answered(server, payload):
    """The core requirement: no identified request may hang the client."""
    server.send(payload)
    frames = server.read_frames(1, timeout=8.0)

    assert frames, f"NO RESPONSE - a client sending {payload} would hang"
    frame = frames[0]
    assert frame.get("id") == PROBE_ID, f"the error must be correlatable: {frame}"
    assert "error" in frame, frame
    assert frame["error"]["code"] in (-32600, -32602), frame


UNCORRELATABLE = [
    pytest.param({"jsonrpc": "2.0", "id": True, "method": "tools/list"}, id="bool id"),
    pytest.param({"jsonrpc": "2.0", "id": 1.5, "method": "tools/list"}, id="float id"),
    pytest.param({"jsonrpc": "2.0", "id": {"a": 1}, "method": "tools/list"}, id="object id"),
    pytest.param({"jsonrpc": "2.0", "id": [1], "method": "tools/list"}, id="array id"),
]


@pytest.mark.parametrize("payload", UNCORRELATABLE)
def test_an_unusable_id_still_draws_an_error_with_a_null_id(server, payload):
    """`id: null` is what §5.1 reserves for "the id could not be determined".
    It is less useful than an echo, and infinitely better than silence."""
    server.send(payload)
    frames = server.read_frames(1, timeout=8.0)

    assert frames, f"NO RESPONSE for {payload}"
    assert frames[0]["id"] is None
    assert frames[0]["error"]["code"] == -32600


RAW_LINES = [
    pytest.param('{ garbage not json', -32700, id="malformed json"),
    pytest.param('"just a string"', -32600, id="json string"),
    pytest.param("42", -32600, id="json number"),
    pytest.param("null", -32600, id="json null"),
    pytest.param("true", -32600, id="json bool"),
    pytest.param('[{"jsonrpc":"2.0","id":911,"method":"tools/list"}]', -32600, id="batch of one"),
    pytest.param('[]', -32600, id="empty batch"),
]


@pytest.mark.parametrize(("line", "code"), RAW_LINES)
def test_unparseable_lines_are_answered_rather_than_swallowed(server, line, code):
    server.send_raw(line)
    frames = server.read_frames(1, timeout=8.0)

    assert frames, f"NO RESPONSE for raw line {line!r}"
    assert frames[0]["error"]["code"] == code, frames[0]


def test_the_server_keeps_serving_after_a_burst_of_malformed_input(server):
    """Answering malformed input must not cost survivability."""
    for payload in MALFORMED_REQUESTS:
        server.send(payload.values[0])
    server.send_raw("{ totally broken")
    server.send_raw("")
    server.send_raw("   ")

    server.send({"jsonrpc": "2.0", "id": 5000, "method": "tools/list"})
    frames = server.read_frames(len(MALFORMED_REQUESTS) + 2, timeout=15.0)
    ids = [frame.get("id") for frame in frames]
    assert 5000 in ids, f"the server stopped serving after malformed input; saw {ids}"

    control = next(frame for frame in frames if frame.get("id") == 5000)
    assert "result" in control
    assert {tool["name"] for tool in control["result"]["tools"]} == {
        "get_customer_record",
        "trigger_refund",
    }


def test_a_malformed_notification_draws_no_response(server):
    """JSON-RPC §4.1 is unconditional. Inventing a frame for a notification
    would desynchronise a client that is not expecting one."""
    server.send({"jsonrpc": "2.0", "method": 42})
    server.send({"jsonrpc": "1.0", "method": "notifications/whatever"})
    server.send({"jsonrpc": "2.0", "id": 777, "method": "tools/list"})

    frames = server.read_frames(2, timeout=8.0)
    assert [frame.get("id") for frame in frames] == [777], frames


def test_valid_requests_are_completely_unaffected(server):
    """The validating layer must not change the meaning of a good request."""
    server.send({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "get_customer_record", "arguments": {"customer_id": "CUST-00042"}},
    })
    frames = server.read_frames(1, timeout=8.0)
    assert frames and "result" in frames[0], frames
    assert "Ada Lovelace" in json.dumps(frames[0])


def test_deep_validation_still_produces_32602(server):
    """The envelope check passes the request on; the schema still rejects it."""
    server.send({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "get_customer_record", "arguments": {"customer_id": "nope"}},
    })
    frames = server.read_frames(1, timeout=8.0)
    assert frames[0]["error"]["code"] == -32602, frames


def test_an_unknown_method_still_produces_32601(server):
    server.send({"jsonrpc": "2.0", "id": 3, "method": "no/such/method"})
    frames = server.read_frames(1, timeout=8.0)
    assert frames[0]["error"]["code"] == -32601, frames


def test_stdout_stays_pure_while_answering_malformed_input():
    """The fix must not have traded the assessment's stdout guarantee away."""
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "CUSTOMER_MCP_DEMO_NOISE": "1"}
    process = subprocess.Popen(
        [sys.executable, "-m", "customer_mcp"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", env=environment,
    )
    lines = [json.dumps(INITIALIZE), '{ garbage', json.dumps({"jsonrpc": "2.0", "id": 9, "method": "tools/list"})]
    stdout, stderr = process.communicate("\n".join(lines) + "\n", timeout=30)

    emitted = [line for line in stdout.splitlines() if line.strip()]
    assert emitted, "no output at all"
    for line in emitted:
        json.loads(line)  # every stdout line must be a JSON-RPC frame
    assert "stray print()" in stderr, "the noise should have gone to stderr"
    assert "stray print()" not in stdout


# --------------------------------------------------------------------------
# Unit-level coverage of the classifier, for the branches a subprocess makes
# awkward to reach.
# --------------------------------------------------------------------------


def test_classifier_passes_valid_requests_through():
    assert classify_line('{"jsonrpc":"2.0","id":1,"method":"tools/list"}') is True
    assert classify_line('{"jsonrpc":"2.0","method":"notifications/initialized"}') is True
    assert classify_line('{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}') is True
    assert classify_line('{"jsonrpc":"2.0","id":1,"method":"tools/list","params":null}') is True


def test_classifier_passes_client_responses_through():
    """A response to a server-initiated request is not a request."""
    assert classify_line('{"jsonrpc":"2.0","id":1,"result":{}}') is True
    assert classify_line('{"jsonrpc":"2.0","id":1,"error":{"code":-1,"message":"x"}}') is True


def test_classifier_drops_blank_lines():
    assert classify_line("") is None
    assert classify_line("\n") is None
    assert classify_line("   \t \n") is None


def test_classifier_drops_malformed_notifications_silently():
    assert classify_line('{"jsonrpc":"1.0","method":"x"}') is None
    assert classify_line('{"jsonrpc":"2.0","method":42}') is None


def test_classifier_echoes_a_string_id():
    verdict = classify_line('{"jsonrpc":"2.0","id":"abc","method":42}')
    assert verdict.message.id == "abc"
    assert verdict.message.error.code == -32600
