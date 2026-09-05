"""STDIO isolation, proved against a real subprocess.

This is the one property that cannot be tested in-process: it is about what
lands on file descriptor 1. Each test spawns `python -m customer_mcp`, speaks
real JSON-RPC to it, and requires that *every* line on stdout parses as a
JSON-RPC message.

The strong case is `test_stdout_stays_pure_despite_a_noisy_dependency`, which
turns on CUSTOMER_MCP_DEMO_NOISE so the server deliberately prints to stdout
mid-session. Clean framing under those conditions is a real guarantee; clean
framing from well-behaved code only shows nobody happened to call print().
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "purity-probe", "version": "1.0"},
    },
}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}
LIST_TOOLS = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
VALID_CALL = {
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {"name": "get_customer_record", "arguments": {"customer_id": "CUST-00042"}},
}
INVALID_CALL = {
    "jsonrpc": "2.0",
    "id": 4,
    "method": "tools/call",
    "params": {"name": "trigger_refund", "arguments": {"customer_id": "nope", "amount": -1, "reason": "x"}},
}
DOMAIN_ERROR_CALL = {
    "jsonrpc": "2.0",
    "id": 5,
    "method": "tools/call",
    "params": {"name": "get_customer_record", "arguments": {"customer_id": "CUST-55555"}},
}


def _env(noisy: bool) -> dict[str, str]:
    env = {
        **os.environ,
        "CUSTOMER_MCP_LOG_LEVEL": "DEBUG",
        "PYTHONWARNINGS": "always",
        # Force a UTF-8 pipe so Windows' default code page cannot mangle frames.
        "PYTHONIOENCODING": "utf-8",
    }
    if noisy:
        env["CUSTOMER_MCP_DEMO_NOISE"] = "1"
    return env


MALFORMED = "{ this is not json at all }"


def run_server(
    requests: list[dict | str], *, noisy: bool = False, expect: int | None = None
) -> tuple[str, str]:
    """Drive a real server process the way a real client does.

    stdin is held open until the expected responses have been read, then closed
    to shut the server down. `subprocess.run` cannot be used here: it closes
    stdin the instant the input is written, and the server's writer task can be
    torn down on EOF with responses still queued, silently losing the last few
    frames. A real client (Claude Desktop, `claude mcp`) keeps the pipe open for
    the life of the session, so that is what this models.

    Args:
        expect: how many id-bearing responses to wait for. Defaults to the
            number of requests that carry an id.
    """
    if expect is None:
        expect = sum(1 for request in requests if isinstance(request, dict) and "id" in request)

    process = subprocess.Popen(
        [sys.executable, "-m", "customer_mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
        cwd=PROJECT_ROOT,
        env=_env(noisy),
    )

    stdout_lines: list[str] = []
    stderr_chunks: list[str] = []
    seen_responses = threading.Event()

    def drain_stdout() -> None:
        assert process.stdout is not None
        responses = 0
        for line in process.stdout:
            stdout_lines.append(line)
            # Count only well-formed responses; a corrupted line is exactly what
            # these tests exist to catch, so it must not satisfy the wait.
            try:
                if "id" in json.loads(line):
                    responses += 1
            except json.JSONDecodeError:
                pass
            if responses >= expect:
                seen_responses.set()

    def drain_stderr() -> None:
        assert process.stderr is not None
        stderr_chunks.append(process.stderr.read())

    readers = [
        threading.Thread(target=drain_stdout, daemon=True),
        threading.Thread(target=drain_stderr, daemon=True),
    ]
    for reader in readers:
        reader.start()

    try:
        assert process.stdin is not None
        for request in requests:
            # A raw string is written verbatim, so tests can send junk on purpose.
            line = request if isinstance(request, str) else json.dumps(request)
            process.stdin.write(line + "\n")
        process.stdin.flush()

        # Wait for the server to answer while its stdin is still open.
        seen_responses.wait(timeout=30)
    finally:
        with contextlib.suppress(OSError, ValueError):
            process.stdin.close()  # type: ignore[union-attr]
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover - hung server
            process.kill()
            process.wait(timeout=10)
        for reader in readers:
            reader.join(timeout=10)

    return "".join(stdout_lines), "".join(stderr_chunks)


def parse_frames(stdout: str) -> list[dict]:
    """Assert every non-empty stdout line is a JSON-RPC message, and return them.

    This is the assertion the whole design turns on: one stray `print()` on fd 1
    and a line fails to parse here.
    """
    frames = []
    for number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"stdout line {number} is not valid JSON, so it corrupted the wire: {line!r} ({exc})"
            )
        assert isinstance(message, dict), f"stdout line {number} is not a JSON-RPC object: {line!r}"
        assert message.get("jsonrpc") == "2.0", f"stdout line {number} lacks jsonrpc 2.0: {line!r}"
        assert "result" in message or "error" in message or "method" in message, (
            f"stdout line {number} is not a response or notification: {line!r}"
        )
        frames.append(message)
    return frames


ALL_REQUESTS = [INITIALIZE, INITIALIZED, LIST_TOOLS, VALID_CALL, INVALID_CALL, DOMAIN_ERROR_CALL]


def test_stdout_is_pure_jsonrpc_over_a_full_session() -> None:
    stdout, _ = run_server(ALL_REQUESTS)
    frames = parse_frames(stdout)

    responses = {frame["id"]: frame for frame in frames if "id" in frame}
    assert {1, 2, 3, 4, 5} <= set(responses), f"missing responses, got ids {sorted(responses)}"


def test_stdout_stays_pure_despite_a_noisy_dependency() -> None:
    """The real test: the server prints to stdout on purpose and must still frame cleanly."""
    stdout, stderr = run_server(ALL_REQUESTS, noisy=True)
    frames = parse_frames(stdout)

    assert len(frames) >= 5

    # The noise was genuinely emitted - otherwise this test proves nothing.
    assert "stray print()" in stderr, "demo noise never ran; the test is vacuous"
    assert "chatty dependency log line" in stderr
    assert "a library warning nobody asked for" in stderr

    # ...and none of it reached the wire.
    for marker in ("stray print()", "raw stdout write", "chatty dependency", "library warning"):
        assert marker not in stdout, f"{marker!r} leaked onto stdout"


def test_logs_go_to_stderr_and_never_to_stdout() -> None:
    _, stderr = run_server(ALL_REQUESTS)
    assert "Starting customer-mcp" in stderr, "startup log missing from stderr"
    assert "Executing tool 'get_customer_record'" in stderr


def test_invalid_input_is_an_error_frame_and_valid_input_is_a_result_frame() -> None:
    """The two channels, verified on the wire rather than through the client."""
    stdout, _ = run_server(ALL_REQUESTS)
    responses = {frame["id"]: frame for frame in parse_frames(stdout) if "id" in frame}

    # Schema violation -> a JSON-RPC error object with code -32602.
    assert "error" in responses[4], "invalid arguments should produce a JSON-RPC error"
    assert responses[4]["error"]["code"] == -32602

    # Unknown customer -> a successful response carrying isError.
    assert "result" in responses[5], "a domain refusal should be a result, not an error"
    assert responses[5]["result"]["isError"] is True

    # Happy path -> a plain result.
    assert "result" in responses[3]
    assert responses[3]["result"].get("isError", False) is False


def test_malformed_json_does_not_crash_the_server() -> None:
    """A junk line must not take the process down or desynchronise framing."""
    stdout, _ = run_server(
        [INITIALIZE, INITIALIZED, MALFORMED, LIST_TOOLS],
        # The junk line may or may not draw an error frame of its own; wait for
        # the initialize and tools/list responses that must arrive regardless.
        expect=2,
    )
    frames = parse_frames(stdout)
    ids = {frame["id"] for frame in frames if "id" in frame}
    assert 2 in ids, "server stopped serving after a malformed line"
