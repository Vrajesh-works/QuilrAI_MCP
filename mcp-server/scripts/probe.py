"""Drive the server over real stdio and print the transcript.

For a human reviewer who wants to see the framing without wiring up a client:

    uv run python scripts/probe.py

Requests go out on the subprocess's stdin, responses come back on its stdout,
and the server's own logs land on stderr - shown at the end so you can confirm
they never touched the wire.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from queue import Empty, Queue

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCENARIOS: list[tuple[str, dict]] = [
    (
        "Handshake",
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "probe", "version": "1.0"},
            },
        },
    ),
    ("Discover the tool surface", {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    (
        "Valid lookup",
        {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "get_customer_record", "arguments": {"customer_id": "CUST-00042"}},
        },
    ),
    (
        "Malformed customer id -> JSON-RPC error -32602",
        {
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "get_customer_record", "arguments": {"customer_id": "cust-42"}},
        },
    ),
    (
        "Negative amount and a too-short reason -> -32602 naming both fields",
        {
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {
                "name": "trigger_refund",
                "arguments": {"customer_id": "CUST-00042", "amount": -3, "reason": "nope"},
            },
        },
    ),
    (
        "Unknown tool -> JSON-RPC error -32601",
        {
            "jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "params": {"name": "admin_reset_key", "arguments": {}},
        },
    ),
    (
        "Valid refund -> result",
        {
            "jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {
                "name": "trigger_refund",
                "arguments": {
                    "customer_id": "CUST-00042",
                    "amount": 120.00,
                    "reason": "Duplicate charge on the April invoice.",
                },
            },
        },
    ),
    (
        "Refund above the remaining balance -> isError result, NOT a protocol error",
        {
            "jsonrpc": "2.0", "id": 8, "method": "tools/call",
            "params": {
                "name": "trigger_refund",
                "arguments": {
                    "customer_id": "CUST-00042",
                    "amount": 5_000.00,
                    "reason": "Trying to refund more than is available.",
                },
            },
        },
    ),
    (
        "Frozen account -> isError result",
        {
            "jsonrpc": "2.0", "id": 9, "method": "tools/call",
            "params": {
                "name": "trigger_refund",
                "arguments": {
                    "customer_id": "CUST-01337",
                    "amount": 10.00,
                    "reason": "Refund against a frozen account.",
                },
            },
        },
    ),
]


def describe(response: dict) -> str:
    if "error" in response:
        error = response["error"]
        return f"JSON-RPC ERROR {error['code']}: {error['message']}"
    result = response.get("result", {})
    if result.get("isError"):
        return "result with isError=true (a business refusal, delivered to the model)"
    return "result"


def main() -> int:
    process = subprocess.Popen(
        [sys.executable, "-m", "customer_mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
        cwd=PROJECT_ROOT,
    )
    assert process.stdin and process.stdout and process.stderr

    responses: Queue[str] = Queue()
    stderr_lines: list[str] = []
    threading.Thread(
        target=lambda: [responses.put(line) for line in process.stdout], daemon=True
    ).start()
    threading.Thread(
        target=lambda: stderr_lines.append(process.stderr.read()), daemon=True
    ).start()

    process.stdin.write(json.dumps(SCENARIOS[0][1]) + "\n")
    process.stdin.flush()
    handshake = responses.get(timeout=15)
    process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
    process.stdin.flush()

    print("=" * 78)
    print(f"  {SCENARIOS[0][0]}")
    print("=" * 78)
    print(f"<-- {handshake.strip()[:200]}...\n")

    clean = True
    for label, request in SCENARIOS[1:]:
        print("=" * 78)
        print(f"  {label}")
        print("=" * 78)
        print(f"--> {json.dumps(request)}")
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        try:
            raw = responses.get(timeout=15)
        except Empty:
            print("    (no response)\n")
            clean = False
            continue
        try:
            response = json.loads(raw)
        except json.JSONDecodeError:
            print(f"!!! stdout carried non-JSON, the wire is corrupt: {raw!r}")
            clean = False
            continue
        print(f"<-- [{describe(response)}]")
        print(json.dumps(response, indent=2))
        print()

    process.stdin.close()
    process.wait(timeout=15)

    print("=" * 78)
    print("  Server stderr (logs, kept entirely off the JSON-RPC wire)")
    print("=" * 78)
    print("".join(stderr_lines).strip())
    print()
    print("stdout was pure JSON-RPC." if clean else "stdout was NOT clean - see above.")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
