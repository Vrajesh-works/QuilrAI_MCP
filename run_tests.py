"""Run every suite in the workspace and report a combined total.

    uv run python run_tests.py            # all four
    uv run python run_tests.py mcp-server # just one

Why a script rather than plain `uv run pytest`: all four projects have a
`tests/conftest.py`, and each puts its own `tests/` directory on `sys.path` so
its helpers can be imported by name. In a single pytest process those four
`conftest` modules collide - whichever directory lands on the path first wins,
and the other suites fail to import. Running each project in its own process
keeps the isolation the per-project layout assumes, while still being one
command.

Pass any extra pytest arguments after the project list; they are forwarded:

    uv run python run_tests.py -- -k redactor -v
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

PROJECTS = [
    ("mcp-server", "MCP server - stdio transport, strict validation, stdout purity"),
    ("mcp-gateway", "MCP security gateway - bearer auth, tool authorization"),
    ("llm-gateway-guardrail", "LLM gateway - streaming PII redaction"),
    ("llm-gateway-router", "LLM gateway - rate limiting and model failover"),
]


def parse_total(output: str) -> tuple[int, int]:
    """Pull (passed, failed) out of pytest's summary line."""
    passed = failed = 0
    for line in reversed(output.strip().splitlines()):
        if " passed" in line or " failed" in line or " error" in line:
            for part in line.replace("=", " ").split():
                if part.isdigit():
                    count = int(part)
                elif part.startswith("passed"):
                    passed = count
                elif part.startswith(("failed", "error")):
                    failed += count
            if passed or failed:
                return passed, failed
    return 0, 0


def main() -> int:
    arguments = sys.argv[1:]
    if "--" in arguments:
        split = arguments.index("--")
        selected, pytest_arguments = arguments[:split], arguments[split + 1 :]
    else:
        selected, pytest_arguments = arguments, []

    projects = [item for item in PROJECTS if not selected or item[0] in selected]
    if not projects:
        print(f"No such project. Known: {', '.join(name for name, _ in PROJECTS)}")
        return 2

    results: list[tuple[str, int, int, float]] = []
    started = time.perf_counter()

    for name, description in projects:
        print(f"\n{'=' * 78}\n  {description}\n{'=' * 78}")
        begin = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *pytest_arguments],
            cwd=ROOT / name,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - begin

        output = completed.stdout + completed.stderr
        passed, failed = parse_total(output)
        # Show everything on failure; just the tail when green.
        print(output if failed or completed.returncode != 0 else output.strip().splitlines()[-1])
        results.append((name, passed, failed, elapsed))

    total_passed = sum(passed for _, passed, _, _ in results)
    total_failed = sum(failed for _, _, failed, _ in results)

    print(f"\n{'=' * 78}\n  Summary\n{'=' * 78}")
    for name, passed, failed, elapsed in results:
        status = "FAIL" if failed else "ok"
        print(f"  {name:<24} {passed:>4} passed  {failed:>3} failed  {elapsed:>6.1f}s  {status}")
    print(f"  {'TOTAL':<24} {total_passed:>4} passed  {total_failed:>3} failed  "
          f"{time.perf_counter() - started:>6.1f}s")

    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
