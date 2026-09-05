"""Run the console: `python -m demo_console`."""

from __future__ import annotations

import os

import uvicorn

from demo_console.app import create_app


def main() -> None:
    port = int(os.environ.get("CONSOLE_PORT", "8000"))
    print(f"Console on http://127.0.0.1:{port}")
    uvicorn.run(create_app(), host=os.environ.get("CONSOLE_HOST", "127.0.0.1"), port=port, log_level="warning")


if __name__ == "__main__":
    main()
