"""Run the mock downstream server: `python -m mock_downstream`."""

from __future__ import annotations

import os

import uvicorn

from mock_downstream.app import create_app


def main() -> None:
    uvicorn.run(
        create_app(),
        host=os.environ.get("MOCK_HOST", "127.0.0.1"),
        port=int(os.environ.get("MOCK_PORT", "8081")),
    )


if __name__ == "__main__":
    main()
