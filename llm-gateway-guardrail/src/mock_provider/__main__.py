"""Run the mock provider: `python -m mock_provider`."""

from __future__ import annotations

import os

import uvicorn

from mock_provider.app import create_app


def main() -> None:
    uvicorn.run(create_app(), host=os.environ.get("MOCK_HOST", "127.0.0.1"), port=int(os.environ.get("MOCK_PORT", "8091")))


if __name__ == "__main__":
    main()
