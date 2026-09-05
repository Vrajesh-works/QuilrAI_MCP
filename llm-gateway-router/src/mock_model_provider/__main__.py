"""Run one mock provider: `MOCK_NAME=primary MOCK_PORT=8101 python -m mock_model_provider`.

Two instances on different ports stand in for the primary and the backup.
"""

from __future__ import annotations

import os

import uvicorn

from mock_model_provider.app import create_app


def main() -> None:
    name = os.environ.get("MOCK_NAME", "primary")
    uvicorn.run(
        create_app(name),
        host=os.environ.get("MOCK_HOST", "127.0.0.1"),
        port=int(os.environ.get("MOCK_PORT", "8101" if name == "primary" else "8102")),
    )


if __name__ == "__main__":
    main()
