"""Run the gateway: `python -m mcp_gateway`."""

from __future__ import annotations

import os

import uvicorn

from mcp_gateway.app import create_app
from mcp_gateway.logging_setup import configure_logging


def main() -> None:
    configure_logging()
    uvicorn.run(
        create_app(),
        host=os.environ.get("MCP_GATEWAY_HOST", "127.0.0.1"),
        port=int(os.environ.get("MCP_GATEWAY_PORT", "8080")),
        log_config=None,
    )


if __name__ == "__main__":
    main()
