"""Logging for the gateway process."""

from __future__ import annotations

import logging
import os
import sys


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, os.environ.get("MCP_GATEWAY_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )
