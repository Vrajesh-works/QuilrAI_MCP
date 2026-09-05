"""Logging that can never contaminate the JSON-RPC wire.

Every handler this module installs targets stderr. stdout belongs to the
protocol; see `transport.py` for how that guarantee is enforced at the file
descriptor level.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def configure_logging() -> None:
    """Install a single stderr handler and route warnings through it.

    Idempotent, so importing twice cannot bolt on a second handler and double
    every line.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("CUSTOMER_MCP_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )

    root = logging.getLogger()
    # Drop anything a dependency may have installed at import time; we cannot
    # assume someone else's handler points at stderr.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    # warnings.warn() otherwise prints straight to sys.stderr unformatted; route
    # it through logging so everything shares one destination and format.
    logging.captureWarnings(True)

    _CONFIGURED = True
