"""Entrypoint: `python -m customer_mcp`, speaking MCP over stdio.

On STDIO isolation
------------------
stdout must carry nothing but JSON-RPC. Two things guarantee that here:

1. `configure_logging()` runs before anything else, so every log record and
   captured warning this process emits is bound to stderr from the first line.

2. `stdio_server()` in mcp >= 2.x claims the standard descriptors itself: while
   serving, fd 1 is pointed at stderr and the wire is served from a private
   duplicate. A stray `print()` - ours, a dependency's, or a subprocess's -
   lands on stderr and cannot corrupt a frame.

Because the SDK performs that redirection, this module must *not* dup2 fd 1
itself. Doing so before `stdio_server()` starts would leave the transport
serving JSON-RPC onto stderr, which is exactly the failure the redirect is
meant to prevent. The test suite pins the behaviour.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

import anyio

from customer_mcp.logging_setup import configure_logging

# Before any other import-time side effect gets a chance to log.
configure_logging()

from customer_mcp.server import SERVER_NAME, SERVER_VERSION, build_server  # noqa: E402

logger = logging.getLogger(__name__)


def _emit_demo_noise() -> None:
    """Deliberately misbehave, to demonstrate that the wire survives it.

    Enabled with CUSTOMER_MCP_DEMO_NOISE=1. This is what a careless dependency
    does; the stdout-purity test turns it on and still requires clean framing.
    """
    print("stray print() straight to stdout - this must not reach the wire")
    sys.stdout.write("another raw stdout write\n")
    sys.stdout.flush()
    warnings.warn("a library warning nobody asked for", stacklevel=1)
    logging.getLogger("noisy.dependency").warning("chatty dependency log line")


async def _serve() -> None:
    from customer_mcp.transport import validating_stdio_server

    server = build_server()
    logger.info("Starting %s v%s on stdio", SERVER_NAME, SERVER_VERSION)

    # `validating_stdio_server` wraps the SDK's `stdio_server`, answering the
    # malformed request shapes the SDK silently discards. It leaves the fd 1
    # claim - and therefore the stdout-purity guarantee documented above -
    # entirely to the SDK. See `transport.py`.
    async with validating_stdio_server() as (read_stream, write_stream):
        if os.environ.get("CUSTOMER_MCP_DEMO_NOISE") == "1":
            # Inside the context, where the SDK's descriptor claim is active.
            _emit_demo_noise()
        await server.run(read_stream, write_stream, server.create_initialization_options())

    logger.info("Server stopped")


def main() -> None:
    try:
        anyio.run(_serve)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        logger.info("Interrupted; shutting down")


if __name__ == "__main__":
    main()
