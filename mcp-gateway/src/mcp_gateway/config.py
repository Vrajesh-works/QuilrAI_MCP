"""Gateway configuration, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    downstream_url: str
    request_timeout_seconds: float
    filter_tools_list: bool
    # A JSON-RPC control message is small. This is a memory-exhaustion guard,
    # not a business limit: without it `await request.body()` buffers whatever
    # the client sends, and with no concurrency cap either that is the cheapest
    # way to take the process down.
    max_body_bytes: int = 1_048_576
    # Forward only known MCP methods. See policy.KNOWN_METHODS for why this
    # defaults to on and how to extend it.
    enforce_method_allowlist: bool = True

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            downstream_url=os.environ.get("MCP_DOWNSTREAM_URL", "http://127.0.0.1:8081/mcp"),
            request_timeout_seconds=float(os.environ.get("MCP_GATEWAY_TIMEOUT", "30")),
            max_body_bytes=int(os.environ.get("MCP_GATEWAY_MAX_BODY_BYTES", "1048576")),
            enforce_method_allowlist=os.environ.get("MCP_GATEWAY_METHOD_ALLOWLIST", "on").lower()
            not in {"0", "off", "false", "no"},
            # Off by default: a client needs the full tool list to plan, and an
            # agent that cannot see a tool cannot explain why it failed. Enabling
            # this trades that away to avoid advertising the privileged surface.
            filter_tools_list=os.environ.get("MCP_GATEWAY_FILTER_TOOLS_LIST", "").lower()
            in {"1", "true", "yes"},
        )
