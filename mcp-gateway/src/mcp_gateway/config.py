"""Gateway configuration, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    downstream_url: str
    request_timeout_seconds: float
    filter_tools_list: bool

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            downstream_url=os.environ.get("MCP_DOWNSTREAM_URL", "http://127.0.0.1:8081/mcp"),
            request_timeout_seconds=float(os.environ.get("MCP_GATEWAY_TIMEOUT", "30")),
            # Off by default: a client needs the full tool list to plan, and an
            # agent that cannot see a tool cannot explain why it failed. Enabling
            # this trades that away to avoid advertising the privileged surface.
            filter_tools_list=os.environ.get("MCP_GATEWAY_FILTER_TOOLS_LIST", "").lower()
            in {"1", "true", "yes"},
        )
