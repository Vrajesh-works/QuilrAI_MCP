"""An MCP security gateway: Bearer auth, tool-level authorization, JSON-RPC proxy."""

from mcp_gateway.app import create_app
from mcp_gateway.config import Config

__all__ = ["Config", "create_app"]
