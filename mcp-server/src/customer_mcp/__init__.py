"""An MCP server exposing customer lookup and refund tools over stdio."""

from customer_mcp.server import SERVER_NAME, SERVER_VERSION, build_server

__all__ = ["SERVER_NAME", "SERVER_VERSION", "build_server"]
