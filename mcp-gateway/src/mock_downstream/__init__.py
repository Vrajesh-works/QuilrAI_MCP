"""A deliberately trusting MCP server, used as the gateway's downstream."""

from mock_downstream.app import create_app, received, reset_received

__all__ = ["create_app", "received", "reset_received"]
