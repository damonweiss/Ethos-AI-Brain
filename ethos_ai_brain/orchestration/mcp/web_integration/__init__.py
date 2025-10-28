"""
MCP Web Integration Package

Flask and web framework integration for MCP servers.
Provides REST API endpoints for web frontend access to MCP tools.
"""

from .flask_blueprint import mcp_blueprint, get_registered_tools, is_tool_registered, get_tool_info, clear_tool_registry

__all__ = [
    "mcp_blueprint",
    "get_registered_tools", 
    "is_tool_registered",
    "get_tool_info",
    "clear_tool_registry"
]

__version__ = "1.0.0"
