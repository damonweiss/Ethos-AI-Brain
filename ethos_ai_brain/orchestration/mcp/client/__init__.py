"""
MCP Client Package

Unified MCP client supporting both direct execution and JSON-RPC server modes.
Provides automatic mode detection, fallback chains, and flexible routing.
"""

from .mcp_client_router import MCPClientRouter
from .mcp_client_factory import create_client, create_direct_client, create_json_rpc_client
from .mcp_client_base import MCPConfig, RoutingConfig

__all__ = [
    "MCPClientRouter",
    "create_client", 
    "create_direct_client",
    "create_json_rpc_client",
    "MCPConfig",
    "RoutingConfig"
]

__version__ = "1.0.0"
