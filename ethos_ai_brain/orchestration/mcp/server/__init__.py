"""
MCP Server Package

Multi-headed MCP server supporting both direct and JSON-RPC protocols.
Managed by AI Engine for Agent Zero integration.
"""

from .mcp_server_base import MCPServerBase, ServerConfig
from .mcp_server_direct import DirectMCPServer
from .mcp_server_json_rpc import JsonRpcMCPServer
from .mcp_server_router import MCPServerRouter
from .mcp_server_factory import (
    create_server, 
    create_direct_server, 
    create_json_rpc_server,
    create_development_server,
    create_production_server
)

__all__ = [
    "MCPServerBase",
    "ServerConfig", 
    "DirectMCPServer",
    "JsonRpcMCPServer",
    "MCPServerRouter",
    "create_server",
    "create_direct_server", 
    "create_json_rpc_server",
    "create_development_server",
    "create_production_server"
]

__version__ = "1.0.0"
