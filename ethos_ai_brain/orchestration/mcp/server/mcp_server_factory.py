#!/usr/bin/env python3
"""
MCP Server Factory

Factory functions for creating pre-configured MCP server routers.
Mirrors the client factory pattern for consistency.
"""

import logging
from pathlib import Path
from typing import Optional

from .mcp_server_base import ServerConfig
from .mcp_server_router import MCPServerRouter

logger = logging.getLogger(__name__)


def create_server(protocol: str = "direct",
                 host: str = "localhost", 
                 port: int = 8051,
                 tools_dir: Optional[Path] = None,
                 tool_registry=None) -> MCPServerRouter:
    """Create a server router with specified protocol."""
    config = ServerConfig(
        host=host,
        port=port, 
        protocol=protocol,
        tools_directory=tools_dir
    )
    return MCPServerRouter(config, tool_registry)


def create_direct_server(host: str = "localhost",
                        port: int = 8051,
                        tools_dir: Optional[Path] = None,
                        tool_registry=None) -> MCPServerRouter:
    """Create a direct HTTP server router."""
    config = ServerConfig(
        host=host,
        port=port,
        protocol="direct", 
        tools_directory=tools_dir,
        enable_cors=True
    )
    return MCPServerRouter(config, tool_registry)


def create_json_rpc_server(host: str = "localhost",
                          port: int = 8052,
                          tools_dir: Optional[Path] = None,
                          tool_registry=None) -> MCPServerRouter:
    """Create a JSON-RPC server router."""
    config = ServerConfig(
        host=host,
        port=port,
        protocol="json_rpc",
        tools_directory=tools_dir,
        enable_cors=False
    )
    return MCPServerRouter(config, tool_registry)


def create_development_server(port: int = 8051,
                             tools_dir: Optional[Path] = None,
                             tool_registry=None) -> MCPServerRouter:
    """Create a development server (direct protocol, CORS enabled)."""
    config = ServerConfig(
        host="localhost",
        port=port,
        protocol="direct",
        tools_directory=tools_dir,
        enable_cors=True
    )
    return MCPServerRouter(config, tool_registry)


def create_production_server(port: int = 8052,
                            tools_dir: Optional[Path] = None,
                            tool_registry=None) -> MCPServerRouter:
    """Create a production server (JSON-RPC, external access)."""
    config = ServerConfig(
        host="0.0.0.0",  # Accept external connections
        port=port,
        protocol="json_rpc",
        tools_directory=tools_dir,
        enable_cors=False,  # More restrictive
        max_connections=1000
    )
    return MCPServerRouter(config, tool_registry)
