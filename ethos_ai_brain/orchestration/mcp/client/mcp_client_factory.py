#!/usr/bin/env python3
"""
MCP Client Factory

Factory functions for creating pre-configured MCP client routers.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from .mcp_client_base import MCPConfig, RoutingConfig
from .mcp_client_router import MCPClientRouter

logger = logging.getLogger(__name__)


def create_client(mode: str = "auto",
                 tools_dir: Optional[Path] = None,
                 server_url: str = "http://localhost:8051") -> MCPClientRouter:
    """
    Create an MCP client router with automatic configuration.
    
    Args:
        mode: Execution mode ("auto", "direct", "json_rpc", "hybrid")
        tools_dir: Directory containing MCP tools (for direct mode)
        server_url: URL of MCP server (for JSON-RPC mode)
        
    Returns:
        Configured MCPClientRouter instance
    """
    config = MCPConfig(
        server_url=server_url,
        tools_directory=tools_dir,
        default_mode=mode
    )
    
    router = MCPClientRouter(config)
    logger.info(f"Created MCP client router with mode: {mode}")
    return router


def create_direct_client(tools_dir: Optional[Path] = None) -> MCPClientRouter:
    """
    Create an MCP client router configured for direct execution only.
    
    Args:
        tools_dir: Directory containing MCP tools
        
    Returns:
        MCPClientRouter configured for direct execution
    """
    config = MCPConfig(
        tools_directory=tools_dir,
        default_mode="direct"
    )
    
    router = MCPClientRouter(config)
    logger.info("Created direct-only MCP client router")
    return router


def create_json_rpc_client(server_url: str = "http://localhost:8051") -> MCPClientRouter:
    """
    Create an MCP client router configured for JSON-RPC server communication only.
    
    Args:
        server_url: URL of the MCP server
        
    Returns:
        MCPClientRouter configured for JSON-RPC communication
    """
    config = MCPConfig(
        server_url=server_url,
        default_mode="json_rpc"
    )
    
    router = MCPClientRouter(config)
    logger.info(f"Created JSON-RPC-only MCP client router for {server_url}")
    return router


def create_hybrid_client(tools_dir: Optional[Path] = None,
                        server_url: str = "http://localhost:8051",
                        routing_rules: Optional[Dict[str, str]] = None,
                        namespace_rules: Optional[Dict[str, str]] = None) -> MCPClientRouter:
    """
    Create an MCP client router with hybrid mode and custom routing rules.
    
    Args:
        tools_dir: Directory containing MCP tools
        server_url: URL of MCP server
        routing_rules: Tool-specific routing rules (tool_name -> strategy)
        namespace_rules: Namespace-based routing rules (namespace -> strategy)
        
    Returns:
        MCPClientRouter configured for hybrid routing
    """
    # Default namespace rules for common patterns
    default_namespace_rules = {
        "brain.": "direct",      # Brain tools via direct execution
        "actions.": "json_rpc",  # Action tools via server
        "web.": "json_rpc",      # Web tools via server (better isolation)
        "api.": "json_rpc"       # API tools via server
    }
    
    # Merge with provided rules
    final_namespace_rules = default_namespace_rules.copy()
    if namespace_rules:
        final_namespace_rules.update(namespace_rules)
    
    routing_config = RoutingConfig(
        routing_rules=routing_rules or {},
        namespace_rules=final_namespace_rules
    )
    
    config = MCPConfig(
        server_url=server_url,
        tools_directory=tools_dir,
        default_mode="hybrid",
        routing_config=routing_config
    )
    
    router = MCPClientRouter(config)
    logger.info("Created hybrid MCP client router with custom routing rules")
    return router


def create_development_client(tools_dir: Optional[Path] = None) -> MCPClientRouter:
    """
    Create an MCP client router optimized for development.
    
    Uses direct execution for faster iteration and debugging.
    
    Args:
        tools_dir: Directory containing MCP tools
        
    Returns:
        MCPClientRouter optimized for development
    """
    return create_direct_client(tools_dir)


def create_production_client(server_url: str = "http://localhost:8051",
                           fallback_tools_dir: Optional[Path] = None) -> MCPClientRouter:
    """
    Create an MCP client router optimized for production.
    
    Prefers JSON-RPC server with direct execution fallback.
    
    Args:
        server_url: URL of production MCP server
        fallback_tools_dir: Fallback tools directory if server unavailable
        
    Returns:
        MCPClientRouter optimized for production
    """
    # Production routing: prefer server, fallback to direct
    routing_config = RoutingConfig(
        namespace_rules={
            "": "json_rpc"  # Default all tools to server
        }
    )
    
    config = MCPConfig(
        server_url=server_url,
        tools_directory=fallback_tools_dir,
        default_mode="auto",  # Auto-detect with server preference
        routing_config=routing_config
    )
    
    router = MCPClientRouter(config)
    logger.info(f"Created production MCP client router for {server_url}")
    return router


# Convenience aliases
create_auto_client = create_client
create_dev_client = create_development_client
create_prod_client = create_production_client
