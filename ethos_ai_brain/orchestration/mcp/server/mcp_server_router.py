#!/usr/bin/env python3
"""
MCP Server Router

Single server router that matches client architecture pattern.
Routes to appropriate server implementation based on configuration.
"""

import logging
from typing import Any, Dict, Optional

from .mcp_server_base import MCPServerBase, ServerConfig
from .mcp_server_direct import DirectMCPServer
from .mcp_server_json_rpc import JsonRpcMCPServer

logger = logging.getLogger(__name__)


class MCPServerRouter:
    """Router that manages a single MCP server implementation (matches client pattern)."""
    
    def __init__(self, config: ServerConfig, tool_registry=None):
        self.config = config
        self.tool_registry = tool_registry
        self._server = None
        self.is_running = False
        
        # Initialize the appropriate server based on protocol
        self._initialize_server()
    
    def _initialize_server(self):
        """Initialize server based on configuration."""
        if self.config.protocol == "direct":
            self._server = DirectMCPServer(self.config, self.tool_registry)
        elif self.config.protocol == "json_rpc":
            self._server = JsonRpcMCPServer(self.config, self.tool_registry)
        else:
            raise ValueError(f"Unsupported protocol: {self.config.protocol}")
        
        logger.info(f"Initialized {self.config.protocol} server router")
    
    @property
    def protocol_name(self) -> str:
        """Get the protocol name."""
        return self._server.protocol_name if self._server else "unknown"
    
    async def start(self) -> None:
        """Start the routed server."""
        if self.is_running:
            logger.warning("Server router already running")
            return
        
        try:
            await self._server.start()
            self.is_running = True
            logger.info(f"Server router started ({self.config.protocol})")
        except Exception as e:
            logger.error(f"Failed to start server router: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the routed server."""
        if not self.is_running:
            return
        
        try:
            await self._server.stop()
            self.is_running = False
            logger.info("Server router stopped")
        except Exception as e:
            logger.error(f"Error stopping server router: {e}")
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route request to underlying server."""
        if not self._server:
            return {"error": "No server available", "success": False}
        
        return await self._server.handle_request(request)
    
    def health_check(self) -> Dict[str, Any]:
        """Check router and underlying server health."""
        if not self._server:
            return {
                "status": "error",
                "error": "No server available"
            }
        
        return self._server.health_check()
    
    async def list_tools(self) -> Dict[str, Any]:
        """List tools via routed server."""
        if not self._server:
            return {"error": "No server available", "tools": []}
        
        return await self._server.list_tools()
    
    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool via routed server."""
        if not self._server:
            return {"error": "No server available", "success": False}
        
        return await self._server.execute_tool(tool_name, params)
    
    async def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get tool info via routed server."""
        if not self._server:
            return {"error": "No server available"}
        
        return await self._server.get_tool_info(tool_name)
    
    def __str__(self) -> str:
        return f"MCPServerRouter({self.config.protocol}:{self.config.host}:{self.config.port})"
