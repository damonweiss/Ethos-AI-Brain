#!/usr/bin/env python3
"""
MCP Server Base Classes

Abstract base classes and configuration for multi-headed MCP servers.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for MCP servers."""
    host: str = "localhost"
    port: int = 8051
    protocol: str = "direct"  # "direct" or "json_rpc"
    tools_directory: Optional[Path] = None
    enable_cors: bool = True
    max_connections: int = 100
    timeout: int = 30


class MCPServerBase(ABC):
    """Abstract base class for MCP servers."""
    
    def __init__(self, config: ServerConfig, tool_registry=None):
        self.config = config
        self.tool_registry = tool_registry
        self.is_running = False
        self._server = None
        
        # Set up logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @property
    @abstractmethod
    def protocol_name(self) -> str:
        """Name of the protocol this server implements."""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the MCP server."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the MCP server."""
        pass
    
    @abstractmethod
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming request."""
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        return {
            "status": "healthy" if self.is_running else "stopped",
            "protocol": self.protocol_name,
            "host": self.config.host,
            "port": self.config.port,
            "tools_available": len(self.tool_registry.discover_tools()) if self.tool_registry else 0
        }
    
    async def list_tools(self) -> Dict[str, Any]:
        """List available tools."""
        if not self.tool_registry:
            return {"error": "No tool registry available", "tools": []}
        
        try:
            tools = self.tool_registry.discover_tools()
            
            # Filter out non-serializable fields (like function objects)
            serializable_tools = []
            for tool in tools:
                serializable_tool = {}
                for key, value in tool.items():
                    # Skip function objects and other non-serializable types
                    if key == 'function' or callable(value):
                        continue
                    # Convert Path objects to strings
                    elif hasattr(value, '__fspath__'):  # Path-like objects
                        serializable_tool[key] = str(value)
                    else:
                        try:
                            # Test if the value is JSON serializable
                            import json
                            json.dumps(value)
                            serializable_tool[key] = value
                        except (TypeError, ValueError):
                            # Skip non-serializable values
                            continue
                
                serializable_tools.append(serializable_tool)
            
            return {
                "tools": serializable_tools,
                "count": len(serializable_tools),
                "protocol": self.protocol_name
            }
        except Exception as e:
            self.logger.error(f"Failed to list tools: {e}")
            return {"error": str(e), "tools": []}
    
    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool."""
        if not self.tool_registry:
            return {"error": "No tool registry available", "success": False}
        
        try:
            result = self.tool_registry.execute_tool(tool_name, **params)
            return {
                "success": True,
                "result": result,
                "tool_name": tool_name,
                "protocol": self.protocol_name
            }
        except Exception as e:
            self.logger.error(f"Failed to execute tool {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name,
                "protocol": self.protocol_name
            }
    
    async def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get information about a specific tool."""
        if not self.tool_registry:
            return {"error": "No tool registry available"}
        
        try:
            tools = self.tool_registry.discover_tools()
            tool = next((t for t in tools if t.get("name") == tool_name), None)
            
            if tool:
                return {
                    "success": True,
                    "tool_info": tool,  # tool is already a dict from discover_tools
                    "protocol": self.protocol_name
                }
            else:
                return {
                    "success": False,
                    "error": f"Tool not found: {tool_name}",
                    "protocol": self.protocol_name
                }
        except Exception as e:
            self.logger.error(f"Failed to get tool info for {tool_name}: {e}")
            return {"error": str(e), "success": False}
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.config.host}:{self.config.port})"
