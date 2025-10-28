#!/usr/bin/env python3
"""
MCP Client JSON-RPC Strategy

JSON-RPC strategy using HTTP/JSON-RPC protocol for MCP server communication.
"""

import logging
import requests
from typing import Any, Dict, List

from .mcp_client_base import MCPStrategy

logger = logging.getLogger(__name__)


class JsonRpcStrategy(MCPStrategy):
    """Strategy for execution via MCP server using JSON-RPC over HTTP."""
    
    def __init__(self, server_url: str = "http://localhost:8052"):
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
    
    @property
    def strategy_name(self) -> str:
        return "json_rpc"
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool via MCP server using HTTP requests."""
        try:
            # Make HTTP request to server
            url = f"{self.server_url}/tools/{tool_name}/execute"
            response = self.session.post(url, json=params, timeout=30)
            
            if response.ok:
                result = response.json()
            else:
                result = {
                    'success': False,
                    'error': f'HTTP {response.status_code}: {response.reason}',
                    'strategy': self.strategy_name
                }
            
            # Ensure result has strategy info
            if isinstance(result, dict):
                result["strategy"] = self.strategy_name
            return result
        except Exception as e:
            logger.error(f"JSON-RPC execution failed for {tool_name}: {e}")
            return {
                "success": False,
                "error": f"JSON-RPC execution failed: {str(e)}",
                "strategy": self.strategy_name,
                "tool_name": tool_name
            }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List tools via MCP server."""
        try:
            url = f"{self.server_url}/tools"
            response = self.session.get(url, timeout=30)
            
            if response.ok:
                data = response.json()
                tools = data.get("tools", [])
                # Add strategy info to each tool
                for tool in tools:
                    tool["available_via"] = tool.get("available_via", []) + [self.strategy_name]
                return tools
            else:
                logger.error(f"JSON-RPC tool listing failed: HTTP {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"JSON-RPC tool listing failed: {e}")
            return []
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get tool info via MCP server."""
        try:
            url = f"{self.server_url}/tools/{tool_name}"
            response = self.session.get(url, timeout=30)
            
            if response.ok:
                result = response.json()
                result["strategy"] = self.strategy_name
                return result
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.reason}",
                    "strategy": self.strategy_name
                }
        except Exception as e:
            logger.error(f"Failed to get tool info for {tool_name}: {e}")
            return {
                "success": False,
                "error": f"Failed to get tool info: {str(e)}",
                "strategy": self.strategy_name
            }
    
    def is_available(self) -> bool:
        """Check if MCP server is available."""
        try:
            url = f"{self.server_url}/health"
            response = self.session.get(url, timeout=5)
            return response.ok
        except Exception as e:
            logger.debug(f"JSON-RPC strategy not available: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Health check via MCP server."""
        try:
            url = f"{self.server_url}/health"
            response = self.session.get(url, timeout=10)
            
            if response.ok:
                result = response.json()
                result["strategy"] = self.strategy_name
                return result
            else:
                return {
                    "status": "unhealthy",
                    "strategy": self.strategy_name,
                    "error": f"HTTP {response.status_code}: {response.reason}"
                }
        except Exception as e:
            logger.error(f"JSON-RPC strategy health check failed: {e}")
            return {
                "status": "unhealthy",
                "strategy": self.strategy_name,
                "error": str(e)
            }
    
    def refresh_tools(self) -> Dict[str, Any]:
        """Refresh tools via MCP server."""
        try:
            url = f"{self.server_url}/refresh"
            response = self.session.post(url, timeout=30)
            
            if response.ok:
                result = response.json()
                result["strategy"] = self.strategy_name
                return result
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.reason}",
                    "strategy": self.strategy_name
                }
        except Exception as e:
            logger.error(f"JSON-RPC tool refresh failed: {e}")
            return {
                "success": False,
                "error": f"Tool refresh failed: {str(e)}",
                "strategy": self.strategy_name
            }
