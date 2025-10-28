#!/usr/bin/env python3
"""
MCP Client Direct Strategy

Direct execution strategy using MCPToolManager for subprocess-based tool execution.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .mcp_client_base import MCPStrategy
from ..tool_management.mcp_tool_manager import MCPToolManager

logger = logging.getLogger(__name__)


class DirectStrategy(MCPStrategy):
    """Strategy for direct tool execution via MCPToolManager."""
    
    def __init__(self, tools_dir: Optional[Path] = None):
        self.tools_dir = tools_dir
        self._manager = None
    
    @property
    def strategy_name(self) -> str:
        return "direct"
    
    @property
    def manager(self) -> MCPToolManager:
        """Lazy-loaded MCPToolManager instance."""
        if self._manager is None:
            if self.tools_dir:
                self._manager = MCPToolManager.get_instance(self.tools_dir)
            else:
                self._manager = MCPToolManager.get_instance()
        return self._manager
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool directly via MCPToolManager."""
        try:
            result = self.manager.execute_tool(tool_name, **params)
            # Ensure result has strategy info
            if isinstance(result, dict):
                result["strategy"] = self.strategy_name
            return result
        except Exception as e:
            logger.error(f"Direct execution failed for {tool_name}: {e}")
            return {
                "success": False,
                "error": f"Direct execution failed: {str(e)}",
                "strategy": self.strategy_name,
                "tool_name": tool_name
            }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List tools via MCPToolManager discovery."""
        try:
            tools = self.manager.discover_tools()
            # Add strategy info to each tool
            for tool in tools:
                tool["available_via"] = tool.get("available_via", []) + [self.strategy_name]
            return tools
        except Exception as e:
            logger.error(f"Direct tool listing failed: {e}")
            return []
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get tool info from discovered tools."""
        try:
            discovered_tools = self.manager.discovered_tools
            if tool_name in discovered_tools:
                tool_info = discovered_tools[tool_name].copy()
                tool_info["strategy"] = self.strategy_name
                return {
                    "success": True,
                    "tool_info": tool_info,
                    "strategy": self.strategy_name
                }
            else:
                return {
                    "success": False,
                    "error": f"Tool not found: {tool_name}",
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
        """Check if direct execution is available."""
        try:
            # Check if we can create/access the manager
            _ = self.manager
            return True
        except Exception as e:
            logger.debug(f"Direct strategy not available: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for direct execution."""
        try:
            if self.is_available():
                tools_count = len(self.manager.discover_tools())
                return {
                    "status": "healthy",
                    "strategy": self.strategy_name,
                    "tools_available": tools_count,
                    "tools_directory": str(self.manager.tools_dir)
                }
            else:
                return {
                    "status": "unhealthy",
                    "strategy": self.strategy_name,
                    "error": "MCPToolManager not available"
                }
        except Exception as e:
            logger.error(f"Direct strategy health check failed: {e}")
            return {
                "status": "unhealthy",
                "strategy": self.strategy_name,
                "error": str(e)
            }
