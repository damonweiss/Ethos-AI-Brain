#!/usr/bin/env python3
"""
MCP Client Router

Main router class that manages multiple MCP strategies and provides intelligent routing.
"""

import logging
from typing import Any, Dict, List, Optional

from .mcp_client_base import MCPConfig, MCPStrategy
from .mcp_client_direct import DirectStrategy
from .mcp_client_json_rpc import JsonRpcStrategy

logger = logging.getLogger(__name__)


class HybridStrategy(MCPStrategy):
    """Strategy that routes tools to different backends based on rules."""
    
    def __init__(self, 
                 direct_strategy: DirectStrategy,
                 json_rpc_strategy: JsonRpcStrategy,
                 config: MCPConfig):
        self.direct_strategy = direct_strategy
        self.json_rpc_strategy = json_rpc_strategy
        self.config = config
    
    @property
    def strategy_name(self) -> str:
        return "hybrid"
    
    def _get_strategy_for_tool(self, tool_name: str) -> MCPStrategy:
        """Determine which strategy to use for a specific tool."""
        routing_config = self.config.routing_config
        
        # Check explicit tool rules
        if tool_name in routing_config.routing_rules:
            mode = routing_config.routing_rules[tool_name]
            if mode == "direct":
                return self.direct_strategy
            elif mode == "json_rpc":
                return self.json_rpc_strategy
        
        # Check namespace rules
        for namespace, mode in routing_config.namespace_rules.items():
            if tool_name.startswith(namespace):
                if mode == "direct":
                    return self.direct_strategy
                elif mode == "json_rpc":
                    return self.json_rpc_strategy
        
        # Default: prefer JSON-RPC if available, fallback to direct
        if self.json_rpc_strategy.is_available():
            return self.json_rpc_strategy
        else:
            return self.direct_strategy
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool using appropriate strategy."""
        strategy = self._get_strategy_for_tool(tool_name)
        result = strategy.execute_tool(tool_name, params)
        
        # Add hybrid routing info
        if isinstance(result, dict):
            result["routed_via"] = self.strategy_name
            result["executed_by"] = strategy.strategy_name
        
        return result
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List tools from all available strategies."""
        tools = []
        
        # Get tools from direct strategy
        if self.direct_strategy.is_available():
            direct_tools = self.direct_strategy.list_tools()
            tools.extend(direct_tools)
        
        # Get tools from JSON-RPC strategy
        if self.json_rpc_strategy.is_available():
            json_rpc_tools = self.json_rpc_strategy.list_tools()
            tools.extend(json_rpc_tools)
        
        # Deduplicate by tool name and merge availability info
        seen_tools = {}
        for tool in tools:
            name = tool.get("name", "unknown")
            if name in seen_tools:
                # Merge availability info
                existing_via = seen_tools[name].get("available_via", [])
                new_via = tool.get("available_via", [])
                seen_tools[name]["available_via"] = list(set(existing_via + new_via))
            else:
                seen_tools[name] = tool
        
        return list(seen_tools.values())
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get tool info using appropriate strategy."""
        strategy = self._get_strategy_for_tool(tool_name)
        result = strategy.get_tool_info(tool_name)
        
        # Add hybrid routing info
        if isinstance(result, dict):
            result["routed_via"] = self.strategy_name
            result["queried_by"] = strategy.strategy_name
        
        return result
    
    def is_available(self) -> bool:
        """Check if any strategy is available."""
        return (self.direct_strategy.is_available() or 
                self.json_rpc_strategy.is_available())
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for all strategies."""
        return {
            "status": "healthy" if self.is_available() else "unhealthy",
            "strategy": self.strategy_name,
            "strategies": {
                "direct": self.direct_strategy.health_check(),
                "json_rpc": self.json_rpc_strategy.health_check()
            }
        }


class MCPClientRouter:
    """Main MCP client router that manages multiple execution strategies."""
    
    def __init__(self, config: Optional[MCPConfig] = None):
        self.config = config or MCPConfig()
        self._strategy = None
        self._initialize_strategy()
    
    def _initialize_strategy(self):
        """Initialize the execution strategy based on configuration."""
        # Create individual strategies
        direct_strategy = DirectStrategy(self.config.tools_directory)
        json_rpc_strategy = JsonRpcStrategy(self.config.server_url)
        
        # Choose strategy based on mode
        if self.config.default_mode == "direct":
            self._strategy = direct_strategy
        elif self.config.default_mode == "json_rpc":
            self._strategy = json_rpc_strategy
        elif self.config.default_mode == "hybrid":
            self._strategy = HybridStrategy(direct_strategy, json_rpc_strategy, self.config)
        elif self.config.default_mode == "auto":
            # Auto-detect best available mode
            if json_rpc_strategy.is_available():
                self._strategy = json_rpc_strategy
                logger.info("Auto-detected mode: json_rpc")
            elif direct_strategy.is_available():
                self._strategy = direct_strategy
                logger.info("Auto-detected mode: direct")
            else:
                # Fallback to hybrid for maximum compatibility
                self._strategy = HybridStrategy(direct_strategy, json_rpc_strategy, self.config)
                logger.warning("Auto-detection failed, using hybrid mode")
        else:
            raise ValueError(f"Unknown mode: {self.config.default_mode}")
    
    # Public API - same interface regardless of backend
    def execute_tool(self, tool_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a tool with given parameters."""
        return self._strategy.execute_tool(tool_name, params or {})
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return self._strategy.list_tools()
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get information about a specific tool."""
        return self._strategy.get_tool_info(tool_name)
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the MCP client."""
        result = self._strategy.health_check()
        result["router_config"] = {
            "mode": self.config.default_mode,
            "server_url": self.config.server_url,
            "tools_directory": str(self.config.tools_directory) if self.config.tools_directory else None
        }
        return result
    
    def is_available(self) -> bool:
        """Check if the client is available for use."""
        return self._strategy.is_available()
    
    def refresh_tools(self) -> Dict[str, Any]:
        """Refresh the tool registry."""
        if hasattr(self._strategy, 'refresh_tools'):
            return self._strategy.refresh_tools()
        else:
            # For strategies that don't support refresh, reinitialize
            self._initialize_strategy()
            return {
                "success": True,
                "message": "Strategy reinitialized",
                "strategy": self._strategy.strategy_name
            }
    
    # Convenience methods
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        tools = self.list_tools()
        return [tool.get("name", "unknown") for tool in tools]
    
    def get_tools_by_namespace(self, namespace: str) -> List[Dict[str, Any]]:
        """Get tools filtered by namespace."""
        tools = self.list_tools()
        return [tool for tool in tools 
                if tool.get("namespace", "").startswith(namespace)]
    
    @property
    def current_strategy(self) -> str:
        """Get the name of the current strategy."""
        return self._strategy.strategy_name
