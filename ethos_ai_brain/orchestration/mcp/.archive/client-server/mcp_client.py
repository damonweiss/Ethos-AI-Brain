#!/usr/bin/env python3
"""
Unified MCP Client

A unified client that supports both direct execution and MCP server modes.
Provides automatic mode detection, fallback chains, and flexible routing.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

# Import existing implementations
from .client_old import EthosMCPClient, AsyncEthosMCPClient
from ...mcp_tools.mcp_tool_manager import MCPToolManager

logger = logging.getLogger(__name__)


@dataclass
class ToolRoutingConfig:
    """Configuration for routing tools to different backends."""
    routing_rules: Dict[str, str] = field(default_factory=dict)  # tool_name -> preferred_mode
    namespace_rules: Dict[str, str] = field(default_factory=dict)  # namespace -> preferred_mode
    fallback_rules: Dict[str, List[str]] = field(default_factory=dict)  # tool -> fallback_chain


@dataclass
class UnifiedMCPConfig:
    """Configuration for the unified MCP client."""
    server_url: str = "http://localhost:8051"
    tools_directory: Optional[Path] = None
    default_mode: str = "auto"
    routing_config: ToolRoutingConfig = field(default_factory=ToolRoutingConfig)
    timeout_settings: Dict[str, int] = field(default_factory=lambda: {
        "health_check": 1,
        "tool_execution": 30,
        "tool_listing": 5
    })


class MCPExecutionStrategy(ABC):
    """Abstract strategy for MCP tool execution."""
    
    @abstractmethod
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given parameters."""
        pass
    
    @abstractmethod
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        pass
    
    @abstractmethod
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get information about a specific tool."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this strategy is currently available."""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check the health of this strategy."""
        pass


class DirectExecutionStrategy(MCPExecutionStrategy):
    """Strategy for direct tool execution via MCPToolManager."""
    
    def __init__(self, tools_dir: Optional[Path] = None):
        self.tools_dir = tools_dir
        self._manager = None
    
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
            return self.manager.execute_tool(tool_name, **params)
        except Exception as e:
            return {
                "success": False,
                "error": f"Direct execution failed: {str(e)}",
                "strategy": "direct"
            }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List tools via MCPToolManager discovery."""
        try:
            return self.manager.discover_tools()
        except Exception as e:
            logger.error(f"Direct tool listing failed: {e}")
            return []
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get tool info from discovered tools."""
        try:
            discovered_tools = self.manager.discovered_tools
            if tool_name in discovered_tools:
                return {
                    "success": True,
                    "tool_info": discovered_tools[tool_name],
                    "strategy": "direct"
                }
            else:
                return {
                    "success": False,
                    "error": f"Tool not found: {tool_name}",
                    "strategy": "direct"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get tool info: {str(e)}",
                "strategy": "direct"
            }
    
    def is_available(self) -> bool:
        """Check if direct execution is available."""
        try:
            # Check if we can create/access the manager
            _ = self.manager
            return True
        except Exception:
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for direct execution."""
        try:
            if self.is_available():
                tools_count = len(self.manager.discover_tools())
                return {
                    "status": "healthy",
                    "strategy": "direct",
                    "tools_available": tools_count,
                    "tools_directory": str(self.manager.tools_dir)
                }
            else:
                return {
                    "status": "unhealthy",
                    "strategy": "direct",
                    "error": "MCPToolManager not available"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "strategy": "direct",
                "error": str(e)
            }


class ServerExecutionStrategy(MCPExecutionStrategy):
    """Strategy for execution via MCP server."""
    
    def __init__(self, server_url: str = "http://localhost:8051"):
        self.server_url = server_url
        self._client = EthosMCPClient(server_url)
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool via MCP server."""
        result = self._client.execute_tool(tool_name, params)
        result["strategy"] = "server"
        return result
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List tools via MCP server."""
        try:
            response = self._client.list_tools()
            if "error" in response:
                logger.error(f"Server tool listing failed: {response['error']}")
                return []
            return response.get("tools", [])
        except Exception as e:
            logger.error(f"Server tool listing failed: {e}")
            return []
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get tool info via MCP server."""
        result = self._client.get_tool_info(tool_name)
        result["strategy"] = "server"
        return result
    
    def is_available(self) -> bool:
        """Check if MCP server is available."""
        return self._client.is_server_running()
    
    def health_check(self) -> Dict[str, Any]:
        """Health check via MCP server."""
        result = self._client.health_check()
        result["strategy"] = "server"
        return result


class HybridExecutionStrategy(MCPExecutionStrategy):
    """Strategy that routes tools to different backends based on rules."""
    
    def __init__(self, 
                 direct_strategy: DirectExecutionStrategy,
                 server_strategy: ServerExecutionStrategy,
                 routing_config: ToolRoutingConfig):
        self.direct_strategy = direct_strategy
        self.server_strategy = server_strategy
        self.routing_config = routing_config
    
    def _get_strategy_for_tool(self, tool_name: str) -> MCPExecutionStrategy:
        """Determine which strategy to use for a specific tool."""
        # Check explicit tool rules
        if tool_name in self.routing_config.routing_rules:
            mode = self.routing_config.routing_rules[tool_name]
            if mode == "direct":
                return self.direct_strategy
            elif mode == "server":
                return self.server_strategy
        
        # Check namespace rules
        for namespace, mode in self.routing_config.namespace_rules.items():
            if tool_name.startswith(namespace):
                if mode == "direct":
                    return self.direct_strategy
                elif mode == "server":
                    return self.server_strategy
        
        # Default: prefer server if available, fallback to direct
        if self.server_strategy.is_available():
            return self.server_strategy
        else:
            return self.direct_strategy
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool using appropriate strategy."""
        strategy = self._get_strategy_for_tool(tool_name)
        return strategy.execute_tool(tool_name, params)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List tools from all available strategies."""
        tools = []
        
        # Get tools from direct strategy
        if self.direct_strategy.is_available():
            direct_tools = self.direct_strategy.list_tools()
            for tool in direct_tools:
                tool["available_via"] = tool.get("available_via", []) + ["direct"]
            tools.extend(direct_tools)
        
        # Get tools from server strategy
        if self.server_strategy.is_available():
            server_tools = self.server_strategy.list_tools()
            for tool in server_tools:
                tool["available_via"] = tool.get("available_via", []) + ["server"]
            tools.extend(server_tools)
        
        # Deduplicate by tool name
        seen_tools = {}
        for tool in tools:
            name = tool.get("name", "unknown")
            if name in seen_tools:
                # Merge availability info
                seen_tools[name]["available_via"].extend(tool.get("available_via", []))
            else:
                seen_tools[name] = tool
        
        return list(seen_tools.values())
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get tool info using appropriate strategy."""
        strategy = self._get_strategy_for_tool(tool_name)
        return strategy.get_tool_info(tool_name)
    
    def is_available(self) -> bool:
        """Check if any strategy is available."""
        return (self.direct_strategy.is_available() or 
                self.server_strategy.is_available())
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for all strategies."""
        return {
            "status": "healthy" if self.is_available() else "unhealthy",
            "strategy": "hybrid",
            "direct": self.direct_strategy.health_check(),
            "server": self.server_strategy.health_check()
        }


class UnifiedMCPClient:
    """Unified MCP client supporting multiple execution modes."""
    
    def __init__(self, config: Optional[UnifiedMCPConfig] = None):
        self.config = config or UnifiedMCPConfig()
        self._strategy = None
        self._initialize_strategy()
    
    def _initialize_strategy(self):
        """Initialize the execution strategy based on configuration."""
        # Create individual strategies
        direct_strategy = DirectExecutionStrategy(self.config.tools_directory)
        server_strategy = ServerExecutionStrategy(self.config.server_url)
        
        # Choose strategy based on mode
        if self.config.default_mode == "direct":
            self._strategy = direct_strategy
        elif self.config.default_mode == "server":
            self._strategy = server_strategy
        elif self.config.default_mode == "hybrid":
            self._strategy = HybridExecutionStrategy(
                direct_strategy, 
                server_strategy, 
                self.config.routing_config
            )
        elif self.config.default_mode == "auto":
            # Auto-detect best available mode
            if server_strategy.is_available():
                self._strategy = server_strategy
                logger.info("Auto-detected mode: server")
            elif direct_strategy.is_available():
                self._strategy = direct_strategy
                logger.info("Auto-detected mode: direct")
            else:
                # Fallback to hybrid for maximum compatibility
                self._strategy = HybridExecutionStrategy(
                    direct_strategy, 
                    server_strategy, 
                    self.config.routing_config
                )
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
        return self._strategy.health_check()
    
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
            return {"status": "refreshed", "message": "Strategy reinitialized"}
    
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


# Factory functions for common configurations
def create_auto_client(tools_dir: Optional[Path] = None, 
                      server_url: str = "http://localhost:8051") -> UnifiedMCPClient:
    """Create a client with automatic mode detection."""
    config = UnifiedMCPConfig(
        server_url=server_url,
        tools_directory=tools_dir,
        default_mode="auto"
    )
    return UnifiedMCPClient(config)


def create_hybrid_client(tools_dir: Optional[Path] = None,
                        server_url: str = "http://localhost:8051",
                        routing_rules: Optional[Dict[str, str]] = None) -> UnifiedMCPClient:
    """Create a client with hybrid mode and custom routing."""
    routing_config = ToolRoutingConfig(
        routing_rules=routing_rules or {},
        namespace_rules={
            "brain.": "direct",
            "actions.": "server",
            "web.": "server"
        }
    )
    config = UnifiedMCPConfig(
        server_url=server_url,
        tools_directory=tools_dir,
        default_mode="hybrid",
        routing_config=routing_config
    )
    return UnifiedMCPClient(config)


# Example usage
if __name__ == "__main__":
    # Test the unified client
    client = create_auto_client()
    
    print("=== Unified MCP Client Test ===")
    
    # Health check
    health = client.health_check()
    print(f"Health: {health}")
    
    # List tools
    tools = client.list_tools()
    print(f"Available tools: {len(tools)}")
    for tool in tools[:3]:  # Show first 3
        print(f"  - {tool.get('name', 'unknown')}")
    
    # Test tool execution (if tools available)
    if tools:
        tool_name = tools[0].get("name")
        if tool_name:
            print(f"\nTesting tool: {tool_name}")
            result = client.execute_tool(tool_name, {})
            print(f"Result: {result.get('success', False)}")
