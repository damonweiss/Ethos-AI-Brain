#!/usr/bin/env python3
"""
MCP Client Base Classes

Abstract base classes and configuration for MCP client strategies.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RoutingConfig:
    """Configuration for routing tools to different backends."""
    routing_rules: Dict[str, str] = field(default_factory=dict)  # tool_name -> preferred_mode
    namespace_rules: Dict[str, str] = field(default_factory=dict)  # namespace -> preferred_mode
    fallback_rules: Dict[str, List[str]] = field(default_factory=dict)  # tool -> fallback_chain


@dataclass
class MCPConfig:
    """Configuration for MCP client."""
    server_url: str = "http://localhost:8051"
    tools_directory: Optional[Path] = None
    default_mode: str = "auto"  # "auto", "direct", "json_rpc", "hybrid"
    routing_config: RoutingConfig = field(default_factory=RoutingConfig)
    timeout_settings: Dict[str, int] = field(default_factory=lambda: {
        "health_check": 1,
        "tool_execution": 30,
        "tool_listing": 5
    })


class MCPStrategy(ABC):
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
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Name of this strategy."""
        pass
