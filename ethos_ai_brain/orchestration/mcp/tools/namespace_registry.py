#!/usr/bin/env python3
"""
MCP Tool Namespace Registry

Simple infrastructure for organizing and discovering MCP tools.
Configuration-driven approach using YAML for namespace definitions.
No hardcoded intelligence - pure tool organization.
"""

import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path


class ConfigurableNamespaceRegistry:
    """Configuration-driven namespace registry for MCP tools."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize registry with YAML configuration."""
        if config_file is None:
            # Default to config file in ethos_mcp package
            config_file = Path(__file__).parent.parent / "ethos_mcp" / "config" / "namespaces.yaml"
        
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        # Simple tool organization - no intelligence
        self.tools_by_namespace: Dict[str, List[Dict[str, Any]]] = {}
        self.tools_by_category: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize namespace containers
        for namespace in self.config.get("namespaces", {}):
            self.tools_by_namespace[namespace] = []
    
    def _load_config(self) -> Dict[str, Any]:
        """Load namespace configuration from YAML file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    return yaml.safe_load(f) or {}
            else:
                print(f"Warning: Config file not found: {self.config_file}")
                return self._get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Fallback configuration if YAML file not available."""
        return {
            "namespaces": {
                "default": {
                    "description": "Default namespace for unclassified tools",
                    "task_patterns": [["general", "utility", "basic"]]
                }
            }
        }
    
    def register_tool(self, tool_info: Dict[str, Any]):
        """Register a tool in the appropriate namespace and category."""
        tool_name = tool_info.get("name", "")
        declared_namespace = tool_info.get("namespace", "default")
        declared_category = tool_info.get("category", "general")
        
        # Validate namespace against configuration
        if declared_namespace not in self.config.get("namespaces", {}):
            print(f"Warning: Tool {tool_name} declares unknown namespace: {declared_namespace}")
            declared_namespace = "default"
        
        # Add to namespace mapping
        if declared_namespace not in self.tools_by_namespace:
            self.tools_by_namespace[declared_namespace] = []
        self.tools_by_namespace[declared_namespace].append(tool_info)
        
        # Add to category mapping
        if declared_category not in self.tools_by_category:
            self.tools_by_category[declared_category] = []
        self.tools_by_category[declared_category].append(tool_info)
    
    def get_tools_by_namespace(self, namespace: str) -> List[Dict[str, Any]]:
        """Get all tools in a specific namespace."""
        return self.tools_by_namespace.get(namespace, [])
    
    def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all tools in a specific category."""
        return self.tools_by_category.get(category, [])
    
    def list_namespaces(self) -> List[str]:
        """List all available namespaces from configuration."""
        return list(self.config.get("namespaces", {}).keys())
    
    def get_namespace_info(self, namespace: str) -> Optional[Dict[str, Any]]:
        """Get information about a namespace from configuration."""
        return self.config.get("namespaces", {}).get(namespace)
    
    def get_task_patterns(self, namespace: str) -> List[List[str]]:
        """Get task patterns for a namespace."""
        namespace_config = self.get_namespace_info(namespace)
        if namespace_config:
            return namespace_config.get("task_patterns", [])
        return []
    
    def get_tool_capabilities(self) -> Dict[str, Any]:
        """Get a summary of all tool capabilities."""
        capabilities = {
            "namespaces": {},
            "categories": {},
            "total_tools": 0
        }
        
        # Namespace summary
        for namespace, tools in self.tools_by_namespace.items():
            namespace_config = self.get_namespace_info(namespace)
            capabilities["namespaces"][namespace] = {
                "description": namespace_config.get("description", "") if namespace_config else "",
                "tool_count": len(tools),
                "tools": [tool["name"] for tool in tools]
            }
        
        # Category summary  
        for category, tools in self.tools_by_category.items():
            capabilities["categories"][category] = len(tools)
        
        # Total count
        capabilities["total_tools"] = sum(len(tools) for tools in self.tools_by_namespace.values())
        
        return capabilities


# Utility functions for backward compatibility
def get_namespaced_tool_name(namespace: str, tool_name: str) -> str:
    """Generate a namespaced tool name."""
    return f"{namespace}/{tool_name}" if namespace != "default" else tool_name


def parse_namespaced_tool_name(full_name: str) -> tuple[str, str]:
    """Parse a namespaced tool name into namespace and tool name."""
    if "/" in full_name:
        namespace, tool_name = full_name.split("/", 1)
        return namespace, tool_name
    return "default", full_name


# Global registry instance (backward compatibility)
namespace_registry = ConfigurableNamespaceRegistry()


if __name__ == "__main__":
    # Demo the new configuration-driven namespace system
    registry = ConfigurableNamespaceRegistry()
    
    print("=== YAML-Configured Namespace Registry Demo ===")
    print(f"Available namespaces: {registry.list_namespaces()}")
    
    # Show namespace configurations
    for namespace in registry.list_namespaces():
        info = registry.get_namespace_info(namespace)
        if info:
            print(f"\n{namespace}:")
            print(f"  Description: {info.get('description', 'N/A')}")
            print(f"  Task patterns: {info.get('task_patterns', [])}")
    
    print(f"\nCapabilities: {registry.get_tool_capabilities()}")
