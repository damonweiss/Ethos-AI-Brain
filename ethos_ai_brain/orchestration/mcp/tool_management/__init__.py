"""
MCP Tool Management

Consolidated tool management system for MCP.
Includes tool manager, namespace registry, validation, and wrappers.
"""

from .mcp_tool_manager import MCPToolManager
from .namespace_registry import namespace_registry, get_namespaced_tool_name, parse_namespaced_tool_name

# Import wrappers if they exist
try:
    from .tool_wrappers import APIWrapper, FunctionWrapper, ModuleWrapper
    _wrappers_available = True
except ImportError:
    _wrappers_available = False

__all__ = [
    "MCPToolManager",
    "namespace_registry",
    "get_namespaced_tool_name", 
    "parse_namespaced_tool_name"
]

if _wrappers_available:
    __all__.extend(["APIWrapper", "FunctionWrapper", "ModuleWrapper"])

__version__ = "1.0.0"
