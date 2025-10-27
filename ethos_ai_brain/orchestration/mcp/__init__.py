"""
Ethos-MCP: Universal MCP Server with API Wrapper capabilities

A standalone MCP server that provides tools and capabilities with Universal API Wrapper support.
"""

__version__ = "0.1.0"
__author__ = "Damon Weiss"
__email__ = "damon@example.com"

# TODO: Fix imports when server structure is clarified
# from .mcp.server import UnifiedMCPServer
# from .mcp.client import EthosMCPClient, AsyncEthosMCPClient
# TODO: Fix server imports - directory name with hyphen causes import issues
# Placeholder classes until server structure is fixed
UnifiedMCPServer = None
EthosMCPClient = None
AsyncEthosMCPClient = None

__all__ = [
    "UnifiedMCPServer",
    "EthosMCPClient", 
    "AsyncEthosMCPClient",
]
