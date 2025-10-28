"""
MCP Tools Package

UV-First dynamic tool discovery system.
Every tool has a pyproject.toml for metadata and versioning.
"""

import os
import importlib.util
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def discover_tools() -> List[Dict[str, Any]]:
    """
    Discover MCP tools with dependency management.
    
    Priority:
    1. Modern tools (pyproject.toml + tool.py)
    2. Legacy tools (tool.py only) - for backward compatibility
    """
    # Try modern tool discovery
    try:
        from .mcp_tool_manager import MCPToolManager
        tools_dir = Path(__file__).parent
        manager = MCPToolManager.get_instance(tools_dir)
        tools = manager.discover_tools()
        
        logger.info(f"MCP tool discovery attempt: found {len(tools)} tools")
        for tool in tools:
            logger.info(f"  - {tool['name']}: uv_managed={tool.get('uv_managed', False)}")
        
        if tools:
            logger.info(f"Using MCP tool discovery: found {len(tools)} tools")
            return tools
        else:
            logger.warning("MCP tool discovery found no tools, falling back to legacy")
    except Exception as e:
        logger.warning(f"MCP tool discovery failed, falling back to legacy: {e}")
        import traceback
        logger.warning(f"MCP tool discovery traceback: {traceback.format_exc()}")
    
    # Fallback to legacy discovery
    return discover_tools_legacy()


def discover_tools_legacy() -> List[Dict[str, Any]]:
    """
    Dynamically discover all tools in the mcp_tools directory structure.
    
    Searches for tool.py files in subdirectories and loads their TOOLS metadata.
    Also checks for tool-specific requirements.txt files.
    
    Returns:
        List of tool dictionaries with name, description, function, and parameters
    """
    tools = []
    tools_dir = Path(__file__).parent
    
    # Walk through all subdirectories looking for tool.py files
    for tool_file in tools_dir.rglob('tool.py'):
        try:
            # Get relative path for module name
            relative_path = tool_file.relative_to(tools_dir)
            module_path = str(relative_path.with_suffix(''))
            module_name = f"mcp_tools.{module_path.replace(os.sep, '.')}"
            
            # Check for tool-specific requirements (both requirements.txt and pyproject.toml)
            requirements_file = tool_file.parent / 'requirements.txt'
            pyproject_file = tool_file.parent / 'pyproject.toml'
            tool_requirements = []
            dependency_source = None
            
            # Try requirements.txt first
            if requirements_file.exists():
                try:
                    tool_requirements = requirements_file.read_text().strip().split('\n')
                    tool_requirements = [req.strip() for req in tool_requirements if req.strip()]
                    dependency_source = str(requirements_file)
                except Exception as e:
                    logger.warning(f"Failed to read requirements from {requirements_file}: {e}")
            
            # Try pyproject.toml if no requirements.txt
            elif pyproject_file.exists():
                try:
                    # Try tomllib (Python 3.11+) first, then tomli as fallback
                    try:
                        import tomllib
                        with open(pyproject_file, 'rb') as f:
                            pyproject_data = tomllib.load(f)
                    except ImportError:
                        import tomli
                        with open(pyproject_file, 'rb') as f:
                            pyproject_data = tomli.load(f)
                    
                    # Look for dependencies in various locations
                    deps = []
                    if 'project' in pyproject_data and 'dependencies' in pyproject_data['project']:
                        deps.extend(pyproject_data['project']['dependencies'])
                    
                    if 'tool' in pyproject_data and 'uv' in pyproject_data['tool'] and 'dependencies' in pyproject_data['tool']['uv']:
                        deps.extend(pyproject_data['tool']['uv']['dependencies'])
                    
                    tool_requirements = deps
                    dependency_source = str(pyproject_file)
                    
                except ImportError as e:
                    logger.warning(f"Neither tomllib nor tomli available for parsing {pyproject_file}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to read pyproject.toml from {pyproject_file}: {e}")
            
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, tool_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Check if module has TOOLS metadata
                if hasattr(module, 'TOOLS'):
                    discovered_tools = module.TOOLS
                    if isinstance(discovered_tools, list):
                        # Add requirements info to each tool
                        for tool in discovered_tools:
                            if tool_requirements:
                                tool['dependency_source'] = dependency_source
                                tool['external_dependencies'] = tool_requirements
                            
                        tools.extend(discovered_tools)
                        logger.info(f"Loaded {len(discovered_tools)} tools from {relative_path}")
                        if tool_requirements:
                            logger.info(f"  - Found requirements: {tool_requirements}")
                    else:
                        logger.warning(f"TOOLS in {relative_path} is not a list")
                else:
                    logger.warning(f"No TOOLS metadata found in {relative_path}")
                    
        except Exception as e:
            logger.error(f"Failed to load tools from {tool_file}: {e}")
    
    logger.info(f"Discovered {len(tools)} total tools")
    return tools


# Dynamically load all available tools
AVAILABLE_TOOLS = discover_tools()

# Export for backward compatibility
__all__ = ['AVAILABLE_TOOLS', 'discover_tools']
