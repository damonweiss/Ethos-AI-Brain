"""
MCP Tool Manager

Manages MCP tools with pyproject.toml metadata and uv dependency management.
Provides tool discovery, execution, and namespace organization.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from .namespace_registry import namespace_registry, get_namespaced_tool_name, parse_namespaced_tool_name
import logging

logger = logging.getLogger(__name__)

class MCPToolManager:
    """Manages MCP tools with integrated dependency management and namespace support."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, tools_dir: Path = None):
        if cls._instance is None:
            cls._instance = super(MCPToolManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, tools_dir: Path = None):
        # Only initialize once
        if MCPToolManager._initialized:
            return
            
        if tools_dir is None:
            # Default tools directory
            tools_dir = Path(__file__).parent
            
        self.tools_dir = tools_dir
        self.discovered_tools = {}
        MCPToolManager._initialized = True
        logger.info(f"MCPToolManager singleton initialized with tools_dir: {tools_dir}")
    
    @classmethod
    def get_instance(cls, tools_dir: Path = None) -> 'MCPToolManager':
        """Get the singleton instance of MCPToolManager."""
        if cls._instance is None:
            cls._instance = cls(tools_dir)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None
        cls._initialized = False
        
    def discover_tools(self) -> List[Dict[str, Any]]:
        """
        Discover all MCP tools with pyproject.toml metadata.
        Supports namespace organization and dependency management.
        """
        tools = []
        
        # Find all pyproject.toml files (each represents a tool)
        for pyproject_file in self.tools_dir.rglob('pyproject.toml'):
            # Skip base environments and non-tool projects
            if '_shared_envs' in str(pyproject_file):
                continue
                
            tool_dir = pyproject_file.parent
            tool_info = self._analyze_tool(tool_dir)
            
            if tool_info:
                tools.extend(tool_info)
        
        return tools
    
    def _analyze_tool(self, tool_dir: Path) -> Optional[Dict[str, Any]]:
        """Analyze a tool directory and extract metadata."""
        try:
            pyproject_file = tool_dir / 'pyproject.toml'
            tool_py_file = tool_dir / 'tool.py'
            
            if not pyproject_file.exists():
                return None
            
            # Parse pyproject.toml for metadata
            import tomllib
            with open(pyproject_file, 'rb') as f:
                pyproject_data = tomllib.load(f)
            
            # Extract tool metadata from pyproject.toml
            project_info = pyproject_data.get('project', {})
            tool_name = project_info.get('name', tool_dir.name)
            
            # Load tool.py if it exists to get TOOLS metadata
            tools_metadata = []
            if tool_py_file.exists():
                spec = importlib.util.spec_from_file_location(f"{tool_name}_module", tool_py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    
                    # Suppress stdout during tool loading to avoid spam
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    try:
                        sys.stdout = open(os.devnull, 'w')
                        sys.stderr = open(os.devnull, 'w')
                        spec.loader.exec_module(module)
                    finally:
                        sys.stdout.close()
                        sys.stderr.close()
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                    
                    if hasattr(module, 'TOOLS'):
                        tools_metadata = module.TOOLS
            
            # Process each tool in the TOOLS list
            enhanced_tools = []
            for tool_info in tools_metadata:
                tool_name = tool_info.get('name', '')
                if tool_name:
                    # Determine namespace from tool metadata or directory structure
                    namespace = self._determine_namespace(tool_info, tool_dir)
                    
                    # Create namespaced tool name - respect tool's own namespace declaration
                    if tool_info.get('namespace'):
                        # Tool already declares its namespace, use the name as-is
                        namespaced_name = tool_name
                    elif namespace and namespace != "default":
                        # Tool doesn't declare namespace, infer from directory structure
                        namespaced_name = get_namespaced_tool_name(namespace, tool_name)
                    else:
                        namespaced_name = tool_name
                    
                    # Add metadata including namespace info
                    enhanced_tool_info = tool_info.copy()
                    enhanced_tool_info.update({
                        'tool_dir': str(tool_dir),
                        'pyproject_file': str(pyproject_file),
                        'tool_py_file': str(tool_py_file) if tool_py_file.exists() else None,
                        'uv_managed': True,
                        'namespace': namespace,
                        'namespaced_name': namespaced_name,
                        'original_name': tool_name,
                        'project_info': project_info
                    })
                    
                    # Add dependency information
                    dependencies, dep_source = self._extract_dependencies(pyproject_data)
                    enhanced_tool_info['dependencies'] = dependencies
                    enhanced_tool_info['dependency_source'] = dep_source
                    enhanced_tool_info['complexity'] = self._assess_complexity(dependencies)
                    
                    enhanced_tools.append(enhanced_tool_info)
                    
                    # Store in discovered_tools for execution lookup
                    self.discovered_tools[namespaced_name] = enhanced_tool_info
                    
                    # Register with namespace registry
                    try:
                        namespace_registry.register_tool(enhanced_tool_info)
                    except Exception as e:
                        logger.warning(f"Failed to register tool {tool_name} with namespace registry: {e}")
            
            return enhanced_tools
            
        except Exception as e:
            logger.error(f"Failed to analyze tool {tool_dir}: {e}")
            return None
    
    def _determine_namespace(self, tool_info: Dict[str, Any], tool_dir: Path) -> str:
        """Determine the namespace for a tool based on metadata and directory structure."""
        # Check if namespace is explicitly defined in tool metadata
        if 'namespace' in tool_info:
            return tool_info['namespace']
        
        # Infer from directory structure
        # Convert path parts to namespace (e.g., utilities/web -> utilities.web)
        relative_path = tool_dir.relative_to(self.tools_dir)
        path_parts = relative_path.parts[:-1]  # Exclude the tool directory itself
        
        if path_parts:
            return '.'.join(path_parts)
        
        return 'default'
    
    def _extract_dependencies(self, pyproject_data: Dict[str, Any]) -> tuple[List[str], str]:
        """Extract dependencies from pyproject.toml."""
        dependencies = []
        dependency_source = "none"
        
        # Try project dependencies first
        project_deps = pyproject_data.get('project', {}).get('dependencies', [])
        if project_deps:
            dependencies.extend(project_deps)
            dependency_source = "project.dependencies"
        
        # Try tool.uv dependencies as fallback
        if not dependencies:
            uv_deps = pyproject_data.get('tool', {}).get('uv', {}).get('dependencies', [])
            if uv_deps:
                dependencies.extend(uv_deps)
                dependency_source = "tool.uv.dependencies"
        
        return dependencies, dependency_source
    
    def _assess_complexity(self, dependencies: List[str]) -> str:
        """Assess tool complexity based on dependencies."""
        if not dependencies:
            return 'simple'
        
        heavy_deps = ['torch', 'tensorflow', 'transformers', 'opencv', 'selenium']
        medium_deps = ['requests', 'beautifulsoup4', 'pandas', 'numpy', 'matplotlib']
        
        dep_str = ' '.join(dependencies).lower()
        
        if any(heavy in dep_str for heavy in heavy_deps):
            return 'complex'
        elif any(medium in dep_str for medium in medium_deps):
            return 'medium'
        else:
            return 'simple'
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name with given parameters."""
        print(f"[DEBUG] Executing tool: {tool_name}", file=sys.stderr)
        print(f"[DEBUG] Available in discovered_tools: {list(self.discovered_tools.keys())}", file=sys.stderr)
        
        # Handle both namespaced and non-namespaced tool names
        tool_info = self.discovered_tools.get(tool_name)
        
        # If not found with exact name, try to find by original name
        if not tool_info:
            namespace, original_name = parse_namespaced_tool_name(tool_name)
            for tool in self.discovered_tools.values():
                if tool.get('original_name') == original_name or tool.get('name') == tool_name:
                    tool_info = tool
                    break
        
        if not tool_info:
            return {
                'success': False,
                'error': f'Tool not found: {tool_name}',
                'available_tools': list(self.discovered_tools.keys())
            }
        
        # Execute function directly
        if 'function' in tool_info and callable(tool_info['function']):
            try:
                print(f"[DEBUG] Executing function directly: {tool_info['function'].__name__}", file=sys.stderr)
                
                # Try to determine function signature to decide how to call it
                import inspect
                func = tool_info['function']
                sig = inspect.signature(func)
                
                # If function takes a single 'params' parameter, pass kwargs as dict
                if len(sig.parameters) == 1 and 'params' in sig.parameters:
                    result = func(kwargs)
                else:
                    # Otherwise, unpack kwargs as individual parameters
                    result = func(**kwargs)
                
                return {
                    'success': True,
                    'result': result,
                    'tool_name': tool_name,
                    'execution_method': 'direct_function'
                }
            except Exception as e:
                logger.error(f"Direct function execution failed for {tool_name}: {e}")
                return {
                    'success': False,
                    'error': f'Function execution failed: {type(e).__name__}: {str(e)}',
                    'tool_name': tool_name,
                    'execution_method': 'direct_function'
                }
        else:
            return {
                'success': False,
                'error': f'Tool {tool_name} has no executable function',
                'tool_name': tool_name,
                'execution_method': 'none'
            }


def demo_tool_manager():
    """Demo the MCP tool manager."""
    print("=== MCP Tool Manager Demo ===")
    
    # Create manager
    tools_dir = Path(__file__).parent
    manager = MCPToolManager(tools_dir)
    
    # Discover tools
    tools = manager.discover_tools()
    
    print(f"Found {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool.get('name', 'Unknown')}")
        print(f"    Namespace: {tool.get('namespace', 'default')}")
        print(f"    Dependencies: {len(tool.get('dependencies', []))}")
        print(f"    Complexity: {tool.get('complexity', 'unknown')}")
        print()


if __name__ == "__main__":
    demo_tool_manager()
