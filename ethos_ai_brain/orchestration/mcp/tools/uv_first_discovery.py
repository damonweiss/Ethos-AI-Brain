"""
UV-First Tool Discovery System

Modern approach where every tool has a pyproject.toml for metadata and versioning,
with uv handling all dependency management and execution.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import namespace registry - handle import gracefully
try:
    from .namespace_registry import namespace_registry, get_namespaced_tool_name, parse_namespaced_tool_name
except ImportError:
    # Fallback for when namespace registry isn't available
    class MockNamespaceRegistry:
        def register_tool(self, tool_info): pass
    namespace_registry = MockNamespaceRegistry()
    def get_namespaced_tool_name(namespace, name): return f"{namespace}/{name}" if namespace != "default" else name
    def parse_namespaced_tool_name(name): return name.split("/", 1) if "/" in name else ("default", name)
import logging

logger = logging.getLogger(__name__)
class UVFirstToolManager:
    """Manages MCP tools using uv-first approach."""
    
    def __init__(self, tools_dir: Path):
        self.tools_dir = tools_dir
        self.discovered_tools = {}
        
    def discover_tools(self) -> List[Dict[str, Any]]:
        """
        Discover all tools using uv-first approach.
        Every tool must have a pyproject.toml.
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
                
        logger.info(f"Discovered {len(tools)} uv-managed tools")
        return tools
    
    def _analyze_tool(self, tool_dir: Path) -> Optional[List[Dict[str, Any]]]:
        """Analyze a tool directory and extract tool metadata with namespace support."""
        pyproject_file = tool_dir / 'pyproject.toml'
        tool_py_file = tool_dir / 'tool.py'
        
        if not tool_py_file.exists():
            logger.warning(f"No tool.py found in {tool_dir}")
            return None
            
        try:
            # Read pyproject.toml metadata
            import tomli
            with open(pyproject_file, 'rb') as f:
                pyproject_data = tomli.load(f)
            
            # Extract project metadata
            project_info = pyproject_data.get('project', {})
            mcp_config = pyproject_data.get('tool', {}).get('mcp', {})
            
            # Parse dependencies
            dependencies, dependency_source = self._parse_dependencies(pyproject_data)
            
            # Debug: print tool discovery
            logger.info(f"Analyzing tool in {tool_dir}")
            logger.info(f"Found pyproject.toml: {pyproject_file.exists()}")
            logger.info(f"Found tool.py: {tool_py_file.exists()}")
            
            # Load tool.py to get TOOLS metadata
            spec = importlib.util.spec_from_file_location("tool", tool_py_file)
            if not spec or not spec.loader:
                logger.error(f"Could not load spec for {tool_py_file}")
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'TOOLS'):
                logger.warning(f"No TOOLS metadata in {tool_py_file}")
                return None
            
            tools = []
            
            # Extract tool information with namespace support
            for tool_info in module.TOOLS:
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
                        'name': namespaced_name,  # Update to namespaced name
                        'original_name': tool_name,  # Keep original for reference
                        'namespace': namespace,
                        'category': tool_info.get('category', self._infer_category(tool_name)),
                        'uv_managed': True,
                        'tool_dir': str(tool_dir),
                        'dependencies': dependencies,
                        'dependency_source': dependency_source,
                        'pyproject_path': str(pyproject_file),
                        'execution_mode': 'uv'
                    })
                    
                    tools.append(enhanced_tool_info)
                    
                    # Store in discovered_tools for execution lookup
                    self.discovered_tools[namespaced_name] = enhanced_tool_info
                    
                    # Register with namespace registry (if available)
                    try:
                        namespace_registry.register_tool(enhanced_tool_info)
                    except Exception as e:
                        logger.warning(f"Failed to register tool with namespace registry: {e}")
            
            return tools
            
        except Exception as e:
            logger.error(f"Failed to analyze tool {tool_dir}: {e}")
            return None
    
    def _determine_namespace(self, tool_info: Dict[str, Any], tool_dir: Path) -> str:
        """Determine the namespace for a tool based on metadata and directory structure."""
        # Check if namespace is explicitly defined in tool metadata
        if 'namespace' in tool_info:
            return tool_info['namespace']
        
        # Infer from directory structure
        dir_name = tool_dir.name
        parent_dir = tool_dir.parent.name
        
        # Standard namespace mappings
        if dir_name == "meta_reasoning" or parent_dir == "meta_reasoning":
            return "meta.reasoning"
        elif dir_name in ["web", "web_tools"] or parent_dir in ["web", "web_tools"]:
            return "domain.web"
        elif dir_name in ["system", "system_tools"] or parent_dir in ["system", "system_tools"]:
            return "domain.system"
        elif dir_name in ["api", "external_api"] or parent_dir in ["api", "external_api"]:
            return "external.api"
        
        # Default namespace
        return "default"
    
    def _infer_category(self, tool_name: str) -> str:
        """Infer tool category from tool name."""
        name_lower = tool_name.lower()
        
        # Meta-reasoning categories
        if any(term in name_lower for term in ["assess", "complexity", "analyze"]):
            return "complexity_analysis"
        elif any(term in name_lower for term in ["extract", "parameter", "param"]):
            return "parameter_extraction"
        elif any(term in name_lower for term in ["orchestrate", "coordinate", "manage"]):
            return "tool_orchestration"
        elif any(term in name_lower for term in ["validate", "quality", "check"]):
            return "quality_assessment"
        
        # Domain categories
        elif any(term in name_lower for term in ["scrape", "web", "html", "url"]):
            return "web_scraping"
        elif any(term in name_lower for term in ["validate", "ping", "check"]):
            return "web_validation"
        elif any(term in name_lower for term in ["time", "system", "status"]):
            return "system_info"
        elif any(term in name_lower for term in ["echo", "message", "communicate"]):
            return "communication"
        
        return "general"
    
    def _parse_dependencies(self, pyproject_data: Dict[str, Any]) -> tuple[List[str], str]:
        """Parse dependencies from pyproject.toml."""
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
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with automatic UV delegation for uv_managed tools."""
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
            print(f"[DEBUG] Tool not found: {tool_name}", file=sys.stderr)
            return {"error": f"Tool {tool_name} not found"}
        
        # Check if tool is UV-managed and delegate accordingly
        is_uv_managed = tool_info.get('uv_managed', False)
        print(f"[DEBUG] Tool {tool_name}: uv_managed={is_uv_managed}", file=sys.stderr)
        
        if is_uv_managed:
            # Use UV execution for tools with dependencies
            tool_dir = Path(tool_info['tool_dir'])
            return self._execute_with_uv(tool_dir, tool_name, params)
        else:
            # Direct function call for simple tools
            try:
                tool_function = tool_info.get('function')
                if tool_function:
                    return tool_function(params)
                else:
                    return {"error": f"Tool {tool_name} has no function"}
            except Exception as e:
                return {"error": f"Tool execution failed: {str(e)}"}
    
    def _execute_with_uv(self, tool_dir: Path, function_name: str, params: Dict) -> Dict[str, Any]:
        """Execute tool using uv run."""
        
        # Get the original function name from the tool info
        tool_info = self.discovered_tools.get(function_name)
        original_name = tool_info.get('original_name', function_name) if tool_info else function_name
        
        exec_script = f"""
import sys
import json
from pathlib import Path

# Add both current directory and parent mcp_tools directory to path
current_dir = Path.cwd()
sys.path.insert(0, str(current_dir))

# Find the mcp_tools parent directory
mcp_tools_parent = current_dir
while mcp_tools_parent.name != 'mcp_tools' and mcp_tools_parent.parent != mcp_tools_parent:
    mcp_tools_parent = mcp_tools_parent.parent

if mcp_tools_parent.name == 'mcp_tools':
    sys.path.insert(0, str(mcp_tools_parent.parent))

# Import the tool module
from tool import TOOLS

params = json.loads(r'''{json.dumps(params)}''')

# Find and execute the requested tool by original name
for tool_info in TOOLS:
    if tool_info['name'] == '{original_name}':
        try:
            result = tool_info['function'](params)
            print(json.dumps(result))
            sys.exit(0)
        except Exception as e:
            import traceback
            error_msg = f"Tool execution failed: {{str(e)}}\\nTraceback: {{traceback.format_exc()}}"
            print(json.dumps({{"error": error_msg}}))
            sys.exit(1)

print(json.dumps({{"error": "Tool {original_name} not found in TOOLS registry"}}))
sys.exit(1)
"""
        
        # Write temporary execution script
        temp_script = tool_dir / '_temp_exec.py'
        temp_script.write_text(exec_script)
        
        try:
            # Execute using uv run with environment isolation
            env = os.environ.copy()
            # Remove conflicting VIRTUAL_ENV to let UV manage its own environment
            env.pop('VIRTUAL_ENV', None)
            env.pop('CONDA_DEFAULT_ENV', None)
            
            result = subprocess.run([
                'uv', 'run', 
                '--project', str(tool_dir),
                '--isolated',  # Use isolated environment
                'python', '_temp_exec.py'
            ], 
            capture_output=True, 
            text=True,
            cwd=tool_dir,
            env=env
            )
            
            if result.returncode == 0:
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError as e:
                    return {
                        "error": f"UV execution succeeded but output is not valid JSON: {e}",
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "returncode": result.returncode
                    }
            else:
                return {
                    "error": f"UV execution failed (exit code {result.returncode})",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
                
        finally:
            # Cleanup temp script
            if temp_script.exists():
                temp_script.unlink()
    
    def create_tool_template(self, tool_name: str, tool_type: str = 'simple', 
                           base_environment: Optional[str] = None) -> Path:
        """Create a new tool with proper uv structure."""
        tool_dir = self.tools_dir / tool_name
        tool_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine dependencies based on type
        dependencies = self._get_template_dependencies(tool_type, base_environment)
        
        # Create pyproject.toml
        pyproject_content = self._generate_pyproject_toml(
            tool_name, tool_type, dependencies, base_environment
        )
        (tool_dir / 'pyproject.toml').write_text(pyproject_content)
        
        # Create tool.py template
        tool_content = self._generate_tool_template(tool_name, tool_type)
        (tool_dir / 'tool.py').write_text(tool_content)
        
        # Create __init__.py
        (tool_dir / '__init__.py').write_text('"""MCP Tool Package"""')
        
        # Initialize with uv
        subprocess.run(['uv', 'sync', '--project', str(tool_dir)], 
                      capture_output=True, text=True)
        
        logger.info(f"Created {tool_type} tool: {tool_name}")
        return tool_dir
    
    def _get_template_dependencies(self, tool_type: str, base_env: Optional[str]) -> List[str]:
        """Get template dependencies based on tool type."""
        templates = {
            'simple': [],
            'web': ['requests>=2.31.0', 'beautifulsoup4>=4.12.0'],
            'ai': ['torch>=2.1.0', 'transformers>=4.30.0'],
            'data': ['pandas>=2.0.0', 'numpy>=1.24.0', 'matplotlib>=3.7.0'],
            'cv': ['torch>=2.1.0', 'torchvision>=0.16.0', 'opencv-python>=4.8.0']
        }
        return templates.get(tool_type, [])
    
    def _generate_pyproject_toml(self, name: str, tool_type: str, 
                                dependencies: List[str], base_env: Optional[str]) -> str:
        """Generate pyproject.toml content."""
        return f'''[project]
name = "{name}"
version = "0.1.0"
description = "MCP tool: {name}"
dependencies = {json.dumps(dependencies)}
authors = [
    {{name = "Ethos Collaborative", email = "tools@ethos.dev"}}
]
readme = "README.md"
license = {{text = "MIT"}}

[tool.uv]
dev-dependencies = []

[tool.mcp]
type = "{tool_type}"
complexity = "{self._assess_complexity(dependencies)}"
{f'base_environment = "{base_env}"' if base_env else ''}

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
'''
    
    def _generate_tool_template(self, name: str, tool_type: str) -> str:
        """Generate tool.py template."""
        return f'''"""
{name.title().replace('_', ' ')} Tool

{tool_type.title()} MCP tool implementation.
"""

from typing import Dict


def {name}(params: Dict) -> Dict:
    """
    {name.title().replace('_', ' ')} implementation.
    
    Args:
        params: Tool parameters
        
    Returns:
        Tool execution result
    """
    # TODO: Implement tool logic
    return {{
        "tool": "{name}",
        "type": "{tool_type}",
        "status": "success",
        "message": "Tool executed successfully",
        "params": params
    }}


# Tool metadata for dynamic discovery
TOOLS = [
    {{
        "name": "{name}",
        "description": "{name.title().replace('_', ' ')} tool",
        "function": {name},
        "parameters": {{
            "type": "object",
            "properties": {{}},
            "required": [],
            "additionalProperties": False
        }},
        "returns": {{
            "type": "object",
            "properties": {{
                "tool": {{"type": "string"}},
                "status": {{"type": "string"}},
                "message": {{"type": "string"}}
            }}
        }},
        "tags": ["{tool_type}", "mcp", "uv-managed"],
        "author": "Ethos Collaborative",
        "version": "1.0.0",
        "category": "{tool_type}",
        "examples": [
            {{
                "description": "Basic usage",
                "input": {{}},
                "output": {{
                    "tool": "{name}",
                    "status": "success"
                }}
            }}
        ]
    }}
]
'''


# Usage example
if __name__ == "__main__":
    import importlib.util
    
    tools_dir = Path(__file__).parent
    manager = UVFirstToolManager(tools_dir)
    
    # Create example tools
    print("Creating uv-first tool examples...")
    
    # Simple tool (no dependencies)
    manager.create_tool_template('hello_world', 'simple')
    
    # Web tool (medium dependencies)  
    manager.create_tool_template('web_fetcher', 'web')
    
    # AI tool (heavy dependencies)
    manager.create_tool_template('text_analyzer', 'ai')
    
    print("✓ UV-first tools created!")
    
    # Discover all tools
    tools = manager.discover_tools()
    print(f"\\nDiscovered {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool['name']} ({tool['complexity']}) - {tool['project_version']}")
