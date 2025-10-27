#!/usr/bin/env python3
"""
Ethos-MCP - Universal MCP Server
A standalone MCP server that provides tools and capabilities with Universal API Wrapper support.
This server runs independently and can integrate with various backends via HTTP.

Author: Damon Weiss
Created: 2025-01-28
License: MIT
"""

import json
import logging
import os
import sys
import threading
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from flask import Flask, jsonify, request
from flask_cors import CORS

# Import MCP tools from organized modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from mcp_tools import AVAILABLE_TOOLS, discover_tools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EthosMCPServer:
    """Universal MCP Server for Ethos ecosystem"""
    
    def __init__(self, name: str = "ethos-mcp-server", config_dir: Optional[Path] = None):
        self.name = name
        self.app = Flask(__name__)
        CORS(self.app)
        self.tools_registry = {}
        self.config_dir = config_dir or Path(__file__).parent.parent / "config" / "api_configs"
        self.flask_backend_url = "http://localhost:8050/api/mcp"
        self.registration_thread = None
        self.setup_routes()
        self.register_default_tools()
        # TODO: Load Universal API Wrapper tools from YAML configs
        # self.load_api_wrapper_tools()
    
    def setup_routes(self):
        """Setup HTTP routes for MCP communication"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'server': self.name,
                'timestamp': datetime.now().isoformat(),
                'tools_count': len(self.tools_registry),
                'config_dir': str(self.config_dir)
            })
        
        @self.app.route('/tools', methods=['GET'])
        def list_tools():
            """List available MCP tools"""
            return jsonify({
                'tools': list(self.tools_registry.keys()),
                'count': len(self.tools_registry),
                'server': self.name
            })
        
        @self.app.route('/tools/<tool_name>', methods=['GET'])
        def get_tool_info(tool_name: str):
            """Get information about a specific tool"""
            if tool_name not in self.tools_registry:
                return jsonify({'error': f'Tool {tool_name} not found'}), 404
            
            tool_info = self.tools_registry[tool_name].copy()
            # Remove function object for JSON serialization
            if 'function' in tool_info:
                del tool_info['function']
            
            return jsonify({
                'tool': tool_info,
                'available': True,
                'server': self.name
            })
        
        @self.app.route('/tools/<tool_name>/execute', methods=['POST'])
        def execute_tool(tool_name: str):
            """Execute a specific MCP tool using UV if available"""
            if tool_name not in self.tools_registry:
                return jsonify({'error': f'Tool {tool_name} not found'}), 404
            
            try:
                data = request.get_json() or {}
                tool_info = self.tools_registry[tool_name]
                
                # Check if tool is UV-managed
                is_uv_managed = tool_info.get('uv_managed', False)
                logger.info(f"Executing tool {tool_name}: uv_managed={is_uv_managed}")
                
                if is_uv_managed:
                    logger.info(f"Using UV execution for {tool_name}")
                    result = self._execute_uv_tool(tool_name, tool_info, data)
                else:
                    # Direct execution for legacy tools
                    logger.info(f"Using direct execution for {tool_name}")
                    tool_func = tool_info['function']
                    result = tool_func(data)
                
                return jsonify({
                    'result': result,
                    'tool': tool_name,
                    'server': self.name,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'tool': tool_name,
                    'server': self.name
                }), 500
        
        @self.app.route('/refresh', methods=['POST'])
        def handle_refresh():
            """Handle refresh requests from Flask backend"""
            try:
                # Re-register all tools with Flask backend
                self.register_with_flask_backend()
                return jsonify({
                    'message': 'Tools re-registered successfully',
                    'tools_count': len(self.tools_registry),
                    'server': self.name,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'error': f'Failed to refresh tools: {str(e)}',
                    'server': self.name
                }), 500
    
    def register_tool(self, name: str, description: str, function: callable, parameters: Dict = None):
        """Register a new MCP tool"""
        self.tools_registry[name] = {
            'name': name,
            'description': description,
            'function': function,
            'parameters': parameters or {},
            'registered_at': datetime.now().isoformat()
        }
        logger.info(f"Registered MCP tool: {name}")
    
    def register_default_tools(self):
        """Dynamically register all discovered tools."""
        logger.info("Registering tools dynamically...")
        
        for tool_info in AVAILABLE_TOOLS:
            tool_name = tool_info['name']
            tool_function = tool_info['function']
            
            # Create wrapper function to inject server context
            def create_wrapper(func, name):
                def wrapper(params: Dict) -> Dict:
                    result = func(params)
                    
                    # Inject server context based on tool type
                    if name == 'get_project_info':
                        result['tools_available'] = len(self.tools_registry)
                        result['config_dir'] = str(self.config_dir)
                    elif name in ['echo_message', 'get_system_time']:
                        if 'processed_by' in result:
                            result['processed_by'] = self.name
                        if 'server' in result:
                            result['server'] = self.name
                    elif name == 'list_api_configs':
                        # Special handling for list_api_configs which needs config_dir
                        return func(params, self.config_dir)
                    
                    return result
                return wrapper
            
            # Register the tool preserving UV metadata
            enhanced_tool_info = tool_info.copy()
            enhanced_tool_info.update({
                'function': create_wrapper(tool_function, tool_name),
                'registered_at': datetime.now().isoformat()
            })
            self.tools_registry[tool_name] = enhanced_tool_info
            logger.info(f"Registered MCP tool: {tool_name}")
            
        logger.info(f"Registered {len(AVAILABLE_TOOLS)} tools dynamically")
        
        # Debug: Check if tools are UV-managed
        for tool_name, tool_info in self.tools_registry.items():
            is_uv = tool_info.get('uv_managed', False)
            logger.info(f"Tool {tool_name} in registry: uv_managed={is_uv}")
    
    def _execute_uv_tool(self, tool_name: str, tool_info: dict, params: dict):
        """Execute a UV-managed tool in its isolated environment."""
        import subprocess
        import json
        from pathlib import Path
        
        tool_dir = Path(tool_info.get('tool_dir', ''))
        if not tool_dir.exists():
            return {"error": f"Tool directory not found: {tool_dir}"}
        
        # Create execution script that suppresses tool stdout
        exec_script = f"""
import sys
import json
import io
import contextlib
from pathlib import Path

# Import the tool module
sys.path.insert(0, str(Path.cwd()))
from tool import TOOLS

params = json.loads(r'''{json.dumps(params)}''')

# Find and execute the requested tool
for tool_info in TOOLS:
    if tool_info['name'] == '{tool_name}':
        try:
            # Capture and suppress stdout from tool execution
            captured_output = io.StringIO()
            with contextlib.redirect_stdout(captured_output):
                result = tool_info['function'](params)
            
            # Only output the JSON result
            print(json.dumps(result))
            sys.exit(0)
        except Exception as e:
            print(json.dumps({{"error": f"Tool execution failed: {{str(e)}}"}}))
            sys.exit(1)

print(json.dumps({{"error": "Tool {tool_name} not found"}}))
sys.exit(1)
"""
        
        # Write temporary execution script
        temp_script = tool_dir / '_temp_exec.py'
        try:
            temp_script.write_text(exec_script)
            
            # Execute using uv run
            result = subprocess.run([
                'uv', 'run', 
                '--project', str(tool_dir),
                'python', '_temp_exec.py'
            ], 
            capture_output=True, 
            text=True,
            cwd=tool_dir,
            timeout=30
            )
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {
                    "error": f"UV execution failed: {result.stderr}",
                    "stdout": result.stdout,
                    "returncode": result.returncode
                }
                
        finally:
            # Cleanup temp script
            if temp_script.exists():
                temp_script.unlink()
        
        return {"error": "UV tool execution failed"}
    
    def register_with_flask_backend(self):
        """Register all MCP tools with the Flask backend"""
        try:
            # Prepare tools data in the format Flask expects
            tools_data = []
            for tool_name, tool_info in self.tools_registry.items():
                tools_data.append({
                    'name': tool_name,
                    'description': tool_info.get('description', ''),
                    'parameters': tool_info.get('parameters', {}),
                    'endpoint': f'/tools/{tool_name}'
                })
            
            payload = {'tools': tools_data}
            
            # Send registration request to Flask backend
            registration_url = f"{self.flask_backend_url}/register"
            logger.info(f"Attempting to register tools at: {registration_url}")
            
            response = requests.post(
                registration_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully registered {len(tools_data)} tools with Flask backend")
                return True
            else:
                logger.error(f"Failed to register tools with Flask backend: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.warning("Could not connect to Flask backend for tool registration")
            return False
        except Exception as e:
            logger.error(f"Error registering tools with Flask backend: {e}")
            return False
    
    def start_registration_attempts(self):
        """Start background thread to attempt registration with Flask backend"""
        def registration_worker():
            max_attempts = 10
            attempt = 0
            
            while attempt < max_attempts:
                if self.register_with_flask_backend():
                    logger.info("Successfully registered with Flask backend")
                    break
                
                attempt += 1
                logger.info(f"Registration attempt {attempt}/{max_attempts} failed, retrying in 5 seconds...")
                time.sleep(5)
            
            if attempt >= max_attempts:
                logger.warning("Failed to register with Flask backend after maximum attempts")
        
        self.registration_thread = threading.Thread(target=registration_worker, daemon=True)
        self.registration_thread.start()
    
    def run(self, host: str = 'localhost', port: int = 8051, debug: bool = True):
        """Run the MCP server"""
        logger.info(f"Starting {self.name} on {host}:{port}")
        logger.info(f"Registered {len(self.tools_registry)} MCP tools")
        logger.info(f"Config directory: {self.config_dir}")
        
        self.app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )

def main():
    """Main entry point for the MCP server"""
    server = EthosMCPServer()
    
    print("Starting Ethos-MCP Server")
    print(f"Server URL: http://localhost:8051")
    print(f"Tools Available: {len(server.tools_registry)}")
    print(f"Health Check: http://localhost:8051/health")
    print(f"Tools List: http://localhost:8051/tools")
    print(f"Config Directory: {server.config_dir}")
    print(f"Flask Backend: {server.flask_backend_url}")
    
    # Start background registration attempts
    server.start_registration_attempts()
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nMCP Server stopped by user")
    except Exception as e:
        print(f"MCP Server error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
