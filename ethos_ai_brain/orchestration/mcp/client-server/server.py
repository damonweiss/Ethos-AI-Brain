#!/usr/bin/env python3
"""
Unified MCP Meta-Reasoning Server

Supports both JSON-RPC (standard MCP) and HTTP (OpenAI function calling) protocols
with shared meta-reasoning engine and no code duplication.
"""

import sys
import json
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Setup
BASE_DIR = Path(__file__).parent.parent.parent.parent
load_dotenv(BASE_DIR / '.env')
sys.path.insert(0, str(BASE_DIR / 'src'))

from mcp_tools.uv_first_discovery import UVFirstToolManager
from mcp_tools.namespace_registry import namespace_registry, parse_namespaced_tool_name


class UnifiedMCPServer:
    """Unified server supporting both JSON-RPC and HTTP protocols."""
    
    def __init__(self):
        # Shared meta-reasoner instance
        self.meta_reasoner = None
        self.flask_app = None
        self.flask_thread = None
        self._initialize_meta_reasoner()
        self._setup_flask()
    
    def _initialize_meta_reasoner(self):
        """Initialize shared meta-reasoning engine."""
        # Import here to avoid circular imports
        sys.path.insert(0, str(BASE_DIR / 'mcp_tests' / 'tools'))
        from test_meta_reasoning_mcp import MCPMetaReasoner
        
        self.meta_reasoner = MCPMetaReasoner()
        # Debug: print discovered tools count (only in HTTP mode)
        if '--mode' not in sys.argv or 'jsonrpc' not in sys.argv:
            print(f"[DEBUG] Server initialized with {len(self.meta_reasoner.tools)} tools", file=sys.stderr)
    
    def _setup_flask(self):
        """Setup Flask HTTP endpoints for OpenAI function calling mode."""
        self.flask_app = Flask(__name__)
        
        @self.flask_app.route('/health', methods=['GET'])
        def health():
            return jsonify({"status": "healthy", "mode": "unified", "protocols": ["json-rpc", "http"]})
        
        @self.flask_app.route('/tools', methods=['GET'])
        def list_tools():
            """HTTP endpoint - returns tools for OpenAI function calling."""
            tools = list(self.meta_reasoner.tools.keys())
            return jsonify({"tools": tools, "mode": "discovery"})
        
        @self.flask_app.route('/tools/<tool_name>/execute', methods=['POST'])
        def execute_tool(tool_name: str):
            """HTTP endpoint - direct tool execution."""
            try:
                params = request.get_json() or {}
                result = self.meta_reasoner.tool_manager.execute_tool(tool_name, params)
                return jsonify({
                    "result": result,
                    "tool": tool_name,
                    "mode": "direct_execution"
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.flask_app.route('/meta_reason', methods=['POST'])
        async def meta_reason_http():
            """HTTP endpoint - full meta-reasoning pipeline."""
            try:
                data = request.get_json()
                user_request = data.get('request', '')
                
                # Run async meta-reasoning in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.meta_reasoner.meta_reason(user_request))
                loop.close()
                
                return jsonify({
                    "result": result,
                    "mode": "meta_reasoning"
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    
    async def handle_jsonrpc_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle JSON-RPC requests (standard MCP protocol)."""
        method = request.get("method")
        params = request.get("params", {})
        req_id = request.get("id")
        
        try:
            if method == "initialize":
                # MCP initialization
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "meta_reasoning": {}
                    },
                    "serverInfo": {
                        "name": "ethos-mcp-unified",
                        "version": "1.0.0"
                    }
                }
            
            elif method == "tools/list":
                # Standard MCP tool listing with namespace support
                tools_list = []
                for name, tool_info in self.meta_reasoner.tools.items():
                    # Include namespace information in tool listing
                    namespace, original_name = parse_namespaced_tool_name(name)
                    
                    tool_entry = {
                        "name": name,  # Full namespaced name
                        "description": tool_info.get("description", f"Execute {original_name} tool"),
                        "inputSchema": {
                            "type": "object",
                            "properties": tool_info.get("parameters", {}).get("properties", {}),
                            "required": tool_info.get("parameters", {}).get("required", [])
                        }
                    }
                    
                    # Add namespace metadata
                    if namespace != "default":
                        tool_entry["namespace"] = namespace
                        tool_entry["category"] = tool_info.get("category", "general")
                    
                    tools_list.append(tool_entry)
                
                result = {"tools": tools_list}
            
            elif method == "tools/call":
                # Standard MCP tool execution
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                if tool_name not in self.meta_reasoner.tools:
                    # Debug: show available tools
                    available = list(self.meta_reasoner.tools.keys())
                    print(f"[ERROR] Tool '{tool_name}' not found", file=sys.stderr)
                    print(f"[ERROR] Available tools: {available}", file=sys.stderr)
                    print(f"[ERROR] Tool count: {len(available)}", file=sys.stderr)
                    raise ValueError(f"Tool {tool_name} not found. Available: {available[:3]}...")
                
                tool_result = self.meta_reasoner.tool_manager.execute_tool(tool_name, arguments)
                print(f"[DEBUG] Tool execution result: {tool_result}", file=sys.stderr)
                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(tool_result, indent=2)
                        }
                    ]
                }
            
            elif method == "namespaces/list":
                # Custom endpoint to list namespace capabilities
                capabilities = namespace_registry.get_tool_capabilities()
                result = {
                    "namespaces": capabilities["namespaces"],
                    "categories": capabilities["categories"],
                    "total_tools": capabilities["total_tools"],
                    "available_namespaces": namespace_registry.list_namespaces()
                }
            
            elif method == "namespaces/find_tools":
                # Custom endpoint to find tools for a task
                task_description = params.get("task", "")
                print(f"[DEBUG] find_tools_for_task called with: {task_description}", file=sys.stderr)
                try:
                    recommendations = namespace_registry.find_tools_for_task(task_description)
                    print(f"[DEBUG] find_tools_for_task succeeded", file=sys.stderr)
                    result = {
                        "task": task_description,
                        "recommendations": recommendations
                    }
                except Exception as e:
                    print(f"[ERROR] find_tools_for_task failed: {e}", file=sys.stderr)
                    import traceback
                    print(f"[ERROR] Traceback: {traceback.format_exc()}", file=sys.stderr)
                    result = {
                        "task": task_description,
                        "recommendations": {"meta_reasoning": [], "domain_tools": [], "supporting_tools": []},
                        "error": str(e)
                    }
            
            elif method == "meta_reason":
                # Custom meta-reasoning endpoint for JSON-RPC
                user_request = params.get("request", "")
                meta_result = await self.meta_reasoner.meta_reason(user_request)
                
                result = {
                    "content": [
                        {
                            "type": "text", 
                            "text": meta_result["final_response"]
                        }
                    ],
                    "metadata": {
                        "complexity": meta_result["complexity_analysis"]["complexity"],
                        "quality_score": meta_result["validation_result"]["overall_quality"],
                        "tools_used": meta_result["orchestration_result"]["tools_executed"]
                    }
                }
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result
            }
        
        except Exception as e:
            # Debug: print full error details
            import traceback
            print(f"[ERROR] Exception in handle_jsonrpc_request: {e}", file=sys.stderr)
            print(f"[ERROR] Traceback: {traceback.format_exc()}", file=sys.stderr)
            return {
                "jsonrpc": "2.0", 
                "id": req_id,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }
    
    async def run_jsonrpc_server(self):
        """Run JSON-RPC server on stdin/stdout."""
        # Silent in JSON-RPC mode to avoid stdout pollution
        
        try:
            while True:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line or not line.strip():
                    break
                
                try:
                    request = json.loads(line.strip())
                    response = await self.handle_jsonrpc_request(request)
                    print(json.dumps(response), flush=True)
                except json.JSONDecodeError:
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": "Parse error"
                        }
                    }
                    print(json.dumps(error_response), flush=True)
        
        except KeyboardInterrupt:
            pass  # Silent exit in JSON-RPC mode
    
    def run_flask_server(self, host='localhost', port=8006):
        """Run Flask HTTP server in background thread."""
        def flask_runner():
            print(f"[HTTP] Server starting on {host}:{port}", file=sys.stderr)
            self.flask_app.run(host=host, port=port, debug=False, use_reloader=False)
        
        self.flask_thread = threading.Thread(target=flask_runner, daemon=True)
        self.flask_thread.start()
        print(f"[HTTP] Server thread started", file=sys.stderr)
    
    async def run_unified(self, mode='auto', http_port=8006):
        """
        Run unified server with automatic protocol detection.
        
        Args:
            mode: 'auto', 'jsonrpc', 'http', or 'both'
            http_port: Port for HTTP server
        """
        if mode != 'jsonrpc':
            print(f"[UNIFIED] Starting server in {mode} mode", file=sys.stderr)
        
        if mode == 'auto':
            # Auto-detect: if stdin is a TTY, run HTTP; otherwise JSON-RPC
            if sys.stdin.isatty():
                mode = 'http'
                print("[UNIFIED] TTY detected, running HTTP mode", file=sys.stderr)
            else:
                mode = 'jsonrpc'
        
        if mode in ['http', 'both']:
            self.run_flask_server(port=http_port)
        
        if mode in ['jsonrpc', 'both']:
            await self.run_jsonrpc_server()
        elif mode == 'http':
            # Keep HTTP server alive
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("[HTTP] Server stopped", file=sys.stderr)


async def main():
    """Main entry point with CLI argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified MCP Meta-Reasoning Server')
    parser.add_argument('--mode', choices=['auto', 'jsonrpc', 'http', 'both'], 
                       default='auto', help='Server mode')
    parser.add_argument('--port', type=int, default=8006, 
                       help='HTTP server port')
    
    args = parser.parse_args()
    
    server = UnifiedMCPServer()
    await server.run_unified(mode=args.mode, http_port=args.port)


if __name__ == "__main__":
    asyncio.run(main())
