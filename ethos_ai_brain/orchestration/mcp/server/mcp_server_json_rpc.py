#!/usr/bin/env python3
"""
JSON-RPC MCP Server

Standards-compliant MCP server using JSON-RPC 2.0 protocol.
"""

import json
import asyncio
from aiohttp import web, web_request
from typing import Any, Dict, Optional

from .mcp_server_base import MCPServerBase, ServerConfig


class JsonRpcMCPServer(MCPServerBase):
    """JSON-RPC 2.0 compliant MCP server."""
    
    @property
    def protocol_name(self) -> str:
        return "json_rpc"
    
    async def start(self) -> None:
        """Start the JSON-RPC server."""
        if self.is_running:
            self.logger.warning("Server already running")
            return
        
        try:
            # Create aiohttp application
            app = web.Application()
            
            # Add JSON-RPC endpoint
            app.router.add_post('/', self._handle_json_rpc)
            app.router.add_get('/health', self._handle_health)  # Non-standard health endpoint
            
            # Add CORS if enabled
            if self.config.enable_cors:
                self._add_cors_headers(app)
            
            # Create and start server
            runner = web.AppRunner(app)
            await runner.setup()
            
            site = web.TCPSite(runner, self.config.host, self.config.port)
            await site.start()
            
            self._server = runner
            
            self.is_running = True
            self.logger.info(f"JSON-RPC MCP server started on {self.config.host}:{self.config.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start JSON-RPC server: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the JSON-RPC server."""
        if not self.is_running:
            return
        
        try:
            if self._server:
                await self._server.cleanup()
                self._server = None
            
            self.is_running = False
            self.logger.info("JSON-RPC MCP server stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping JSON-RPC server: {e}")
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a JSON-RPC request."""
        # Validate JSON-RPC format
        if not self._is_valid_json_rpc_request(request):
            return self._create_error_response(
                None, -32600, "Invalid Request", "Invalid JSON-RPC 2.0 request"
            )
        
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            # Route to appropriate handler
            if method == "tools/list":
                result = await self.list_tools()
            elif method == "tools/call":
                tool_name = params.get("name")
                tool_params = params.get("arguments", {})
                result = await self.execute_tool(tool_name, tool_params)
            elif method == "tools/info":
                tool_name = params.get("name")
                result = await self.get_tool_info(tool_name)
            else:
                return self._create_error_response(
                    request_id, -32601, "Method not found", f"Unknown method: {method}"
                )
            
            return self._create_success_response(request_id, result)
            
        except Exception as e:
            self.logger.error(f"Error handling JSON-RPC request: {e}")
            return self._create_error_response(
                request_id, -32603, "Internal error", str(e)
            )
    
    # HTTP Route Handlers
    
    async def _handle_json_rpc(self, request: web_request.Request) -> web.Response:
        """Handle JSON-RPC requests."""
        try:
            # Parse JSON body
            if request.content_type != 'application/json':
                return web.json_response(
                    self._create_error_response(
                        None, -32600, "Invalid Request", "Content-Type must be application/json"
                    ),
                    status=400
                )
            
            json_request = await request.json()
            
            # Handle batch requests
            if isinstance(json_request, list):
                responses = []
                for req in json_request:
                    response = await self.handle_request(req)
                    if response:  # Don't include responses for notifications
                        responses.append(response)
                return web.json_response(responses)
            else:
                response = await self.handle_request(json_request)
                return web.json_response(response)
                
        except json.JSONDecodeError:
            return web.json_response(
                self._create_error_response(
                    None, -32700, "Parse error", "Invalid JSON"
                ),
                status=400
            )
        except Exception as e:
            self.logger.error(f"Error handling JSON-RPC request: {e}")
            return web.json_response(
                self._create_error_response(
                    None, -32603, "Internal error", str(e)
                ),
                status=500
            )
    
    async def _handle_health(self, request: web_request.Request) -> web.Response:
        """Handle health check requests (non-standard endpoint)."""
        health = self.health_check()
        return web.json_response(health)
    
    # JSON-RPC Utilities
    
    def _is_valid_json_rpc_request(self, request: Dict[str, Any]) -> bool:
        """Validate JSON-RPC 2.0 request format."""
        return (
            isinstance(request, dict) and
            request.get("jsonrpc") == "2.0" and
            "method" in request and
            isinstance(request["method"], str)
        )
    
    def _create_success_response(self, request_id: Optional[Any], result: Any) -> Dict[str, Any]:
        """Create a JSON-RPC success response."""
        response = {
            "jsonrpc": "2.0",
            "result": result
        }
        
        if request_id is not None:
            response["id"] = request_id
        
        return response
    
    def _create_error_response(self, request_id: Optional[Any], code: int, 
                             message: str, data: Optional[Any] = None) -> Dict[str, Any]:
        """Create a JSON-RPC error response."""
        error = {
            "code": code,
            "message": message
        }
        
        if data is not None:
            error["data"] = data
        
        response = {
            "jsonrpc": "2.0",
            "error": error
        }
        
        if request_id is not None:
            response["id"] = request_id
        
        return response
    
    def _add_cors_headers(self, app: web.Application) -> None:
        """Add CORS headers to all responses."""
        @web.middleware
        async def cors_handler(request: web_request.Request, handler):
            response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response
        
        app.middlewares.append(cors_handler)
