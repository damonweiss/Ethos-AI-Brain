#!/usr/bin/env python3
"""
Direct MCP Server

Fast, custom HTTP protocol server for internal/development use.
"""

import json
import asyncio
from aiohttp import web, web_request
from typing import Any, Dict

from .mcp_server_base import MCPServerBase, ServerConfig


class DirectMCPServer(MCPServerBase):
    """Direct protocol MCP server with custom HTTP endpoints."""
    
    @property
    def protocol_name(self) -> str:
        return "direct"
    
    async def start(self) -> None:
        """Start the direct HTTP server."""
        if self.is_running:
            self.logger.warning("Server already running")
            return
        
        try:
            # Create aiohttp application
            app = web.Application()
            
            # Add routes
            app.router.add_get('/health', self._handle_health)
            app.router.add_get('/tools', self._handle_list_tools)
            app.router.add_get('/tools/{tool_name}', self._handle_get_tool_info)
            app.router.add_post('/tools/{tool_name}/execute', self._handle_execute_tool)
            app.router.add_post('/refresh', self._handle_refresh)
            
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
            self.logger.info(f"Direct MCP server started on {self.config.host}:{self.config.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start direct server: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the direct HTTP server."""
        if not self.is_running:
            return
        
        try:
            if self._server:
                await self._server.cleanup()
                self._server = None
            
            self.is_running = False
            self.logger.info("Direct MCP server stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping direct server: {e}")
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a direct request (not used in HTTP server mode)."""
        # This method is for direct programmatic access if needed
        method = request.get("method")
        params = request.get("params", {})
        
        if method == "list_tools":
            return await self.list_tools()
        elif method == "execute_tool":
            tool_name = params.get("tool_name")
            tool_params = params.get("params", {})
            return await self.execute_tool(tool_name, tool_params)
        elif method == "get_tool_info":
            tool_name = params.get("tool_name")
            return await self.get_tool_info(tool_name)
        else:
            return {"error": f"Unknown method: {method}"}
    
    # HTTP Route Handlers
    
    async def _handle_health(self, request: web_request.Request) -> web.Response:
        """Handle health check requests."""
        health = self.health_check()
        return web.json_response(health)
    
    async def _handle_list_tools(self, request: web_request.Request) -> web.Response:
        """Handle list tools requests."""
        tools = await self.list_tools()
        return web.json_response(tools)
    
    async def _handle_get_tool_info(self, request: web_request.Request) -> web.Response:
        """Handle get tool info requests."""
        tool_name = request.match_info['tool_name']
        tool_info = await self.get_tool_info(tool_name)
        
        # Return appropriate HTTP status based on success
        if tool_info.get("success", False):
            return web.json_response(tool_info, status=200)
        else:
            # Check if it's a "tool not found" error
            error_msg = tool_info.get("error", "").lower()
            if "not found" in error_msg or "tool not found" in error_msg:
                return web.json_response(tool_info, status=404)
            else:
                return web.json_response(tool_info, status=400)
    
    async def _handle_execute_tool(self, request: web_request.Request) -> web.Response:
        """Handle tool execution requests."""
        tool_name = request.match_info['tool_name']
        
        try:
            # Parse JSON body
            if request.content_type == 'application/json':
                params = await request.json()
            else:
                params = {}
            
            result = await self.execute_tool(tool_name, params)
            
            # Check if the tool execution was successful and return appropriate HTTP status
            if result.get("success", False):
                return web.json_response(result, status=200)
            else:
                # Check if it's a "tool not found" error
                error_msg = result.get("error", "").lower()
                if "not found" in error_msg or "tool not found" in error_msg:
                    return web.json_response(result, status=404)
                else:
                    return web.json_response(result, status=400)
            
        except json.JSONDecodeError:
            return web.json_response({
                "success": False,
                "error": "Invalid JSON in request body"
            }, status=400)
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    async def _handle_refresh(self, request: web_request.Request) -> web.Response:
        """Handle refresh tools requests."""
        try:
            if self.tool_registry:
                # Trigger tool discovery refresh
                tools = self.tool_registry.discover_all()
                return web.json_response({
                    "success": True,
                    "message": "Tools refreshed",
                    "count": len(tools)
                })
            else:
                return web.json_response({
                    "success": False,
                    "error": "No tool registry available"
                })
        except Exception as e:
            self.logger.error(f"Error refreshing tools: {e}")
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    def _add_cors_headers(self, app: web.Application) -> None:
        """Add CORS headers to all responses."""
        @web.middleware
        async def cors_handler(request: web_request.Request, handler):
            response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response
        
        app.middlewares.append(cors_handler)
