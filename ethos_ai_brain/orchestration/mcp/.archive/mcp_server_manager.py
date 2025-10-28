#!/usr/bin/env python3
"""
MCP Server Manager

Manages multiple MCP servers with different protocols for AI Engine integration.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from pathlib import Path

from .mcp_server_base import MCPServerBase, ServerConfig
from .mcp_server_direct import DirectMCPServer
from .mcp_server_json_rpc import JsonRpcMCPServer

logger = logging.getLogger(__name__)


class MCPServerManager:
    """Manages multiple MCP servers for different protocols and use cases."""
    
    def __init__(self, tool_registry=None):
        self.tool_registry = tool_registry
        self.servers: Dict[str, MCPServerBase] = {}
        self.is_running = False
    
    async def start_server(self, name: str, config: ServerConfig) -> bool:
        """Start a specific MCP server."""
        if name in self.servers:
            logger.warning(f"Server {name} already exists")
            return False
        
        try:
            # Create server based on protocol
            if config.protocol == "direct":
                server = DirectMCPServer(config, self.tool_registry)
            elif config.protocol == "json_rpc":
                server = JsonRpcMCPServer(config, self.tool_registry)
            else:
                logger.error(f"Unknown protocol: {config.protocol}")
                return False
            
            # Start the server
            await server.start()
            self.servers[name] = server
            
            logger.info(f"Started MCP server '{name}' ({config.protocol}) on {config.host}:{config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start server {name}: {e}")
            return False
    
    async def stop_server(self, name: str) -> bool:
        """Stop a specific MCP server."""
        if name not in self.servers:
            logger.warning(f"Server {name} not found")
            return False
        
        try:
            server = self.servers[name]
            await server.stop()
            del self.servers[name]
            
            logger.info(f"Stopped MCP server '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop server {name}: {e}")
            return False
    
    async def start_all(self) -> None:
        """Start all configured servers."""
        if self.is_running:
            logger.warning("Server manager already running")
            return
        
        self.is_running = True
        logger.info("MCP Server Manager started")
    
    async def stop_all(self) -> None:
        """Stop all running servers."""
        if not self.is_running:
            return
        
        # Stop all servers
        stop_tasks = []
        for name in list(self.servers.keys()):
            stop_tasks.append(self.stop_server(name))
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        self.is_running = False
        logger.info("MCP Server Manager stopped")
    
    def get_server(self, name: str) -> Optional[MCPServerBase]:
        """Get a server by name."""
        return self.servers.get(name)
    
    def list_servers(self) -> Dict[str, Dict]:
        """List all servers and their status."""
        return {
            name: {
                "protocol": server.protocol_name,
                "host": server.config.host,
                "port": server.config.port,
                "running": server.is_running,
                "health": server.health_check()
            }
            for name, server in self.servers.items()
        }
    
    async def health_check_all(self) -> Dict[str, Dict]:
        """Health check all servers."""
        health_checks = {}
        
        for name, server in self.servers.items():
            try:
                health_checks[name] = server.health_check()
            except Exception as e:
                health_checks[name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return health_checks
    
    # Convenience methods for common server configurations
    
    async def start_development_server(self, port: int = 8051, 
                                     tools_dir: Optional[Path] = None) -> bool:
        """Start a development server (direct protocol, fast iteration)."""
        config = ServerConfig(
            host="localhost",
            port=port,
            protocol="direct",
            tools_directory=tools_dir,
            enable_cors=True
        )
        return await self.start_server("development", config)
    
    async def start_production_server(self, port: int = 8052,
                                    tools_dir: Optional[Path] = None) -> bool:
        """Start a production server (JSON-RPC protocol, standards compliant)."""
        config = ServerConfig(
            host="0.0.0.0",  # Accept external connections
            port=port,
            protocol="json_rpc",
            tools_directory=tools_dir,
            enable_cors=False,  # More restrictive for production
            max_connections=1000
        )
        return await self.start_server("production", config)
    
    async def start_internal_server(self, port: int = 8053,
                                  tools_dir: Optional[Path] = None) -> bool:
        """Start an internal server (direct protocol, high performance)."""
        config = ServerConfig(
            host="localhost",
            port=port,
            protocol="direct",
            tools_directory=tools_dir,
            enable_cors=False,
            max_connections=500
        )
        return await self.start_server("internal", config)
    
    def __str__(self) -> str:
        return f"MCPServerManager(servers={len(self.servers)}, running={self.is_running})"
