#!/usr/bin/env python3
"""
Basic MCP Server Tests

Test the multi-headed MCP server functionality.
"""

import asyncio
import pytest
import pytest_asyncio
import aiohttp
import json
from pathlib import Path

from ethos_ai_brain.orchestration.mcp.server import (
    create_server,
    create_development_server,
    create_production_server,
    ServerConfig,
    DirectMCPServer,
    JsonRpcMCPServer,
    MCPServerRouter
)
from ethos_ai_brain.orchestration.mcp.tool_management.mcp_tool_manager import MCPToolManager


@pytest.mark.asyncio
class TestMCPServerBasic:
    """Basic tests for MCP server functionality."""
    
    @pytest_asyncio.fixture
    async def tool_manager(self):
        """Create a test tool manager."""
        tools_dir = Path(__file__).parent.parent.parent.parent / "orchestration" / "mcp_tools"
        manager = MCPToolManager.get_instance(tools_dir)
        yield manager
    
    @pytest_asyncio.fixture
    async def server_manager(self, tool_manager):
        """Create a test server manager."""
        manager = MCPServerManager(tool_registry=tool_manager)
        yield manager
        # Cleanup
        await manager.stop_all()
    
    async def test_direct_server_startup(self, server_manager):
        """Test direct server can start and stop."""
        # Start development server
        success = await server_manager.start_development_server(port=8060)
        assert success, "Failed to start development server"
        
        # Check server is running
        servers = server_manager.list_servers()
        assert "development" in servers
        assert servers["development"]["running"] is True
        assert servers["development"]["protocol"] == "direct"
        
        # Stop server
        success = await server_manager.stop_server("development")
        assert success, "Failed to stop development server"
    
    async def test_json_rpc_server_startup(self, server_manager):
        """Test JSON-RPC server can start and stop."""
        # Start production server
        success = await server_manager.start_production_server(port=8061)
        assert success, "Failed to start production server"
        
        # Check server is running
        servers = server_manager.list_servers()
        assert "production" in servers
        assert servers["production"]["running"] is True
        assert servers["production"]["protocol"] == "json_rpc"
        
        # Stop server
        success = await server_manager.stop_server("production")
        assert success, "Failed to stop production server"
    
    async def test_direct_server_health_endpoint(self, server_manager):
        """Test direct server health endpoint."""
        # Start server
        await server_manager.start_development_server(port=8062)
        
        try:
            # Test health endpoint
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8062/health") as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data["status"] == "healthy"
                    assert data["protocol"] == "direct"
        finally:
            await server_manager.stop_server("development")
    
    async def test_direct_server_list_tools(self, server_manager):
        """Test direct server list tools endpoint."""
        # Start server
        await server_manager.start_development_server(port=8063)
        
        try:
            # Test list tools endpoint
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8063/tools") as response:
                    assert response.status == 200
                    data = await response.json()
                    assert "tools" in data
                    assert "count" in data
                    assert isinstance(data["tools"], list)
        finally:
            await server_manager.stop_server("development")
    
    async def test_json_rpc_server_health_endpoint(self, server_manager):
        """Test JSON-RPC server health endpoint."""
        # Start server
        await server_manager.start_production_server(port=8064)
        
        try:
            # Test health endpoint (non-standard but useful)
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8064/health") as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data["status"] == "healthy"
                    assert data["protocol"] == "json_rpc"
        finally:
            await server_manager.stop_server("production")
    
    async def test_json_rpc_server_list_tools(self, server_manager):
        """Test JSON-RPC server list tools via JSON-RPC."""
        # Start server
        await server_manager.start_production_server(port=8065)
        
        try:
            # Test JSON-RPC list tools
            json_rpc_request = {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8065/",
                    json=json_rpc_request,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data["jsonrpc"] == "2.0"
                    assert "result" in data
                    assert "tools" in data["result"]
        finally:
            await server_manager.stop_server("production")
    
    async def test_multiple_servers(self, server_manager):
        """Test running multiple servers simultaneously."""
        # Start both servers
        dev_success = await server_manager.start_development_server(port=8066)
        prod_success = await server_manager.start_production_server(port=8067)
        
        assert dev_success and prod_success, "Failed to start both servers"
        
        try:
            # Check both are running
            servers = server_manager.list_servers()
            assert len(servers) == 2
            assert "development" in servers
            assert "production" in servers
            
            # Test both health endpoints
            async with aiohttp.ClientSession() as session:
                # Direct server
                async with session.get("http://localhost:8066/health") as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data["protocol"] == "direct"
                
                # JSON-RPC server
                async with session.get("http://localhost:8067/health") as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data["protocol"] == "json_rpc"
        
        finally:
            await server_manager.stop_all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
