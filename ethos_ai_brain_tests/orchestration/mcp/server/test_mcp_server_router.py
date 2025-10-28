#!/usr/bin/env python3
"""
MCP Server Router Tests

Test the new router-based MCP server architecture.
Real integration tests with no mocks.
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
    MCPServerRouter
)
from ethos_ai_brain.orchestration.mcp.tool_management.mcp_tool_manager import MCPToolManager


@pytest.mark.asyncio
class TestMCPServerRouter:
    """Test the new router-based server architecture."""
    
    @pytest_asyncio.fixture
    async def tool_manager(self):
        """Create a test tool manager."""
        tools_dir = Path(__file__).parent.parent.parent.parent.parent / "orchestration" / "mcp_tools"
        manager = MCPToolManager.get_instance(tools_dir)
        yield manager
    
    async def test_create_direct_server_router(self, tool_manager):
        """Test creating and starting a direct server router."""
        server = create_development_server(
            port=8070,
            tool_registry=tool_manager
        )
        
        assert isinstance(server, MCPServerRouter)
        assert server.config.protocol == "direct"
        assert server.config.port == 8070
        
        # Start server
        await server.start()
        assert server.is_running
        
        # Test health check
        health = server.health_check()
        assert health["status"] == "healthy"
        assert health["protocol"] == "direct"
        
        # Stop server
        await server.stop()
        assert not server.is_running
    
    async def test_create_json_rpc_server_router(self, tool_manager):
        """Test creating and starting a JSON-RPC server router."""
        server = create_production_server(
            port=8071,
            tool_registry=tool_manager
        )
        
        assert isinstance(server, MCPServerRouter)
        assert server.config.protocol == "json_rpc"
        assert server.config.port == 8071
        
        # Start server
        await server.start()
        assert server.is_running
        
        # Test health check
        health = server.health_check()
        assert health["status"] == "healthy"
        assert health["protocol"] == "json_rpc"
        
        # Stop server
        await server.stop()
        assert not server.is_running
    
    async def test_direct_server_http_endpoints(self, tool_manager):
        """Test direct server HTTP endpoints."""
        server = create_development_server(
            port=8072,
            tool_registry=tool_manager
        )
        
        await server.start()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test health endpoint
                async with session.get("http://localhost:8072/health") as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data["status"] == "healthy"
                    assert data["protocol"] == "direct"
                    print(f"Health check result: {data}")
                
                # Test tools endpoint
                async with session.get("http://localhost:8072/tools") as response:
                    assert response.status == 200
                    data = await response.json()
                    assert "tools" in data
                    assert "count" in data
                    print(f"Tools count: {data['count']}")
                    
        finally:
            await server.stop()
    
    async def test_json_rpc_server_endpoints(self, tool_manager):
        """Test JSON-RPC server endpoints."""
        server = create_production_server(
            port=8073,
            tool_registry=tool_manager
        )
        
        await server.start()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test health endpoint (non-standard but useful)
                async with session.get("http://localhost:8073/health") as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data["status"] == "healthy"
                    assert data["protocol"] == "json_rpc"
                    print(f"JSON-RPC health check: {data}")
                    
        finally:
            await server.stop()
    
    async def test_server_factory_functions(self, tool_manager):
        """Test all factory functions work correctly."""
        # Test generic create_server
        direct_server = create_server(
            protocol="direct",
            port=8074,
            tool_registry=tool_manager
        )
        assert direct_server.config.protocol == "direct"
        
        jsonrpc_server = create_server(
            protocol="json_rpc", 
            port=8075,
            tool_registry=tool_manager
        )
        assert jsonrpc_server.config.protocol == "json_rpc"
        
        # Test convenience functions
        dev_server = create_development_server(port=8076, tool_registry=tool_manager)
        assert dev_server.config.protocol == "direct"
        assert dev_server.config.enable_cors == True
        
        prod_server = create_production_server(port=8077, tool_registry=tool_manager)
        assert prod_server.config.protocol == "json_rpc"
        assert prod_server.config.host == "0.0.0.0"  # External access
        
        print("[SUCCESS] All factory functions work correctly")
    
    async def test_router_tool_operations(self, tool_manager):
        """Test router tool operations."""
        server = create_development_server(
            port=8078,
            tool_registry=tool_manager
        )
        
        await server.start()
        
        try:
            # Test list_tools
            tools_result = await server.list_tools()
            assert "tools" in tools_result
            print(f"Router tools result: {tools_result}")
            
            # Test get_tool_info for non-existent tool
            tool_info = await server.get_tool_info("nonexistent_tool")
            assert "success" in tool_info
            print(f"Tool info result: {tool_info}")
            
        finally:
            await server.stop()
    
    async def test_multiple_servers_different_ports(self, tool_manager):
        """Test running multiple server routers on different ports."""
        servers = []
        
        try:
            # Create multiple servers
            for i, protocol in enumerate(["direct", "json_rpc", "direct"]):
                port = 8080 + i
                server = create_server(
                    protocol=protocol,
                    port=port,
                    tool_registry=tool_manager
                )
                await server.start()
                servers.append(server)
                assert server.is_running
                print(f"Started {protocol} server on port {port}")
            
            # All should be running
            for server in servers:
                health = server.health_check()
                assert health["status"] == "healthy"
            
            print("[SUCCESS] Multiple servers running simultaneously")
            
        finally:
            # Clean up all servers
            for server in servers:
                await server.stop()
                assert not server.is_running


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
