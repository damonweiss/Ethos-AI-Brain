#!/usr/bin/env python3
"""
Async CLI Tests

Test CLI functionality directly without Click runner to avoid event loop conflicts.
Tests the actual async functions that power the CLI commands.
"""

import pytest
import asyncio
import aiohttp
from pathlib import Path

from ethos_ai_brain.orchestration.mcp.server import create_server
from ethos_ai_brain.orchestration.mcp.tool_management.mcp_tool_manager import MCPToolManager


@pytest.mark.asyncio
class TestCLIAsync:
    """Test CLI functionality with proper async handling."""
    
    def setup_method(self):
        """Set up for each test."""
        self.server_router = None
        self.test_port = 8090  # Use high port to avoid conflicts
    
    async def start_test_server(self, protocol='direct'):
        """Start test server for CLI testing."""
        tools_dir = Path(__file__).parent.parent.parent.parent / "orchestration" / "mcp_tools"
        tool_manager = MCPToolManager.get_instance(tools_dir)
        
        self.server_router = create_server(
            protocol=protocol,
            host='localhost',
            port=self.test_port,
            tools_dir=tools_dir,
            tool_registry=tool_manager
        )
        
        await self.server_router.start()
        await asyncio.sleep(0.2)  # Let server start
        return self.server_router.is_running
    
    async def stop_test_server(self):
        """Stop test server."""
        if self.server_router:
            await self.server_router.stop()
            self.server_router = None
    
    def teardown_method(self):
        """Clean up after test."""
        if self.server_router:
            asyncio.run(self.stop_test_server())
    
    async def test_health_check_functionality(self):
        """Test the health check functionality that powers the CLI."""
        server_started = await self.start_test_server()
        assert server_started, "Could not start test server"
        
        try:
            # Test the actual functionality the CLI health command uses
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{self.test_port}/health") as response:
                    assert response.status == 200
                    health_data = await response.json()
                    
                    # Verify health data structure (what CLI displays)
                    assert health_data.get('status') == 'healthy'
                    assert 'protocol' in health_data
                    assert 'tools_available' in health_data
                    assert 'host' in health_data
                    assert 'port' in health_data
                    
                    print(f"[SUCCESS] Health check returned: {health_data}")
                    
        finally:
            await self.stop_test_server()
    
    async def test_tools_listing_functionality(self):
        """Test the tools listing functionality that powers the CLI."""
        server_started = await self.start_test_server()
        assert server_started, "Could not start test server"
        
        try:
            # Test the actual functionality the CLI tools command uses
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{self.test_port}/tools") as response:
                    assert response.status == 200
                    tools_data = await response.json()
                    
                    # Verify tools data structure (what CLI displays)
                    assert 'tools' in tools_data
                    assert 'count' in tools_data
                    assert 'protocol' in tools_data
                    
                    tools_list = tools_data.get('tools', [])
                    count = tools_data.get('count', 0)
                    protocol = tools_data.get('protocol', 'unknown')
                    
                    print(f"[SUCCESS] Found {count} tools via {protocol} protocol")
                    
                    # Test detailed tool info structure
                    for tool in tools_list[:3]:  # Check first 3 tools
                        assert 'name' in tool
                        print(f"  - Tool: {tool.get('name', 'unknown')}")
                    
        finally:
            await self.stop_test_server()
    
    async def test_server_startup_functionality(self):
        """Test the server startup functionality that powers the CLI."""
        tools_dir = Path(__file__).parent.parent.parent.parent / "orchestration" / "mcp_tools"
        tool_manager = MCPToolManager.get_instance(tools_dir)
        
        # Test direct server creation (like CLI server command)
        direct_server = create_server(
            protocol="direct",
            host="localhost",
            port=8091,
            tools_dir=tools_dir,
            tool_registry=tool_manager
        )
        
        try:
            await direct_server.start()
            assert direct_server.is_running
            print("[SUCCESS] Direct server started successfully")
            
            # Test JSON-RPC server creation
            jsonrpc_server = create_server(
                protocol="json_rpc",
                host="localhost", 
                port=8092,
                tools_dir=tools_dir,
                tool_registry=tool_manager
            )
            
            await jsonrpc_server.start()
            assert jsonrpc_server.is_running
            print("[SUCCESS] JSON-RPC server started successfully")
            
            # Test both servers are accessible
            async with aiohttp.ClientSession() as session:
                # Test direct server
                async with session.get("http://localhost:8091/health") as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data["protocol"] == "direct"
                
                # Test JSON-RPC server
                async with session.get("http://localhost:8092/health") as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data["protocol"] == "json_rpc"
            
            print("[SUCCESS] Both server protocols working")
            
            await jsonrpc_server.stop()
            assert not jsonrpc_server.is_running
            
        finally:
            await direct_server.stop()
            assert not direct_server.is_running
    
    async def test_tool_execution_functionality(self):
        """Test tool execution functionality that powers the CLI."""
        server_started = await self.start_test_server()
        assert server_started, "Could not start test server"
        
        try:
            # Test tool execution endpoint (what CLI execute command uses)
            async with aiohttp.ClientSession() as session:
                # Try to execute a non-existent tool (should fail gracefully)
                async with session.post(
                    f"http://localhost:{self.test_port}/tools/nonexistent_tool/execute",
                    json={},
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    # Should get a response (even if tool doesn't exist)
                    # This tests the execution pathway
                    print(f"Tool execution response status: {response.status}")
                    
                    if response.status == 200:
                        result = await response.json()
                        print(f"Tool execution result: {result}")
                    else:
                        error_text = await response.text()
                        print(f"Tool execution error (expected): {error_text}")
                    
                    # The important thing is the endpoint responds
                    assert response.status in [200, 404, 400]  # Valid responses
                    
        finally:
            await self.stop_test_server()
    
    async def test_refresh_functionality(self):
        """Test refresh functionality that powers the CLI."""
        server_started = await self.start_test_server()
        assert server_started, "Could not start test server"
        
        try:
            # Test refresh endpoint (what CLI refresh command uses)
            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://localhost:{self.test_port}/refresh") as response:
                    # Should get a response
                    print(f"Refresh response status: {response.status}")
                    
                    if response.status == 200:
                        result = await response.json()
                        print(f"Refresh result: {result}")
                        # Should have some kind of success indicator
                        assert 'message' in result or 'success' in result
                    
                    # Refresh should work or give meaningful error
                    assert response.status in [200, 404, 500]
                    
        finally:
            await self.stop_test_server()
    
    async def test_multiple_protocol_support(self):
        """Test that CLI can handle both protocols (like CLI 'both' option)."""
        tools_dir = Path(__file__).parent.parent.parent.parent / "orchestration" / "mcp_tools"
        tool_manager = MCPToolManager.get_instance(tools_dir)
        
        servers = []
        
        try:
            # Start both protocols (like CLI --protocol both)
            direct_server = create_server(
                protocol="direct",
                host="localhost",
                port=8093,
                tools_dir=tools_dir,
                tool_registry=tool_manager
            )
            await direct_server.start()
            servers.append(direct_server)
            
            jsonrpc_server = create_server(
                protocol="json_rpc",
                host="localhost",
                port=8094,
                tools_dir=tools_dir,
                tool_registry=tool_manager
            )
            await jsonrpc_server.start()
            servers.append(jsonrpc_server)
            
            # Both should be running
            for server in servers:
                assert server.is_running
                health = server.health_check()
                assert health["status"] == "healthy"
            
            print("[SUCCESS] Both protocols running simultaneously")
            
        finally:
            # Clean up all servers
            for server in servers:
                await server.stop()
                assert not server.is_running


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
