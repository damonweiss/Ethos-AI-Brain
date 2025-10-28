#!/usr/bin/env python3
"""
AI Engine MCP Integration Tests

Test the AI Engine's MCP server management functionality.
Real integration tests with actual server startup/shutdown.
"""

import pytest
import asyncio
import aiohttp
from pathlib import Path

from ethos_ai_brain.core.ai_agent.ai_engine import AIEngine


@pytest.mark.asyncio
class TestAIEngineMCP:
    """Test AI Engine MCP server integration."""
    
    def setup_method(self):
        """Set up for each test."""
        self.ai_engine = None
    
    async def cleanup_engine(self):
        """Clean up AI Engine."""
        if self.ai_engine:
            await self.ai_engine.shutdown()
            self.ai_engine = None
    
    def teardown_method(self):
        """Clean up after test."""
        if self.ai_engine:
            asyncio.run(self.cleanup_engine())
    
    async def test_ai_engine_server_startup(self):
        """Test AI Engine starts MCP servers correctly."""
        self.ai_engine = AIEngine()
        
        # Test initialization starts servers
        await self.ai_engine.initialize()
        
        try:
            # Check that servers are in the registry
            assert len(self.ai_engine.mcp_servers) > 0
            print(f"AI Engine started {len(self.ai_engine.mcp_servers)} MCP servers")
            
            # Verify development server (port 8051)
            assert "development" in self.ai_engine.mcp_servers
            dev_server = self.ai_engine.mcp_servers["development"]
            assert dev_server.is_running
            assert dev_server.config.port == 8051
            assert dev_server.config.protocol == "direct"
            print("[SUCCESS] Development server running on port 8051")
            
            # Verify production server (port 8052)
            assert "production" in self.ai_engine.mcp_servers
            prod_server = self.ai_engine.mcp_servers["production"]
            assert prod_server.is_running
            assert prod_server.config.port == 8052
            assert prod_server.config.protocol == "json_rpc"
            print("[SUCCESS] Production server running on port 8052")
            
        finally:
            await self.cleanup_engine()
    
    async def test_ai_engine_server_health_endpoints(self):
        """Test AI Engine servers are accessible via HTTP."""
        self.ai_engine = AIEngine()
        await self.ai_engine.initialize()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test development server health
                async with session.get("http://localhost:8051/health") as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data["status"] == "healthy"
                    assert data["protocol"] == "direct"
                    print(f"Development server health: {data}")
                
                # Test production server health
                async with session.get("http://localhost:8052/health") as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data["status"] == "healthy"
                    assert data["protocol"] == "json_rpc"
                    print(f"Production server health: {data}")
                
                print("[SUCCESS] Both AI Engine servers accessible via HTTP")
                
        finally:
            await self.cleanup_engine()
    
    async def test_ai_engine_server_status_reporting(self):
        """Test AI Engine's get_mcp_server_status functionality."""
        self.ai_engine = AIEngine()
        await self.ai_engine.initialize()
        
        try:
            # Get server status
            status = self.ai_engine.get_mcp_server_status()
            
            # Should have both servers
            assert "development" in status
            assert "production" in status
            
            # Check development server status
            dev_status = status["development"]
            assert dev_status["protocol"] == "direct"
            assert dev_status["host"] == "localhost"
            assert dev_status["port"] == 8051
            assert dev_status["running"] == True
            assert "health" in dev_status
            
            # Check production server status
            prod_status = status["production"]
            assert prod_status["protocol"] == "json_rpc"
            assert prod_status["host"] == "0.0.0.0"  # External access
            assert prod_status["port"] == 8052
            assert prod_status["running"] == True
            assert "health" in prod_status
            
            print(f"[SUCCESS] Server status reporting: {len(status)} servers")
            for name, server_status in status.items():
                print(f"  - {name}: {server_status['protocol']} on {server_status['host']}:{server_status['port']}")
                
        finally:
            await self.cleanup_engine()
    
    async def test_ai_engine_additional_server_creation(self):
        """Test AI Engine can start additional servers."""
        self.ai_engine = AIEngine()
        await self.ai_engine.initialize()
        
        try:
            # Start additional server
            success = await self.ai_engine.start_additional_server(
                name="test_server",
                protocol="direct",
                port=8099
            )
            assert success, "Failed to start additional server"
            
            # Verify it's in the registry
            assert "test_server" in self.ai_engine.mcp_servers
            test_server = self.ai_engine.mcp_servers["test_server"]
            assert test_server.is_running
            assert test_server.config.port == 8099
            
            # Test it's accessible
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8099/health") as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data["status"] == "healthy"
            
            print("[SUCCESS] Additional server created and accessible")
            
            # Test duplicate server name fails
            duplicate_success = await self.ai_engine.start_additional_server(
                name="test_server",  # Same name
                protocol="json_rpc",
                port=8100
            )
            assert not duplicate_success, "Duplicate server name should fail"
            print("[SUCCESS] Duplicate server name correctly rejected")
            
        finally:
            await self.cleanup_engine()
    
    async def test_ai_engine_shutdown_stops_servers(self):
        """Test AI Engine shutdown properly stops all servers."""
        self.ai_engine = AIEngine()
        await self.ai_engine.initialize()
        
        # Verify servers are running
        assert len(self.ai_engine.mcp_servers) > 0
        for server in self.ai_engine.mcp_servers.values():
            assert server.is_running
        
        # Test shutdown
        await self.ai_engine.shutdown()
        
        # Verify servers are stopped
        assert len(self.ai_engine.mcp_servers) == 0
        print("[SUCCESS] AI Engine shutdown stopped all servers")
        
        # Verify servers are no longer accessible
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get("http://localhost:8051/health", timeout=aiohttp.ClientTimeout(total=1)) as response:
                    # Should not reach here
                    assert False, "Server should not be accessible after shutdown"
            except (aiohttp.ClientError, asyncio.TimeoutError):
                # Expected - server should be unreachable
                print("[SUCCESS] Servers properly shut down and unreachable")
        
        # Engine is already shut down, don't clean up again
        self.ai_engine = None
    
    async def test_ai_engine_tool_manager_integration(self):
        """Test AI Engine integrates with MCP tool manager."""
        self.ai_engine = AIEngine()
        
        # Check tool manager is initialized
        assert self.ai_engine.mcp_manager is not None
        assert len(self.ai_engine.available_tools) >= 0  # May be 0 if no tools found
        
        print(f"AI Engine initialized with {len(self.ai_engine.available_tools)} tools")
        
        await self.ai_engine.initialize()
        
        try:
            # Verify servers have access to tool manager
            for name, server in self.ai_engine.mcp_servers.items():
                assert server.tool_registry is not None
                assert server.tool_registry == self.ai_engine.mcp_manager
                print(f"[SUCCESS] Server {name} has tool registry access")
            
            # Test tools endpoint on servers
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8051/tools") as response:
                    assert response.status == 200
                    data = await response.json()
                    assert "tools" in data
                    assert "count" in data
                    print(f"Development server reports {data['count']} tools")
                
        finally:
            await self.cleanup_engine()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
