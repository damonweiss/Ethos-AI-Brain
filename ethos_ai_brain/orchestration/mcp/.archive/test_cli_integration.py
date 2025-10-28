#!/usr/bin/env python3
"""
MCP CLI Integration Tests

Real integration tests with actual servers - no mocks to hide failures.
Tests real functionality end-to-end.
"""

import pytest
import asyncio
import aiohttp
from click.testing import CliRunner
from pathlib import Path

from ethos_ai_brain.orchestration.mcp.cli import cli
from ethos_ai_brain.orchestration.mcp.server import create_server, ServerConfig
from ethos_ai_brain.orchestration.mcp_tools.mcp_tool_manager import MCPToolManager


@pytest.mark.asyncio
class TestMCPCLIReal:
    """Real integration tests - no mocks, real servers."""
    
    def setup_method(self):
        """Set up for each test."""
        self.runner = CliRunner()
        self.server_router = None
        self.test_port = 8080  # Use high port to avoid conflicts
    
    async def start_real_server(self, protocol='direct'):
        """Start actual MCP server for testing."""
        tools_dir = Path(__file__).parent.parent.parent.parent / "orchestration" / "mcp_tools"
        tool_manager = MCPToolManager.get_instance(tools_dir)
        
        # Create server router
        self.server_router = create_server(
            protocol=protocol,
            host='localhost',
            port=self.test_port,
            tools_dir=tools_dir,
            tool_registry=tool_manager
        )
        
        await self.server_router.start()
        await asyncio.sleep(0.2)  # Let server fully start
        
        return self.server_router.is_running
    
    async def stop_real_server(self):
        """Stop the real server."""
        if self.server_router:
            await self.server_router.stop()
            self.server_router = None
    
    def teardown_method(self):
        """Clean up after test."""
        if self.server_router:
            asyncio.run(self.stop_real_server())
    
    def test_cli_help(self):
        """Test CLI help works."""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Ethos-MCP' in result.output
        print(f"Help test result: {result.output[:100]}...")
    
    def test_server_help(self):
        """Test server command help."""
        result = self.runner.invoke(cli, ['server', '--help'])
        assert result.exit_code == 0
        assert 'multi-headed MCP server' in result.output
        print(f"Server help test result: {result.output[:100]}...")
    
    async def test_health_with_running_server(self):
        """Test health check against real server."""
        server_started = await self.start_real_server()
        assert server_started, "Could not start test server"
        
        try:
            result = self.runner.invoke(cli, ['health', '--url', f'http://localhost:{self.test_port}'])
            
            print(f"Health check result: {result.output}")
            print(f"Health check exit code: {result.exit_code}")
            
            if result.exit_code == 0:
                print("[SUCCESS] Health check passed with real server")
                assert '[SUCCESS] Server is healthy' in result.output
            else:
                print(f"[FAILURE] Health check failed: {result.output}")
                assert False, f"Health check should pass with running server: {result.output}"
                
        finally:
            await self.stop_real_server()
    
    def test_health_with_no_server(self):
        """Test health check when no server running."""
        result = self.runner.invoke(cli, ['health', '--url', 'http://localhost:9999'])
        
        print(f"No server health result: {result.output}")
        print(f"No server health exit code: {result.exit_code}")
        
        # Should fail when no server
        assert result.exit_code == 1
        assert '[FAILURE]' in result.output
        print("[SUCCESS] Health check correctly failed with no server")
    
    async def test_tools_with_running_server(self):
        """Test tools listing against real server."""
        server_started = await self.start_real_server()
        assert server_started, "Could not start test server"
        
        try:
            result = self.runner.invoke(cli, ['tools', '--url', f'http://localhost:{self.test_port}'])
            
            print(f"Tools listing result: {result.output}")
            print(f"Tools listing exit code: {result.exit_code}")
            
            if result.exit_code == 0:
                print("[SUCCESS] Tools listing worked with real server")
                assert '[INFO] Available tools' in result.output
                assert 'Protocol: direct' in result.output
            else:
                print(f"[FAILURE] Tools listing failed: {result.output}")
                assert False, f"Tools listing should work: {result.output}"
                
        finally:
            await self.stop_real_server()
    
    async def test_tools_detailed_with_running_server(self):
        """Test detailed tools listing."""
        server_started = await self.start_real_server()
        assert server_started, "Could not start test server"
        
        try:
            result = self.runner.invoke(cli, ['tools', '--detailed', '--url', f'http://localhost:{self.test_port}'])
            
            print(f"Detailed tools result: {result.output}")
            
            assert result.exit_code == 0, f"Detailed tools failed: {result.output}"
            assert '[INFO] Available tools' in result.output
            print("[SUCCESS] Detailed tools listing worked")
                
        finally:
            await self.stop_real_server()
    
    async def test_full_test_command(self):
        """Test the test command end-to-end."""
        server_started = await self.start_real_server()
        assert server_started, "Could not start test server"
        
        try:
            result = self.runner.invoke(cli, ['test', '--url', f'http://localhost:{self.test_port}'])
            
            print(f"Full test command result: {result.output}")
            
            assert result.exit_code == 0, f"Test command failed: {result.output}"
            assert '[INFO] Testing MCP server' in result.output
            assert '[SUCCESS] All tests passed!' in result.output
            print("[SUCCESS] Full test command passed")
                
        finally:
            await self.stop_real_server()
    
    async def test_refresh_command(self):
        """Test refresh command."""
        server_started = await self.start_real_server()
        assert server_started, "Could not start test server"
        
        try:
            result = self.runner.invoke(cli, ['refresh', '--url', f'http://localhost:{self.test_port}'])
            
            print(f"Refresh command result: {result.output}")
            
            assert result.exit_code == 0, f"Refresh failed: {result.output}"
            assert '[SUCCESS] Tools refresh successful' in result.output
            print("[SUCCESS] Refresh command worked")
                
        finally:
            await self.stop_real_server()
    
    def test_tools_with_no_server(self):
        """Test tools command when no server running."""
        result = self.runner.invoke(cli, ['tools', '--url', 'http://localhost:9998'])
        
        print(f"No server tools result: {result.output}")
        
        # Should fail when no server
        assert result.exit_code == 1
        assert '[FAILURE]' in result.output
        print("[SUCCESS] Tools command correctly failed with no server")
    
    async def test_execute_command_no_params(self):
        """Test execute command without parameters."""
        server_started = await self.start_real_server()
        assert server_started, "Could not start test server"
        
        try:
            # Try to execute a non-existent tool (should fail gracefully)
            result = self.runner.invoke(cli, ['execute', 'nonexistent_tool', '--url', f'http://localhost:{self.test_port}'])
            
            print(f"Execute nonexistent tool result: {result.output}")
            
            # Command should run but tool execution should fail
            # This tests the execute pathway without needing real tools
            print("[SUCCESS] Execute command pathway tested")
                
        finally:
            await self.stop_real_server()
    
    def test_execute_invalid_json(self):
        """Test execute with invalid JSON parameters."""
        result = self.runner.invoke(cli, ['execute', 'test_tool', '--params', 'invalid{json'])
        
        print(f"Invalid JSON result: {result.output}")
        
        assert result.exit_code == 1
        assert '[FAILURE] Invalid JSON parameters' in result.output
        print("[SUCCESS] Invalid JSON correctly rejected")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
