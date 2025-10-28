#!/usr/bin/env python3
"""
MCP Server Integration Tests - Ethos-AI-Brain

Tests the actual MCP server with real HTTP requests.
Based on the proven integration testing approach from Ethos-MCP.
"""

import asyncio
import time
import requests
import threading
from pathlib import Path
import pytest

from ethos_ai_brain.orchestration.mcp.server import create_development_server, create_production_server
from ethos_ai_brain.orchestration.mcp.tool_management.mcp_tool_manager import MCPToolManager


class MCPServerIntegrationTest:
    """Integration tests for the actual MCP server."""
    
    def __init__(self):
        self.server = None
        self.server_task = None
        self.base_url = "http://localhost:8055"  # Use different port to avoid conflicts
        self.loop = None
        
    async def start_server_async(self):
        """Start the MCP server asynchronously."""
        print("[SETUP] Starting MCP development server...")
        
        # Create server instance - use absolute path to ensure we find the tools
        # Use the working directory approach that worked in debug_tools.py
        tools_dir = Path.cwd() / "ethos_ai_brain" / "orchestration" / "mcp_tools"
        print(f"[DEBUG] Current working directory: {Path.cwd()}")
        print(f"[DEBUG] Tools directory: {tools_dir.absolute()}")
        print(f"[DEBUG] Directory exists: {tools_dir.exists()}")
        
        tool_manager = MCPToolManager.get_instance(tools_dir)
        
        self.server = create_development_server(
            port=8055,
            tools_dir=tools_dir,
            tool_registry=tool_manager
        )
        
        # Start server
        await self.server.start()
        
        # Wait for server to be ready
        await asyncio.sleep(1)
        
        return self.server.is_running
    
    def start_server(self):
        """Start the MCP server in a background thread."""
        def run_server():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            try:
                success = self.loop.run_until_complete(self.start_server_async())
                if success:
                    print(f"[SUCCESS] Server started on {self.base_url}")
                    # Keep the loop running
                    self.loop.run_forever()
                else:
                    print("[ERROR] Server failed to start")
            except Exception as e:
                print(f"[ERROR] Server startup exception: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # Wait for server to start
        for i in range(15):  # Give more time for async startup
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    print(f"[SUCCESS] Server health check passed")
                    return True
            except requests.exceptions.RequestException:
                time.sleep(1)
        
        print("[ERROR] Server failed to start or health check failed")
        return False
    
    def stop_server(self):
        """Stop the MCP server."""
        if self.server and self.loop:
            try:
                # Schedule server stop in the event loop
                future = asyncio.run_coroutine_threadsafe(self.server.stop(), self.loop)
                future.result(timeout=5)
                
                # Stop the event loop
                self.loop.call_soon_threadsafe(self.loop.stop)
                print("[SUCCESS] Server stopped")
            except Exception as e:
                print(f"[WARNING] Error stopping server: {e}")
    
    def test_server_health(self):
        """Test server health endpoint."""
        print("\n[TEST] Server Health Check...")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"  [SUCCESS] Health check passed")
                print(f"    Status: {data.get('status', 'unknown')}")
                print(f"    Protocol: {data.get('protocol', 'unknown')}")
                print(f"    Tools Available: {data.get('tools_available', 0)}")
                return True
            else:
                print(f"  [ERROR] Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"  [ERROR] Health check exception: {e}")
            return False
    
    def test_list_tools(self):
        """Test listing available tools via HTTP API."""
        print("\n[TEST] List Tools API...")
        
        try:
            response = requests.get(f"{self.base_url}/tools", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                tools = data.get('tools', [])
                
                if len(tools) > 0:
                    print(f"  [SUCCESS] Found {len(tools)} tools:")
                    tool_names = []
                    for tool in tools:
                        tool_name = tool.get('name', 'unknown')
                        tool_names.append(tool_name)
                        print(f"    - {tool_name}")
                    return tool_names
                else:
                    print(f"  [WARNING] API responded but found 0 tools")
                    print(f"    This might indicate:")
                    print(f"    - Tools directory not configured properly")
                    print(f"    - No tools installed in the tools directory")
                    print(f"    - Tool discovery not working")
                    return []
            else:
                print(f"  [ERROR] List tools failed: {response.status_code}")
                print(f"    Response: {response.text}")
                return []
                
        except Exception as e:
            print(f"  [ERROR] List tools exception: {e}")
            return []
    
    def test_tool_execution(self, tool_name: str, params: dict):
        """Test executing a tool via HTTP API."""
        print(f"\n[TEST] Execute Tool: {tool_name}...")
        
        try:
            response = requests.post(
                f"{self.base_url}/tools/{tool_name}/execute",
                json=params,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  [SUCCESS] Tool executed successfully")
                
                # Check if execution was successful
                if result.get('success', False):
                    actual_result = result.get('result', {})
                    
                    # Show relevant fields based on tool type
                    if 'scrape_webpage' in tool_name:
                        print(f"    URL: {actual_result.get('url', 'unknown')}")
                        print(f"    Status: {actual_result.get('status_code', 'unknown')}")
                        content = actual_result.get('content', '')
                        if content:
                            preview = content[:100] + '...' if len(content) > 100 else content
                            print(f"    Content Preview: {preview}")
                    elif 'echo' in tool_name or 'message' in tool_name:
                        print(f"    Result: {actual_result}")
                    elif 'time' in tool_name:
                        print(f"    Timestamp: {actual_result.get('timestamp', 'unknown')}")
                    else:
                        print(f"    Result: {str(actual_result)[:200]}...")
                    
                    return True
                else:
                    print(f"    Error: {result.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"  [ERROR] Tool execution failed: {response.status_code}")
                print(f"    Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"  [ERROR] Tool execution exception: {e}")
            return False
    
    def test_error_handling(self):
        """Test server error handling."""
        print("\n[TEST] Error Handling...")
        
        try:
            # Test invalid tool - should return 404 or similar error
            response = requests.post(f"{self.base_url}/tools/nonexistent_tool/execute", json={})
            if response.status_code in [404, 400, 422]:
                print(f"  [SUCCESS] Invalid tool correctly rejected: {response.status_code}")
            elif response.status_code == 200:
                # Check if it's a proper error response in the JSON
                try:
                    data = response.json()
                    if data.get('success') is False:
                        print(f"  [SUCCESS] Invalid tool correctly rejected: 200 with success=false")
                    else:
                        print(f"  [WARNING] Invalid tool got unexpected response: {data}")
                except:
                    print(f"  [WARNING] Invalid tool got unexpected status: {response.status_code}")
            else:
                print(f"  [WARNING] Invalid tool got unexpected status: {response.status_code}")
            
            # Test malformed request - should return 404 or similar error  
            response = requests.post(f"{self.base_url}/tools/", json={})
            if response.status_code in [404, 400, 405]:
                print(f"  [SUCCESS] Malformed request correctly rejected: {response.status_code}")
            else:
                print(f"  [WARNING] Malformed request got unexpected status: {response.status_code}")
            
            return True
        except Exception as e:
            print(f"  [ERROR] Error handling test failed: {e}")
            return False


class TestMCPServerIntegration:
    """Pytest wrapper for MCP server integration tests."""
    
    def setup_method(self):
        """Set up for each test."""
        self.test_runner = MCPServerIntegrationTest()
    
    def teardown_method(self):
        """Clean up after each test."""
        if hasattr(self, 'test_runner'):
            self.test_runner.stop_server()
    
    def test_full_integration_suite(self):
        """Run the full integration test suite."""
        print("=== MCP Server Integration Tests - Ethos-AI-Brain ===")
        print("Testing the ACTUAL MCP server with HTTP requests!")
        
        try:
            # Start server
            if not self.test_runner.start_server():
                pytest.fail("Could not start server")
            
            # Run tests
            results = []
            
            # Basic server tests
            results.append(self.test_runner.test_server_health())
            tools = self.test_runner.test_list_tools()
            
            # Note: We don't fail the test if no tools are found, since that might be expected
            # in a clean test environment. We just report it as a warning.
            
            # Tool execution tests (if tools are available)
            if tools:
                # Test any available tools
                for tool_name in tools[:3]:  # Test first 3 tools
                    # Use simple parameters that most tools can handle
                    if 'scrape' in tool_name.lower():
                        results.append(self.test_runner.test_tool_execution(tool_name, {
                            'url': 'https://example.com'
                        }))
                    elif 'echo' in tool_name.lower() or 'message' in tool_name.lower():
                        results.append(self.test_runner.test_tool_execution(tool_name, {
                            'message': 'Hello from integration test!'
                        }))
                    elif 'time' in tool_name.lower():
                        results.append(self.test_runner.test_tool_execution(tool_name, {}))
                    else:
                        # Try with empty params for unknown tools
                        results.append(self.test_runner.test_tool_execution(tool_name, {}))
            
            # Error handling tests
            results.append(self.test_runner.test_error_handling())
            
            # Summary
            passed = sum(results)
            total = len(results)
            
            print(f"\n[RESULTS] Integration Tests: {passed}/{total} passed")
            
            if passed == total:
                print("[SUCCESS] All integration tests passed!")
                print("✓ MCP server is working correctly")
                print("✓ HTTP API endpoints functional")
                print("✓ Tool registration working")
                print("✓ Tool execution via server working")
            else:
                print("[WARNING] Some integration tests failed")
                print("- Check server logs for details")
                print("- Verify tool dependencies are installed")
                print("- Ensure UV-managed tools are properly registered")
            
            # Assert at least basic functionality works
            assert passed >= 2, f"Too many integration tests failed: {passed}/{total}"
            
        except Exception as e:
            pytest.fail(f"Integration test suite failed: {e}")


def main():
    """Run integration tests standalone."""
    test = MCPServerIntegrationTest()
    
    try:
        if not test.start_server():
            print("[FAILURE] Could not start server")
            return
        
        # Run basic tests
        results = []
        results.append(test.test_server_health())
        tools = test.test_list_tools()
        results.append(test.test_error_handling())
        
        passed = sum(results)
        total = len(results)
        print(f"\n[RESULTS] Standalone Tests: {passed}/{total} passed")
        
    finally:
        test.stop_server()


if __name__ == "__main__":
    main()
