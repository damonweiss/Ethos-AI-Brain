#!/usr/bin/env python3
"""
Ethos-MCP Client utilities

Provides helper functions for interacting with the Ethos-MCP server.
Supports both HTTP-based communication and process management.
"""

import asyncio
import json
import subprocess
import requests
from typing import Any, Dict, List, Optional
from pathlib import Path

class EthosMCPClient:
    """Client for interacting with Ethos-MCP server."""
    
    def __init__(self, 
                 server_url: str = "http://localhost:8051",
                 server_command: Optional[List[str]] = None):
        self.server_url = server_url.rstrip('/')
        self.server_command = server_command or [
            "python", "-m", "src.mcp.server"
        ]
        self.process = None
    
    # HTTP-based communication methods
    def health_check(self) -> Dict[str, Any]:
        """Check server health via HTTP."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=1)
            response.raise_for_status()
            return response.json()
        except requests.ConnectionError as e:
            return {"error": f"Connection failed: {str(e)}"}
        except requests.Timeout as e:
            return {"error": f"Request timed out: {str(e)}"}
        except requests.RequestException as e:
            return {"error": f"Health check failed: {str(e)}"}
    
    def list_tools(self) -> Dict[str, Any]:
        """Get list of available tools via HTTP."""
        try:
            response = requests.get(f"{self.server_url}/tools", timeout=1)
            response.raise_for_status()
            return response.json()
        except requests.ConnectionError as e:
            return {"error": f"Connection failed: {str(e)}"}
        except requests.Timeout as e:
            return {"error": f"Request timed out: {str(e)}"}
        except requests.RequestException as e:
            return {"error": f"Failed to list tools: {str(e)}"}
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get information about a specific tool via HTTP."""
        try:
            response = requests.get(f"{self.server_url}/tools/{tool_name}", timeout=1)
            response.raise_for_status()
            return response.json()
        except requests.ConnectionError as e:
            return {"error": f"Connection failed: {str(e)}"}
        except requests.Timeout as e:
            return {"error": f"Request timed out: {str(e)}"}
        except requests.RequestException as e:
            return {"error": f"Failed to get tool info: {str(e)}"}
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a tool via HTTP."""
        try:
            response = requests.post(
                f"{self.server_url}/tools/{tool_name}/execute",
                json=params or {},
                timeout=2,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
        except requests.ConnectionError as e:
            return {"error": f"Connection failed: {str(e)}"}
        except requests.Timeout as e:
            return {"error": f"Request timed out: {str(e)}"}
        except requests.RequestException as e:
            return {"error": f"Failed to execute tool: {str(e)}"}
    
    def refresh_tools(self) -> Dict[str, Any]:
        """Refresh tools via HTTP."""
        try:
            response = requests.post(f"{self.server_url}/refresh", timeout=1)
            response.raise_for_status()
            return response.json()
        except requests.ConnectionError as e:
            return {"error": f"Connection failed: {str(e)}"}
        except requests.Timeout as e:
            return {"error": f"Request timed out: {str(e)}"}
        except requests.RequestException as e:
            return {"error": f"Failed to refresh tools: {str(e)}"}
    
    # Process management methods
    async def start_server(self) -> subprocess.Popen:
        """Start the MCP server process."""
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.server_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            # Give the server a moment to start
            await asyncio.sleep(2)
            return self.process
        except Exception as e:
            raise RuntimeError(f"Failed to start MCP server: {str(e)}")
    
    async def stop_server(self):
        """Stop the MCP server process."""
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            self.process = None
    
    def is_server_running(self) -> bool:
        """Check if the server is running via HTTP health check."""
        health = self.health_check()
        return "error" not in health and health.get("status") == "healthy"
    
    # Convenience methods
    def echo_test(self, message: str = "Hello, Ethos-MCP!") -> Dict[str, Any]:
        """Test the server with an echo message."""
        return self.execute_tool("echo_message", {"message": message})
    
    def get_project_info(self) -> Dict[str, Any]:
        """Get project information."""
        return self.execute_tool("get_project_info")
    
    def get_system_time(self) -> Dict[str, Any]:
        """Get current system time."""
        return self.execute_tool("get_system_time")
    
    def list_api_configs(self) -> Dict[str, Any]:
        """List available API configurations."""
        return self.execute_tool("list_api_configs")

# Async wrapper for process management
class AsyncEthosMCPClient(EthosMCPClient):
    """Async version of the MCP client with process management."""
    
    async def health_check_async(self) -> Dict[str, Any]:
        """Async health check."""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/health", timeout=5) as response:
                    return await response.json()
        except Exception as e:
            return {"error": f"Health check failed: {str(e)}"}
    
    async def execute_tool_async(self, tool_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Async tool execution."""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.server_url}/tools/{tool_name}/execute",
                    json=params or {},
                    timeout=30
                ) as response:
                    return await response.json()
        except Exception as e:
            return {"error": f"Failed to execute tool: {str(e)}"}

# Example usage and testing functions
async def test_mcp_server():
    """Test the MCP server functionality."""
    client = AsyncEthosMCPClient()
    
    try:
        print("Starting Ethos-MCP server...")
        await client.start_server()
        
        # Wait for server to be ready
        for i in range(10):
            if client.is_server_running():
                print("Server is running!")
                break
            print(f"Waiting for server... ({i+1}/10)")
            await asyncio.sleep(1)
        else:
            print("Server failed to start")
            return
        
        print("\nTesting server functionality...")
        
        # Test health check
        health = client.health_check()
        print(f"Health: {health.get('status', 'unknown')}")
        
        # List tools
        tools_response = client.list_tools()
        tools = tools_response.get('tools', [])
        print(f"Available tools: {tools}")
        
        # Test echo
        echo_result = client.echo_test("Hello from test!")
        print(f"Echo test: {echo_result.get('result', {}).get('echo', 'failed')}")
        
        # Get project info
        project_info = client.get_project_info()
        project_name = project_info.get('result', {}).get('project_name', 'unknown')
        print(f"Project: {project_name}")
        
        # List API configs
        api_configs = client.list_api_configs()
        configs = api_configs.get('result', {}).get('api_configs', [])
        print(f"API configs: {[c.get('name') for c in configs]}")
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Error testing MCP server: {e}")
    finally:
        print("Stopping server...")
        await client.stop_server()

def test_mcp_server_sync():
    """Synchronous test of the MCP server (assumes server is already running)."""
    client = EthosMCPClient()
    
    print("Testing Ethos-MCP server (sync)...")
    
    # Test health check
    health = client.health_check()
    print(f"Health: {health.get('status', 'unknown')}")
    
    if "error" in health:
        print("Server not running. Start it first with: mcp-server")
        return
    
    # List tools
    tools_response = client.list_tools()
    tools = tools_response.get('tools', [])
    print(f"Available tools: {tools}")
    
    # Test echo
    echo_result = client.echo_test("Hello from sync test!")
    echo_message = echo_result.get('result', {}).get('echo', 'failed')
    print(f"Echo test: {echo_message}")
    
    # Get project info
    project_info = client.get_project_info()
    project_name = project_info.get('result', {}).get('project_name', 'unknown')
    print(f"Project: {project_name}")
    
    print("Sync test completed!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        test_mcp_server_sync()
    else:
        asyncio.run(test_mcp_server())
