#!/usr/bin/env python3
"""
Ethos-AI-Brain MCP CLI

Command-line interface for the multi-headed MCP server system.
Supports Direct HTTP and JSON-RPC protocols with AI Engine integration.
"""

import click
import asyncio
import json
import aiohttp
from pathlib import Path
from typing import Optional

from .server import create_server, create_development_server, create_production_server, ServerConfig
from .tool_management.mcp_tool_manager import MCPToolManager
from ...core.ai_agent.ai_engine import AIEngine



@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Ethos-MCP: Universal MCP Server with API Wrapper capabilities."""
    pass

@cli.command()
@click.option('--host', default='localhost', help='Host to bind to')
@click.option('--port', default=8051, help='Port to bind to')
@click.option('--protocol', type=click.Choice(['direct', 'json_rpc', 'both']), 
              default='direct', help='Server protocol')
@click.option('--tools-dir', type=click.Path(path_type=Path), 
              help='Path to MCP tools directory')
def server(host: str, port: int, protocol: str, tools_dir: Optional[Path]):
    """Start the multi-headed MCP server."""
    async def start_server():
        click.echo(f"[INFO] Starting MCP server ({protocol}) on {host}:{port}")
        
        # Initialize tool manager
        if tools_dir:
            tool_manager = MCPToolManager.get_instance(tools_dir)
        else:
            default_tools_dir = Path(__file__).parent.parent / "mcp_tools"
            tool_manager = MCPToolManager.get_instance(default_tools_dir)
        
        servers = []
        
        try:
            if protocol == 'direct' or protocol == 'both':
                # Create direct server router
                direct_server = create_server(
                    protocol="direct",
                    host=host,
                    port=port,
                    tools_dir=tools_dir,
                    tool_registry=tool_manager
                )
                await direct_server.start()
                servers.append(direct_server)
                click.echo(f"[SUCCESS] Direct server started on {host}:{port}")
            
            if protocol == 'json_rpc' or protocol == 'both':
                rpc_port = port + 1 if protocol == 'both' else port
                # Create JSON-RPC server router
                jsonrpc_server = create_server(
                    protocol="json_rpc",
                    host=host,
                    port=rpc_port,
                    tools_dir=tools_dir,
                    tool_registry=tool_manager
                )
                await jsonrpc_server.start()
                servers.append(jsonrpc_server)
                click.echo(f"[SUCCESS] JSON-RPC server started on {host}:{rpc_port}")
            
            # Keep running until interrupted
            click.echo("Press Ctrl+C to stop the server...")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                click.echo("\n[INFO] Shutting down servers...")
                for server in servers:
                    await server.stop()
                click.echo("[SUCCESS] Servers stopped")
                
        except Exception as e:
            click.echo(f"[FAILURE] Server error: {e}")
            for server in servers:
                try:
                    await server.stop()
                except:
                    pass
    
    asyncio.run(start_server())

@cli.command()
@click.option('--url', default='http://localhost:8051', help='MCP server URL')
def test(url: str):
    """Test the MCP server functionality."""
    async def run_tests():
        click.echo(f"[INFO] Testing MCP server at {url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test health endpoint
                async with session.get(f"{url}/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        click.echo(f"[SUCCESS] Health check: {health_data.get('status', 'unknown')}")
                        click.echo(f"   Tools available: {health_data.get('tools_available', 0)}")
                    else:
                        click.echo(f"[FAILURE] Health check failed: {response.status}")
                        return
                
                # Test tools listing
                async with session.get(f"{url}/tools") as response:
                    if response.status == 200:
                        tools_data = await response.json()
                        tool_count = tools_data.get('count', 0)
                        click.echo(f"[SUCCESS] Tools listing: {tool_count} tools found")
                        
                        if tool_count > 0:
                            tools = tools_data.get('tools', [])
                            for tool in tools[:3]:  # Show first 3 tools
                                tool_name = tool.get('name', 'unknown')
                                click.echo(f"   - {tool_name}")
                            if tool_count > 3:
                                click.echo(f"   ... and {tool_count - 3} more")
                    else:
                        click.echo(f"[FAILURE] Tools listing failed: {response.status}")
                
                click.echo("[SUCCESS] All tests passed!")
                
        except Exception as e:
            click.echo(f"[FAILURE] Test failed: {e}")
    
    asyncio.run(run_tests())

@cli.command()
@click.option('--url', default='http://localhost:8051', help='MCP server URL')
def health(url: str):
    """Check server health."""
    async def check_health():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        status = health_data.get('status', 'unknown')
                        protocol = health_data.get('protocol', 'unknown')
                        tools_count = health_data.get('tools_available', 0)
                        click.echo(f"[SUCCESS] Server is {status}")
                        click.echo(f"   Protocol: {protocol}")
                        click.echo(f"   Tools available: {tools_count}")
                        click.echo(f"   Host: {health_data.get('host', 'unknown')}")
                        click.echo(f"   Port: {health_data.get('port', 'unknown')}")
                    else:
                        click.echo(f"[FAILURE] Health check failed: HTTP {response.status}")
                        exit(1)
        except Exception as e:
            click.echo(f"[FAILURE] Health check failed: {e}")
            exit(1)
    
    asyncio.run(check_health())

@cli.command()
@click.option('--url', default='http://localhost:8051', help='MCP server URL')
@click.option('--detailed', is_flag=True, help='Show detailed tool information')
def tools(url: str, detailed: bool):
    """List available tools."""
    async def list_tools():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/tools") as response:
                    if response.status == 200:
                        tools_data = await response.json()
                        tools_list = tools_data.get('tools', [])
                        count = tools_data.get('count', 0)
                        protocol = tools_data.get('protocol', 'unknown')
                        
                        click.echo(f"[INFO] Available tools ({count}) - Protocol: {protocol}")
                        
                        if count == 0:
                            click.echo("   No tools found")
                            return
                        
                        for tool in tools_list:
                            if detailed:
                                name = tool.get('name', 'unknown')
                                desc = tool.get('description', 'No description')
                                click.echo(f"  {name}")
                                click.echo(f"     {desc}")
                            else:
                                name = tool.get('name', str(tool))
                                click.echo(f"  - {name}")
                    else:
                        click.echo(f"[FAILURE] Failed to list tools: HTTP {response.status}")
                        exit(1)
        except Exception as e:
            click.echo(f"[FAILURE] Failed to list tools: {e}")
            exit(1)
    
    asyncio.run(list_tools())

@cli.command()
@click.argument('tool_name')
@click.option('--url', default='http://localhost:8051', help='MCP server URL')
@click.option('--params', help='JSON parameters for the tool')
def execute(tool_name: str, url: str, params: str):
    """Execute a specific tool."""
    async def execute_tool():
        # Parse parameters if provided
        tool_params = {}
        if params:
            try:
                tool_params = json.loads(params)
            except json.JSONDecodeError as e:
                click.echo(f"[FAILURE] Invalid JSON parameters: {e}")
                exit(1)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{url}/tools/{tool_name}/execute",
                    json=tool_params,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        click.echo(f"[SUCCESS] Tool '{tool_name}' executed successfully")
                        if result.get('success'):
                            click.echo(f"Result:")
                            click.echo(json.dumps(result.get('result', {}), indent=2))
                        else:
                            click.echo(f"[FAILURE] Tool execution failed: {result.get('error', 'Unknown error')}")
                    else:
                        error_text = await response.text()
                        click.echo(f"[FAILURE] Tool execution failed: HTTP {response.status}")
                        click.echo(f"   {error_text}")
                        exit(1)
        except Exception as e:
            click.echo(f"[FAILURE] Tool execution failed: {e}")
            exit(1)
    
    asyncio.run(execute_tool())

@cli.command()
@click.option('--url', default='http://localhost:8051', help='MCP server URL')
def refresh(url: str):
    """Refresh MCP tools discovery."""
    async def refresh_tools():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{url}/refresh") as response:
                    if response.status == 200:
                        result = await response.json()
                        click.echo("[SUCCESS] Tools refresh successful")
                        if 'message' in result:
                            click.echo(f"   {result['message']}")
                    else:
                        click.echo(f"[FAILURE] Refresh failed: HTTP {response.status}")
                        exit(1)
        except Exception as e:
            click.echo(f"[FAILURE] Refresh failed: {e}")
            exit(1)
    
    asyncio.run(refresh_tools())

@cli.command()
def ai_engine():
    """Start AI Engine with integrated MCP servers."""
    async def start_ai_engine():
        click.echo("[INFO] Starting AI Engine with MCP integration...")
        
        try:
            engine = AIEngine()
            await engine.initialize()
            
            # Show server status
            server_status = engine.get_mcp_server_status()
            click.echo("[INFO] MCP Server Status:")
            for name, status in server_status.items():
                protocol = status.get('protocol', 'unknown')
                host = status.get('host', 'unknown')
                port = status.get('port', 'unknown')
                running = status.get('running', False)
                status_text = "[SUCCESS]" if running else "[FAILURE]"
                click.echo(f"   {status_text} {name} ({protocol}) - {host}:{port}")
            
            click.echo("Press Ctrl+C to stop AI Engine...")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                click.echo("\n[INFO] Shutting down AI Engine...")
                await engine.shutdown()
                click.echo("[SUCCESS] AI Engine stopped")
                
        except Exception as e:
            click.echo(f"[FAILURE] AI Engine error: {e}")
    
    asyncio.run(start_ai_engine())

@cli.command()
@click.option('--direct-port', default=8051, help='Direct server port')
@click.option('--jsonrpc-port', default=8052, help='JSON-RPC server port')
def status(direct_port: int, jsonrpc_port: int):
    """Check status of all MCP servers."""
    async def check_all_status():
        servers = [
            ("Direct", f"http://localhost:{direct_port}"),
            ("JSON-RPC", f"http://localhost:{jsonrpc_port}")
        ]
        
        click.echo("[INFO] Checking MCP server status...")
        
        for name, url in servers:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=2)) as response:
                        if response.status == 200:
                            health_data = await response.json()
                            protocol = health_data.get('protocol', 'unknown')
                            tools_count = health_data.get('tools_available', 0)
                            click.echo(f"[SUCCESS] {name} Server ({protocol}) - {tools_count} tools")
                        else:
                            click.echo(f"[FAILURE] {name} Server - HTTP {response.status}")
            except asyncio.TimeoutError:
                click.echo(f"[TIMEOUT] {name} Server - Timeout")
            except Exception as e:
                click.echo(f"[FAILURE] {name} Server - {type(e).__name__}")
    
    asyncio.run(check_all_status())

def main():
    """Main CLI entry point."""
    cli()

if __name__ == '__main__':
    main()
