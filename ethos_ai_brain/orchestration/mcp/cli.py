#!/usr/bin/env python3
"""
Ethos-MCP CLI

Command-line interface for the Ethos-MCP server.
"""

import click
import asyncio
from pathlib import Path
from .mcp.server import EthosMCPServer
from .mcp.client import EthosMCPClient, test_mcp_server, test_mcp_server_sync

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Ethos-MCP: Universal MCP Server with API Wrapper capabilities."""
    pass

@cli.command()
@click.option('--host', default='localhost', help='Host to bind to')
@click.option('--port', default=8051, help='Port to bind to')
@click.option('--debug/--no-debug', default=True, help='Enable debug mode')
@click.option('--config-dir', type=click.Path(exists=True, path_type=Path), 
              help='Path to API config directory')
def server(host: str, port: int, debug: bool, config_dir: Path):
    """Start the Ethos-MCP server."""
    click.echo(f"🚀 Starting Ethos-MCP server on {host}:{port}")
    
    server_instance = EthosMCPServer(config_dir=config_dir)
    server_instance.run(host=host, port=port, debug=debug)

@cli.command()
@click.option('--url', default='http://localhost:8051', help='MCP server URL')
@click.option('--async-mode/--sync-mode', default=False, help='Use async client')
def test(url: str, async_mode: bool):
    """Test the MCP server functionality."""
    if async_mode:
        click.echo("Running async tests...")
        asyncio.run(test_mcp_server())
    else:
        click.echo("Running sync tests...")
        test_mcp_server_sync()

@cli.command()
@click.option('--url', default='http://localhost:8051', help='MCP server URL')
def health(url: str):
    """Check server health."""
    client = EthosMCPClient(server_url=url)
    health_result = client.health_check()
    
    if "error" in health_result:
        click.echo(f"Health check failed: {health_result['error']}")
        exit(1)
    else:
        status = health_result.get('status', 'unknown')
        tools_count = health_result.get('tools_count', 0)
        click.echo(f"Server is {status} with {tools_count} tools")

@cli.command()
@click.option('--url', default='http://localhost:8051', help='MCP server URL')
def tools(url: str):
    """List available tools."""
    client = EthosMCPClient(server_url=url)
    tools_result = client.list_tools()
    
    if "error" in tools_result:
        click.echo(f"Failed to list tools: {tools_result['error']}")
        exit(1)
    
    tools_list = tools_result.get('tools', [])
    count = tools_result.get('count', 0)
    
    click.echo(f"Available tools ({count}):")
    for tool in tools_list:
        click.echo(f"  - {tool}")

@cli.command()
@click.argument('tool_name')
@click.option('--url', default='http://localhost:8051', help='MCP server URL')
@click.option('--params', help='JSON parameters for the tool')
def execute(tool_name: str, url: str, params: str):
    """Execute a specific tool."""
    import json
    
    client = EthosMCPClient(server_url=url)
    
    # Parse parameters if provided
    tool_params = {}
    if params:
        try:
            tool_params = json.loads(params)
        except json.JSONDecodeError as e:
            click.echo(f"Invalid JSON parameters: {e}")
            exit(1)
    
    result = client.execute_tool(tool_name, tool_params)
    
    if "error" in result:
        click.echo(f"Tool execution failed: {result['error']}")
        exit(1)
    
    click.echo(f"Tool '{tool_name}' executed successfully:")
    click.echo(json.dumps(result, indent=2))

@cli.command()
@click.option('--url', default='http://localhost:8051', help='MCP server URL')
def configs(url: str):
    """List API configurations."""
    client = EthosMCPClient(server_url=url)
    result = client.list_api_configs()
    
    if "error" in result:
        click.echo(f"Failed to list configs: {result['error']}")
        exit(1)
    
    configs_list = result.get('result', {}).get('api_configs', [])
    count = result.get('result', {}).get('count', 0)
    
    click.echo(f"API Configurations ({count}):")
    for config in configs_list:
        name = config.get('name', 'unknown')
        file = config.get('file', 'unknown')
        click.echo(f"  - {name} ({file})")

def main():
    """Main CLI entry point."""
    cli()

if __name__ == '__main__':
    main()
