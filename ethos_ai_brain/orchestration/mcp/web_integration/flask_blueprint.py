#!/usr/bin/env python3
"""
Ethos-MCP - Flask Blueprint for Web Integration

This module provides Flask blueprints for dynamic MCP tool registration and execution.
It allows the MCP server to register its available tools with the Flask backend,
creating proxy endpoints that the React frontend can discover and use.

Author: Damon Weiss
Created: 2025-01-28
License: MIT
"""

import json
import requests
from datetime import datetime
from flask import Blueprint, jsonify, request
from typing import Dict, List, Any

# Create Flask blueprint for MCP endpoints
mcp_blueprint = Blueprint('mcp', __name__, url_prefix='/api/mcp')

# Global registry for MCP tools
mcp_tools_registry: Dict[str, Dict[str, Any]] = {}
mcp_server_url = "http://localhost:8051"

@mcp_blueprint.route('/register', methods=['POST'])
def register_mcp_tools():
    """
    Register MCP tools from the MCP server.
    
    Expected payload:
    {
        "tools": [
            {
                "name": "tool_name",
                "description": "Tool description",
                "parameters": {...},
                "endpoint": "/tool_endpoint"
            }
        ]
    }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({'error': 'Invalid payload. Expected JSON data with "tools" array.'}), 400
        if 'tools' not in data:
            return jsonify({'error': 'Invalid payload. Expected "tools" array.'}), 400
        
        tools = data['tools']
        registered_count = 0
        
        for tool in tools:
            if 'name' in tool:
                mcp_tools_registry[tool['name']] = {
                    'name': tool['name'],
                    'description': tool.get('description', ''),
                    'parameters': tool.get('parameters', {}),
                    'endpoint': tool.get('endpoint', f"/tools/{tool['name']}"),
                    'registered_at': datetime.now().isoformat()
                }
                registered_count += 1
        
        return jsonify({
            'message': f'Successfully registered {registered_count} MCP tools',
            'tools_registered': list(mcp_tools_registry.keys()),
            'total_tools': len(mcp_tools_registry)
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to register MCP tools: {str(e)}'}), 500

@mcp_blueprint.route('/tools', methods=['GET'])
def list_mcp_tools():
    """List all registered MCP tools."""
    try:
        tools_list = list(mcp_tools_registry.values())
        
        return jsonify({
            'tools': tools_list,
            'count': len(tools_list),
            'mcp_server_url': mcp_server_url,
            'last_updated': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to list MCP tools: {str(e)}'}), 500

@mcp_blueprint.route('/tools/<tool_name>', methods=['GET'])
def get_mcp_tool_info(tool_name: str):
    """Get information about a specific MCP tool."""
    try:
        if tool_name not in mcp_tools_registry:
            return jsonify({'error': f'Tool "{tool_name}" not found'}), 404
        
        tool_info = mcp_tools_registry[tool_name]
        
        return jsonify({
            'tool': tool_info,
            'available': True,
            'mcp_server_url': mcp_server_url
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get tool info: {str(e)}'}), 500

@mcp_blueprint.route('/tools/<tool_name>', methods=['POST'])
def execute_mcp_tool(tool_name: str):
    """
    Execute an MCP tool by proxying the request to the MCP server.
    
    This endpoint acts as a proxy between the React frontend and the MCP server,
    allowing the frontend to execute MCP tools without direct MCP server access.
    """
    try:
        if tool_name not in mcp_tools_registry:
            return jsonify({'error': f'Tool "{tool_name}" not found'}), 404
        
        tool_info = mcp_tools_registry[tool_name]
        
        # Get request data from frontend
        request_data = request.get_json() or {}
        
        # Prepare request to MCP server - add /execute to the endpoint
        mcp_endpoint = f"{mcp_server_url}/tools/{tool_name}/execute"
        
        # Forward the request to the MCP server
        try:
            mcp_response = requests.post(
                mcp_endpoint,
                json=request_data,
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            # Return MCP server response to frontend
            if mcp_response.status_code == 200:
                result = mcp_response.json()
                return jsonify({
                    'success': True,
                    'tool_name': tool_name,
                    'result': result,
                    'executed_at': datetime.now().isoformat(),
                    'mcp_server_response_time': mcp_response.elapsed.total_seconds()
                }), 200
            else:
                return jsonify({
                    'error': f'MCP server returned status {mcp_response.status_code}',
                    'mcp_response': mcp_response.text
                }), mcp_response.status_code
                
        except requests.exceptions.Timeout:
            return jsonify({'error': 'MCP server request timed out'}), 504
        except requests.exceptions.ConnectionError:
            return jsonify({'error': 'Could not connect to MCP server'}), 503
        except Exception as e:
            return jsonify({'error': f'MCP server request failed: {str(e)}'}), 502
        
    except Exception as e:
        return jsonify({'error': f'Failed to execute MCP tool: {str(e)}'}), 500

@mcp_blueprint.route('/health', methods=['GET'])
def mcp_health_check():
    """Check the health of MCP integration."""
    try:
        # Check if MCP server is reachable
        mcp_server_healthy = False
        mcp_server_error = None
        
        try:
            health_response = requests.get(f"{mcp_server_url}/health", timeout=5)
            mcp_server_healthy = health_response.status_code == 200
        except Exception as e:
            mcp_server_error = str(e)
        
        return jsonify({
            'mcp_integration_status': 'healthy',
            'registered_tools_count': len(mcp_tools_registry),
            'mcp_server_url': mcp_server_url,
            'mcp_server_healthy': mcp_server_healthy,
            'mcp_server_error': mcp_server_error,
            'tools_available': list(mcp_tools_registry.keys()),
            'last_check': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'mcp_integration_status': 'unhealthy',
            'error': str(e),
            'last_check': datetime.now().isoformat()
        }), 500

@mcp_blueprint.route('/refresh', methods=['POST'])
def refresh_mcp_tools():
    """
    Refresh MCP tools by requesting the MCP server to re-register its tools.
    This is useful when new tools are added to the MCP server.
    """
    try:
        # Clear current registry
        global mcp_tools_registry
        mcp_tools_registry.clear()
        
        # Request MCP server to re-register tools
        try:
            refresh_response = requests.post(f"{mcp_server_url}/refresh", timeout=10)
            if refresh_response.status_code == 200:
                return jsonify({
                    'message': 'MCP tools refresh initiated',
                    'mcp_server_response': refresh_response.json(),
                    'registry_cleared': True,
                    'refreshed_at': datetime.now().isoformat()
                }), 200
            else:
                return jsonify({
                    'error': f'MCP server refresh failed with status {refresh_response.status_code}',
                    'registry_cleared': True
                }), 502
                
        except requests.exceptions.ConnectionError:
            return jsonify({
                'error': 'Could not connect to MCP server for refresh',
                'registry_cleared': True
            }), 503
        except Exception as e:
            return jsonify({
                'error': f'MCP server refresh request failed: {str(e)}',
                'registry_cleared': True
            }), 502
        
    except Exception as e:
        return jsonify({'error': f'Failed to refresh MCP tools: {str(e)}'}), 500

# Error handlers for the MCP blueprint
@mcp_blueprint.errorhandler(404)
def mcp_not_found(error):
    return jsonify({
        'error': 'MCP endpoint not found',
        'available_endpoints': [
            '/api/mcp/tools',
            '/api/mcp/tools/<tool_name>',
            '/api/mcp/register',
            '/api/mcp/health',
            '/api/mcp/refresh'
        ]
    }), 404

@mcp_blueprint.errorhandler(500)
def mcp_internal_error(error):
    return jsonify({
        'error': 'MCP integration internal error',
        'message': 'An error occurred in the MCP integration system'
    }), 500

# Utility functions for MCP integration
def get_registered_tools() -> List[str]:
    """Get list of registered MCP tool names."""
    return list(mcp_tools_registry.keys())

def is_tool_registered(tool_name: str) -> bool:
    """Check if a specific tool is registered."""
    return tool_name in mcp_tools_registry

def get_tool_info(tool_name: str) -> Dict[str, Any]:
    """Get information about a specific tool."""
    return mcp_tools_registry.get(tool_name, {})

def clear_tool_registry():
    """Clear all registered tools (useful for testing)."""
    global mcp_tools_registry
    mcp_tools_registry.clear()

# Export the blueprint for use in Flask app
__all__ = ['mcp_blueprint', 'get_registered_tools', 'is_tool_registered', 'get_tool_info', 'clear_tool_registry']
