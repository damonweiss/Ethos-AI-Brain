"""
Basic System Utilities

Collection of basic system tools for MCP server functionality.
"""

import time
from datetime import datetime
from typing import Dict


def echo_message(params: Dict) -> Dict:
    """
    Echo back a message for testing MCP communication.
    
    Args:
        params: Dictionary containing:
            - message (str, optional): Message to echo back
            
    Returns:
        Dict containing:
        - echo: The echoed message
        - timestamp: ISO timestamp when processed
        - processed_by: Server name that processed the request
    """
    if params is None:
        params = {}
    
    message = params.get('message', 'No message provided')
    
    return {
        'echo': message,
        'timestamp': datetime.now().isoformat(),
        'processed_by': 'ethos-mcp-server'  # Will be overridden by server
    }


def get_system_time(params: Dict) -> Dict:
    """
    Get current system time and timestamp.
    
    Args:
        params: Dictionary of parameters (not used for this tool)
        
    Returns:
        Dict containing:
        - timestamp: ISO formatted timestamp
        - unix_timestamp: Unix timestamp (seconds since epoch)
        - server: Server name that processed the request
        - timezone: System timezone information
    """
    now = datetime.now()
    
    return {
        'timestamp': now.isoformat(),
        'unix_timestamp': time.time(),
        'server': 'ethos-mcp-server',  # Will be overridden by server
        'timezone': str(now.astimezone().tzinfo),
        'utc_timestamp': datetime.utcnow().isoformat() + 'Z'
    }


# Tool metadata for dynamic discovery
TOOLS = [
    {
        "name": "echo_message",
        "namespace": "domain.system",
        "category": "communication",
        "description": "Echo back a message for testing MCP communication and connectivity",
        "function": echo_message,
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to echo back",
                    "default": "No message provided",
                    "examples": ["Hello World", "Test message", "🚀 MCP is working!"]
                }
            },
            "required": [],
            "additionalProperties": False
        },
        "returns": {
            "type": "object",
            "properties": {
                "echo": {"type": "string", "description": "The echoed message"},
                "timestamp": {"type": "string", "format": "date-time", "description": "ISO timestamp when processed"},
                "processed_by": {"type": "string", "description": "Server name that processed the request"}
            },
            "required": ["echo", "timestamp", "processed_by"]
        },
        "tags": ["testing", "communication", "echo", "debug"],
        "author": "Ethos Collaborative",
        "version": "1.0.0",
        "category": "utilities",
        "examples": [
            {
                "description": "Echo a simple message",
                "input": {"message": "Hello MCP!"},
                "output": {
                    "echo": "Hello MCP!",
                    "timestamp": "2025-01-28T12:00:00.000Z",
                    "processed_by": "ethos-mcp-server"
                }
            },
            {
                "description": "Echo with no message (uses default)",
                "input": {},
                "output": {
                    "timestamp": "2025-01-28T12:00:00.000Z",
                    "processed_by": "ethos-mcp-server"
                }
            }
        ]
    },
    {
        "name": "get_system_time",
        "namespace": "domain.system",
        "category": "system_info",
        "description": "Get current system time and timestamp information",
        "function": get_system_time,
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        },
        "returns": {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string", "format": "date-time", "description": "ISO formatted timestamp"},
                "unix_timestamp": {"type": "number", "description": "Unix timestamp (seconds since epoch)"},
                "server": {"type": "string", "description": "Server name that processed the request"},
                "timezone": {"type": "string", "description": "System timezone information"},
                "utc_timestamp": {"type": "string", "format": "date-time", "description": "UTC timestamp"}
            },
            "required": ["timestamp", "unix_timestamp", "server"]
        },
        "tags": ["time", "system", "timestamp", "utility"],
        "author": "Ethos Collaborative",
        "version": "1.0.0",
        "category": "utilities",
        "examples": [
            {
                "description": "Get current system time",
                "input": {},
                "output": {
                    "timestamp": "2025-01-28T12:00:00.000Z",
                    "unix_timestamp": 1737288000.0,
                    "server": "ethos-mcp-server",
                    "timezone": "UTC"
                }
            }
        ]
    }
]
