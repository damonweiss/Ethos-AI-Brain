"""
API Configuration Tools

Tools for managing and listing API configuration files for the Universal API Wrapper.
"""

from pathlib import Path
from typing import Dict


def list_api_configs(params: Dict, config_dir: Path = None) -> Dict:
    """
    List available API configuration files.
    
    Args:
        params: Dictionary of parameters (not used for this tool)
        config_dir: Path to the API configurations directory (injected by server)
        
    Returns:
        Dict containing:
        - api_configs: List of available API configurations
        - count: Number of configurations found
        - config_dir: Path to the configuration directory
        - error: Error message if directory not found
    """
    if config_dir is None:
        return {
            'error': 'Configuration directory not provided by server',
            'api_configs': [],
            'count': 0,
            'config_dir': 'unknown'
        }
    
    if not config_dir.exists():
        return {
            'error': 'API config directory not found',
            'config_dir': str(config_dir),
            'api_configs': [],
            'count': 0
        }
    
    configs = []
    for config_file in config_dir.glob('*.yaml'):
        configs.append({
            'name': config_file.stem,
            'file': config_file.name,
            'path': str(config_file),
            'size_bytes': config_file.stat().st_size,
            'modified': config_file.stat().st_mtime
        })
    
    # Sort by name for consistent output
    configs.sort(key=lambda x: x['name'])
    
    return {
        'api_configs': configs,
        'count': len(configs),
        'config_dir': str(config_dir),
        'supported_formats': ['yaml'],
        'description': 'API configurations for Universal API Wrapper integration'
    }


# Tool metadata for dynamic discovery
TOOLS = [
    {
        "name": "list_api_configs",
        "description": "List available API configuration files for Universal API Wrapper",
        "function": list_api_configs,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "returns": {
            "type": "object",
            "properties": {
                "api_configs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Configuration name"},
                            "file": {"type": "string", "description": "Filename"},
                            "path": {"type": "string", "description": "Full file path"},
                            "size_bytes": {"type": "integer", "description": "File size in bytes"},
                            "modified": {"type": "number", "description": "Last modified timestamp"}
                        }
                    }
                },
                "count": {"type": "integer", "description": "Number of configurations found"},
                "config_dir": {"type": "string", "description": "Configuration directory path"},
                "supported_formats": {"type": "array", "items": {"type": "string"}},
                "description": {"type": "string"}
            }
        },
        "tags": ["config", "api", "filesystem", "yaml"],
        "author": "Ethos Collaborative",
        "version": "1.0.0",
        "category": "api_management",
        "examples": [
            {
                "description": "List all API configurations",
                "input": {},
                "output": {
                    "api_configs": [
                        {"name": "openai", "file": "openai.yaml", "path": "/path/to/openai.yaml"}
                    ],
                    "count": 1
                }
            }
        ]
    }
]
