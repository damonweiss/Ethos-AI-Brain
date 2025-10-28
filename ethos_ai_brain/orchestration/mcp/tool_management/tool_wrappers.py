#!/usr/bin/env python3
"""
MCP Tool Wrappers

Wrapper classes for different types of tools:
- APIWrapper: YAML-driven REST API integration
- FunctionWrapper: Python function/class introspection
- ModuleWrapper: Automatic Python library wrapping
"""

import yaml
import json
import requests
import inspect
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class APIWrapper:
    """YAML-driven REST API wrapper for MCP tools."""
    
    def __init__(self, config_dir: Path):
        """Initialize with API configuration directory."""
        self.config_dir = Path(config_dir)
        self.configs = {}
        self._load_configurations()
    
    def _load_configurations(self):
        """Load all YAML API configurations."""
        if not self.config_dir.exists():
            logger.warning(f"API config directory not found: {self.config_dir}")
            return
        
        for yaml_file in self.config_dir.glob('*.yaml'):
            try:
                with open(yaml_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                api_name = yaml_file.stem
                self.configs[api_name] = config
                logger.info(f"Loaded API config: {api_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")
    
    def list_apis(self) -> List[Dict[str, Any]]:
        """List available API configurations."""
        apis = []
        for name, config in self.configs.items():
            apis.append({
                'name': name,
                'display_name': config.get('name', name),
                'base_url': config.get('base_url', ''),
                'endpoints': list(config.get('endpoints', {}).keys()),
                'endpoint_count': len(config.get('endpoints', {}))
            })
        return apis
    
    def get_api_tools(self, api_name: str) -> List[Dict[str, Any]]:
        """Get MCP tools for a specific API."""
        if api_name not in self.configs:
            return []
        
        config = self.configs[api_name]
        tools = []
        
        for endpoint_name, endpoint_config in config.get('endpoints', {}).items():
            tool_name = f"{api_name}/{endpoint_name}"
            
            tools.append({
                'name': tool_name,
                'description': f"{config.get('name', api_name)} - {endpoint_name}",
                'function': self._create_api_function(api_name, endpoint_name),
                'parameters': endpoint_config.get('parameters', {}),
                'category': 'api_wrapper',
                'api_name': api_name,
                'endpoint_name': endpoint_name,
                'method': endpoint_config.get('method', 'GET'),
                'path': endpoint_config.get('path', ''),
                'wrapper_type': 'api'
            })
        
        return tools
    
    def _create_api_function(self, api_name: str, endpoint_name: str) -> Callable:
        """Create a function that executes the API call."""
        def api_function(params: Dict[str, Any]) -> Dict[str, Any]:
            return self.execute_api_call(api_name, endpoint_name, params)
        
        return api_function
    
    def execute_api_call(self, api_name: str, endpoint_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an API call based on YAML configuration."""
        if api_name not in self.configs:
            return {
                'success': False,
                'error': f'API configuration not found: {api_name}',
                'available_apis': list(self.configs.keys())
            }
        
        config = self.configs[api_name]
        endpoints = config.get('endpoints', {})
        
        if endpoint_name not in endpoints:
            return {
                'success': False,
                'error': f'Endpoint not found: {endpoint_name}',
                'available_endpoints': list(endpoints.keys())
            }
        
        endpoint = endpoints[endpoint_name]
        
        try:
            # Build request
            base_url = config.get('base_url', '')
            path = endpoint.get('path', '')
            method = endpoint.get('method', 'GET').upper()
            url = f"{base_url.rstrip('/')}{path}"
            
            # Prepare headers
            headers = {}
            auth_config = config.get('authentication', {})
            
            if auth_config.get('type') == 'bearer':
                token = auth_config.get('token', '')
                # Handle environment variable substitution
                if token.startswith('${') and token.endswith('}'):
                    import os
                    env_var = token[2:-1]
                    token = os.getenv(env_var, '')
                
                if token:
                    headers['Authorization'] = f'Bearer {token}'
            
            # Add content type for POST/PUT
            if method in ['POST', 'PUT', 'PATCH']:
                headers['Content-Type'] = 'application/json'
            
            # Validate parameters against schema
            param_schema = endpoint.get('parameters', {})
            validation_result = self._validate_parameters(params, param_schema)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': 'Parameter validation failed',
                    'validation_errors': validation_result['errors']
                }
            
            # Make request
            request_kwargs = {
                'headers': headers,
                'timeout': params.get('timeout', 30)
            }
            
            if method in ['POST', 'PUT', 'PATCH']:
                request_kwargs['json'] = params
            elif method == 'GET':
                request_kwargs['params'] = params
            
            response = requests.request(method, url, **request_kwargs)
            
            # Process response
            result = {
                'success': True,
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'api_name': api_name,
                'endpoint_name': endpoint_name,
                'method': method,
                'url': url
            }
            
            # Parse response body
            try:
                if response.headers.get('content-type', '').startswith('application/json'):
                    result['data'] = response.json()
                else:
                    result['data'] = response.text
            except:
                result['data'] = response.text
            
            # Check for HTTP errors
            if not response.ok:
                result['success'] = False
                result['error'] = f'HTTP {response.status_code}: {response.reason}'
            
            return result
            
        except requests.RequestException as e:
            return {
                'success': False,
                'error': f'Request failed: {str(e)}',
                'api_name': api_name,
                'endpoint_name': endpoint_name
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}',
                'api_name': api_name,
                'endpoint_name': endpoint_name
            }
    
    def _validate_parameters(self, params: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters against JSON schema."""
        errors = []
        
        # Check required parameters
        required = schema.get('required', [])
        for req_param in required:
            if req_param not in params:
                errors.append(f"Missing required parameter: {req_param}")
        
        # Basic type validation
        properties = schema.get('properties', {})
        for param_name, param_value in params.items():
            if param_name in properties:
                expected_type = properties[param_name].get('type')
                if expected_type and not self._check_type(param_value, expected_type):
                    errors.append(f"Parameter {param_name} should be {expected_type}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON schema type."""
        type_map = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, allow it


class FunctionWrapper:
    """Python function/class introspection wrapper."""
    
    def __init__(self):
        """Initialize function wrapper."""
        self.discovered_functions = {}
    
    def wrap_function(self, func: Callable, name: Optional[str] = None) -> Dict[str, Any]:
        """Wrap a Python function as an MCP tool."""
        tool_name = name or func.__name__
        
        # Extract function signature
        sig = inspect.signature(func)
        parameters = self._extract_parameters(sig)
        
        tool_info = {
            'name': tool_name,
            'description': func.__doc__ or f"Python function: {tool_name}",
            'function': func,
            'parameters': parameters,
            'category': 'python_function',
            'wrapper_type': 'function',
            'source_module': func.__module__,
            'source_function': func.__name__
        }
        
        self.discovered_functions[tool_name] = tool_info
        return tool_info
    
    def wrap_module_functions(self, module_name: str, function_names: List[str] = None) -> List[Dict[str, Any]]:
        """Wrap functions from a Python module."""
        try:
            module = importlib.import_module(module_name)
            tools = []
            
            if function_names is None:
                # Auto-discover public functions
                function_names = [name for name in dir(module) 
                                if not name.startswith('_') and callable(getattr(module, name))]
            
            for func_name in function_names:
                if hasattr(module, func_name):
                    func = getattr(module, func_name)
                    if callable(func):
                        tool_name = f"{module_name}.{func_name}"
                        tool_info = self.wrap_function(func, tool_name)
                        tools.append(tool_info)
            
            return tools
            
        except ImportError as e:
            logger.error(f"Failed to import module {module_name}: {e}")
            return []
    
    def _extract_parameters(self, signature: inspect.Signature) -> Dict[str, Any]:
        """Extract parameter schema from function signature."""
        properties = {}
        required = []
        
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue
            
            param_info = {
                'type': 'string',  # Default type
                'description': f'Parameter {param_name}'
            }
            
            # Try to infer type from annotation
            if param.annotation != inspect.Parameter.empty:
                param_info['type'] = self._python_type_to_json_type(param.annotation)
            
            # Check if required (no default value)
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
            else:
                param_info['default'] = param.default
            
            properties[param_name] = param_info
        
        return {
            'type': 'object',
            'properties': properties,
            'required': required
        }
    
    def _python_type_to_json_type(self, python_type) -> str:
        """Convert Python type to JSON schema type."""
        type_map = {
            str: 'string',
            int: 'integer',
            float: 'number',
            bool: 'boolean',
            list: 'array',
            dict: 'object'
        }
        
        return type_map.get(python_type, 'string')


class ModuleWrapper:
    """Automatic Python library wrapper."""
    
    def __init__(self):
        """Initialize module wrapper."""
        self.wrapped_modules = {}
    
    def wrap_library(self, library_name: str, config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Wrap an entire Python library as MCP tools."""
        config = config or {}
        
        try:
            module = importlib.import_module(library_name)
            tools = []
            
            # Get functions to wrap
            include_functions = config.get('functions', [])
            exclude_functions = config.get('exclude_functions', [])
            
            if not include_functions:
                # Auto-discover public functions
                include_functions = [name for name in dir(module) 
                                   if not name.startswith('_') and callable(getattr(module, name))]
            
            for func_name in include_functions:
                if func_name in exclude_functions:
                    continue
                
                if hasattr(module, func_name):
                    func = getattr(module, func_name)
                    if callable(func):
                        tool_name = f"{library_name}.{func_name}"
                        
                        tool_info = {
                            'name': tool_name,
                            'description': func.__doc__ or f"{library_name} function: {func_name}",
                            'function': func,
                            'parameters': self._extract_parameters(func),
                            'category': 'python_library',
                            'wrapper_type': 'module',
                            'library_name': library_name,
                            'function_name': func_name
                        }
                        
                        tools.append(tool_info)
            
            self.wrapped_modules[library_name] = tools
            return tools
            
        except ImportError as e:
            logger.error(f"Failed to wrap library {library_name}: {e}")
            return []
    
    def _extract_parameters(self, func: Callable) -> Dict[str, Any]:
        """Extract parameter schema from function."""
        try:
            sig = inspect.signature(func)
            properties = {}
            required = []
            
            for param_name, param in sig.parameters.items():
                if param_name in ['self', 'cls']:
                    continue
                
                param_info = {
                    'type': 'string',
                    'description': f'Parameter {param_name}'
                }
                
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)
                else:
                    param_info['default'] = param.default
                
                properties[param_name] = param_info
            
            return {
                'type': 'object',
                'properties': properties,
                'required': required
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract parameters for {func}: {e}")
            return {'type': 'object', 'properties': {}}
