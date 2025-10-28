"""
Test System Tools - Must Pass
Basic functionality test for the system tools MCP tool.
"""

import pytest
import importlib.util
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[5]


class TestSystemTools:
    """Test system tools functionality."""
    
    def test_system_tools_import(self):
        """Test that system tools can be imported."""
        system_tools_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "system" / "system_tools" / "tool.py"
        spec = importlib.util.spec_from_file_location("system_tools", system_tools_path)
        system_tools_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(system_tools_module)
        
        assert hasattr(system_tools_module, 'echo_message')
        assert hasattr(system_tools_module, 'get_system_time')
        assert callable(system_tools_module.echo_message)
        assert callable(system_tools_module.get_system_time)
    
    def test_echo_message_functionality(self):
        """Test echo message functionality."""
        system_tools_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "system" / "system_tools" / "tool.py"
        spec = importlib.util.spec_from_file_location("system_tools", system_tools_path)
        system_tools_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(system_tools_module)
        
        # Test with message
        result = system_tools_module.echo_message({"message": "Hello MCP!"})
        assert isinstance(result, dict)
        assert "echo" in result
        assert result["echo"] == "Hello MCP!"
        assert "timestamp" in result
        assert "processed_by" in result
        
        # Test without message
        result = system_tools_module.echo_message({})
        assert isinstance(result, dict)
        assert "echo" in result
        assert result["echo"] == "No message provided"
    
    def test_get_system_time_functionality(self):
        """Test get system time functionality."""
        system_tools_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "system" / "system_tools" / "tool.py"
        spec = importlib.util.spec_from_file_location("system_tools", system_tools_path)
        system_tools_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(system_tools_module)
        
        result = system_tools_module.get_system_time({})
        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "unix_timestamp" in result
        assert "server" in result
        assert "timezone" in result
        assert "utc_timestamp" in result
        
        # Verify timestamp format
        assert isinstance(result["unix_timestamp"], (int, float))
        assert result["server"] == "ethos-mcp-server"
