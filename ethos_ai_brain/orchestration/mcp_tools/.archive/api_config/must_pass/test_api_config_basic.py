"""
Test API Configuration Tool - Must Pass
Basic functionality test for the API configuration MCP tool.
"""

import pytest
import importlib.util
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[5]


class TestAPIConfigTool:
    """Test API configuration tool functionality."""
    
    def test_api_config_import(self):
        """Test that API config tool can be imported."""
        api_config_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "api_management" / "api_config" / "tool.py"
        spec = importlib.util.spec_from_file_location("api_config_tool", api_config_path)
        api_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api_config_module)
        
        assert hasattr(api_config_module, 'list_api_configs')
        assert callable(api_config_module.list_api_configs)
    
    def test_list_api_configs_no_directory(self):
        """Test list_api_configs handles missing directory gracefully."""
        api_config_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "api_management" / "api_config" / "tool.py"
        spec = importlib.util.spec_from_file_location("api_config_tool", api_config_path)
        api_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api_config_module)
        
        # Test without config_dir (should handle gracefully)
        result = api_config_module.list_api_configs({})
        assert isinstance(result, dict)
        assert "error" in result
        assert "api_configs" in result
        assert "count" in result
        assert result["count"] == 0
        assert isinstance(result["api_configs"], list)
    
    def test_list_api_configs_with_directory(self):
        """Test list_api_configs with actual config directory."""
        api_config_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "api_management" / "api_config" / "tool.py"
        spec = importlib.util.spec_from_file_location("api_config_tool", api_config_path)
        api_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api_config_module)
        
        # Test with actual config directory
        config_dir = project_root / "ethos_ai_brain" / "orchestration" / "mcp" / "config" / "api_configs"
        result = api_config_module.list_api_configs({}, config_dir=config_dir)
        
        assert isinstance(result, dict)
        assert "api_configs" in result
        assert "count" in result
        assert "config_dir" in result
        assert isinstance(result["api_configs"], list)
        assert isinstance(result["count"], int)
        
        # Should either find configs or report empty directory (both valid)
        if result["count"] > 0:
            assert len(result["api_configs"]) == result["count"]
        else:
            assert len(result["api_configs"]) == 0
