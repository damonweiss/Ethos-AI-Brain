"""
Test Conventional MCP Tools - Must Pass
Tests for basic MCP tools that don't require reasoning dependencies.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))


class TestSystemTools:
    """Test basic system tools with no external dependencies."""
    
    def test_echo_message_import(self):
        """Test that echo_message can be imported."""
        sys.path.insert(0, str(project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "system" / "system_tools"))
        from tool import echo_message
        assert callable(echo_message)
    
    def test_echo_message_functionality(self):
        """Test echo_message basic functionality."""
        sys.path.insert(0, str(project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "system" / "system_tools"))
        from tool import echo_message
        
        # Test with message
        result = echo_message({"message": "Hello MCP!"})
        assert "echo" in result
        assert result["echo"] == "Hello MCP!"
        assert "timestamp" in result
        assert "processed_by" in result
        
        # Test with no message
        result = echo_message({})
        assert "echo" in result
        assert result["echo"] == "No message provided"
    
    def test_get_system_time_import(self):
        """Test that get_system_time can be imported."""
        sys.path.insert(0, str(project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "system" / "system_tools"))
        from tool import get_system_time
        assert callable(get_system_time)
    
    def test_get_system_time_functionality(self):
        """Test get_system_time basic functionality."""
        sys.path.insert(0, str(project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "system" / "system_tools"))
        from tool import get_system_time
        
        result = get_system_time({})
        assert "timestamp" in result
        assert "unix_timestamp" in result
        assert "server" in result
        assert "timezone" in result
        assert "utc_timestamp" in result
        
        # Verify timestamp format
        assert isinstance(result["unix_timestamp"], (int, float))
        assert result["server"] == "ethos-mcp-server"


class TestWebTools:
    """Test web tools with external dependencies."""
    
    def test_web_scraper_import(self):
        """Test that web scraper can be imported."""
        import importlib.util
        
        web_scraper_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "web" / "web_scraper" / "tool.py"
        spec = importlib.util.spec_from_file_location("web_scraper_tool", web_scraper_path)
        web_scraper_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(web_scraper_module)
        
        assert hasattr(web_scraper_module, 'scrape_webpage')
        assert callable(web_scraper_module.scrape_webpage)
    
    def test_web_scraper_missing_dependencies(self):
        """Test web scraper handles missing dependencies gracefully."""
        import importlib.util
        
        web_scraper_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "web" / "web_scraper" / "tool.py"
        spec = importlib.util.spec_from_file_location("web_scraper_tool", web_scraper_path)
        web_scraper_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(web_scraper_module)
        
        # This should handle missing dependencies gracefully
        result = web_scraper_module.scrape_webpage({"url": "https://example.com"})
        assert isinstance(result, dict)
        # Should either work or return error message
        assert "error" in result or "content" in result
    
    @pytest.mark.slow
    def test_web_scraper_with_real_request(self):
        """Test web scraper with real HTTP request (slow test)."""
        import importlib.util
        
        web_scraper_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "web" / "web_scraper" / "tool.py"
        spec = importlib.util.spec_from_file_location("web_scraper_tool", web_scraper_path)
        web_scraper_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(web_scraper_module)
        
        # Test with httpbin (reliable test endpoint)
        result = web_scraper_module.scrape_webpage({"url": "https://httpbin.org/html"})
        
        if "error" not in result:
            # If no error, should have successful response
            assert "status_code" in result
            assert result["status_code"] == 200
            assert "content" in result
        else:
            # If error (missing deps), should be graceful
            error_msg = result["error"].lower()
            assert ("dependencies" in error_msg or 
                    "import" in error_msg or 
                    "lxml" in error_msg or 
                    "parser" in error_msg), f"Unexpected error: {result['error']}"


class TestToolStructure:
    """Test tool structure and organization."""
    
    def test_system_tools_directory_exists(self):
        """Test that system tools directory exists."""
        system_tools_dir = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "system" / "system_tools"
        assert system_tools_dir.exists()
        assert system_tools_dir.is_dir()
    
    def test_web_tools_directory_exists(self):
        """Test that web tools directory exists."""
        web_tools_dir = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "web" / "web_scraper"
        assert web_tools_dir.exists()
        assert web_tools_dir.is_dir()
    
    def test_tool_files_exist(self):
        """Test that tool.py files exist in expected locations."""
        system_tool_file = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "system" / "system_tools" / "tool.py"
        web_tool_file = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "web" / "web_scraper" / "tool.py"
        
        assert system_tool_file.exists()
        assert web_tool_file.exists()
    
    def test_pyproject_files_exist(self):
        """Test that pyproject.toml files exist for UV management."""
        system_pyproject = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "system" / "system_tools" / "pyproject.toml"
        web_pyproject = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "web" / "web_scraper" / "pyproject.toml"
        
        assert system_pyproject.exists()
        assert web_pyproject.exists()
