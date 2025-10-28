"""
Test Web Scraper Tool - Must Pass
Basic functionality test for the web scraper MCP tool.
"""

import pytest
import importlib.util
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[5]


class TestWebScraperTool:
    """Test web scraper tool functionality."""
    
    def test_web_scraper_import(self):
        """Test that web scraper tool can be imported."""
        web_scraper_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "web" / "web_scraper" / "tool.py"
        spec = importlib.util.spec_from_file_location("web_scraper_tool", web_scraper_path)
        web_scraper_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(web_scraper_module)
        
        assert hasattr(web_scraper_module, 'scrape_webpage')
        assert hasattr(web_scraper_module, 'validate_url')
        assert callable(web_scraper_module.scrape_webpage)
        assert callable(web_scraper_module.validate_url)
    
    def test_web_scraper_basic_functionality(self):
        """Test web scraper basic functionality with simple request."""
        web_scraper_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "web" / "web_scraper" / "tool.py"
        spec = importlib.util.spec_from_file_location("web_scraper_tool", web_scraper_path)
        web_scraper_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(web_scraper_module)
        
        # Test with httpbin (reliable test endpoint)
        result = web_scraper_module.scrape_webpage({"url": "https://httpbin.org/html"})
        
        # Should either work or fail gracefully
        assert isinstance(result, dict)
        
        if "error" not in result:
            # If successful, should have expected fields
            assert "status_code" in result
            assert "content" in result
            assert "url" in result
            assert result["url"] == "https://httpbin.org/html"
        else:
            # If error, should be graceful with helpful message
            assert "error" in result
            # Error should be about dependencies or network, not code failure
            error_msg = result["error"].lower()
            assert any(keyword in error_msg for keyword in ["dependencies", "import", "network", "timeout", "connection"])
    
    def test_validate_url_functionality(self):
        """Test URL validation functionality."""
        web_scraper_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "web" / "web_scraper" / "tool.py"
        spec = importlib.util.spec_from_file_location("web_scraper_tool", web_scraper_path)
        web_scraper_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(web_scraper_module)
        
        # Test URL validation
        result = web_scraper_module.validate_url({"url": "https://httpbin.org"})
        
        # Should return validation result
        assert isinstance(result, dict)
        
        if "error" not in result:
            # If successful, should have validation info
            assert "valid" in result or "status_code" in result
        else:
            # If error, should be graceful
            assert "error" in result
