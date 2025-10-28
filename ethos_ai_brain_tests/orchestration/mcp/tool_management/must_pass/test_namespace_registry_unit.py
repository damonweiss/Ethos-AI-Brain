#!/usr/bin/env python3
"""
Unit Tests for Namespace Registry

Tests the namespace registry functionality in isolation.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from ethos_ai_brain.orchestration.mcp.tool_management.namespace_registry import (
    ConfigurableNamespaceRegistry,
    get_namespaced_tool_name,
    parse_namespaced_tool_name,
    namespace_registry
)


class TestNamespaceRegistryUnit:
    """Unit tests for namespace registry functionality."""
    
    def setup_method(self):
        """Set up for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_namespaces.yaml"
    
    def teardown_method(self):
        """Clean up after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_config(self, config_data):
        """Helper to create a test config file."""
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)
    
    def test_default_config_when_file_missing(self):
        """Test that default config is used when file is missing."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.yaml"
        registry = ConfigurableNamespaceRegistry(str(nonexistent_file))
        
        namespaces = registry.list_namespaces()
        assert "default" in namespaces
        
        default_info = registry.get_namespace_info("default")
        assert default_info is not None
        assert "description" in default_info
    
    def test_load_custom_config(self):
        """Test loading custom namespace configuration."""
        config_data = {
            "namespaces": {
                "utilities": {
                    "description": "Utility tools for common tasks",
                    "task_patterns": [["file", "system", "utility"]]
                },
                "ai": {
                    "description": "AI and machine learning tools",
                    "task_patterns": [["ai", "ml", "model"]]
                }
            }
        }
        self.create_test_config(config_data)
        
        registry = ConfigurableNamespaceRegistry(str(self.config_file))
        
        namespaces = registry.list_namespaces()
        assert "utilities" in namespaces
        assert "ai" in namespaces
        
        utilities_info = registry.get_namespace_info("utilities")
        assert utilities_info["description"] == "Utility tools for common tasks"
        assert utilities_info["task_patterns"] == [["file", "system", "utility"]]
    
    def test_register_tool_valid_namespace(self):
        """Test registering a tool with valid namespace."""
        config_data = {
            "namespaces": {
                "web": {
                    "description": "Web-related tools",
                    "task_patterns": [["web", "http", "scraping"]]
                }
            }
        }
        self.create_test_config(config_data)
        
        registry = ConfigurableNamespaceRegistry(str(self.config_file))
        
        tool_info = {
            "name": "web_scraper",
            "namespace": "web",
            "category": "scraping",
            "description": "Scrapes web pages"
        }
        
        registry.register_tool(tool_info)
        
        web_tools = registry.get_tools_by_namespace("web")
        assert len(web_tools) == 1
        assert web_tools[0]["name"] == "web_scraper"
        
        scraping_tools = registry.get_tools_by_category("scraping")
        assert len(scraping_tools) == 1
        assert scraping_tools[0]["name"] == "web_scraper"
    
    def test_register_tool_invalid_namespace(self):
        """Test registering a tool with invalid namespace falls back to default."""
        config_data = {
            "namespaces": {
                "valid": {
                    "description": "Valid namespace"
                }
            }
        }
        self.create_test_config(config_data)
        
        registry = ConfigurableNamespaceRegistry(str(self.config_file))
        
        tool_info = {
            "name": "test_tool",
            "namespace": "invalid_namespace",
            "category": "general"
        }
        
        registry.register_tool(tool_info)
        
        # Should be registered under default namespace
        default_tools = registry.get_tools_by_namespace("default")
        assert len(default_tools) == 1
        assert default_tools[0]["name"] == "test_tool"
    
    def test_register_tool_no_namespace(self):
        """Test registering a tool without explicit namespace."""
        registry = ConfigurableNamespaceRegistry(str(self.config_file))
        
        tool_info = {
            "name": "simple_tool",
            "category": "utility"
        }
        
        registry.register_tool(tool_info)
        
        default_tools = registry.get_tools_by_namespace("default")
        assert len(default_tools) == 1
        assert default_tools[0]["name"] == "simple_tool"
        
        utility_tools = registry.get_tools_by_category("utility")
        assert len(utility_tools) == 1
        assert utility_tools[0]["name"] == "simple_tool"
    
    def test_get_task_patterns(self):
        """Test retrieving task patterns for namespaces."""
        config_data = {
            "namespaces": {
                "ai": {
                    "description": "AI tools",
                    "task_patterns": [
                        ["machine", "learning", "model"],
                        ["neural", "network", "training"]
                    ]
                }
            }
        }
        self.create_test_config(config_data)
        
        registry = ConfigurableNamespaceRegistry(str(self.config_file))
        
        patterns = registry.get_task_patterns("ai")
        assert len(patterns) == 2
        assert ["machine", "learning", "model"] in patterns
        assert ["neural", "network", "training"] in patterns
        
        # Non-existent namespace should return empty list
        empty_patterns = registry.get_task_patterns("nonexistent")
        assert empty_patterns == []
    
    def test_get_tool_capabilities(self):
        """Test getting comprehensive tool capabilities summary."""
        config_data = {
            "namespaces": {
                "web": {"description": "Web tools"},
                "ai": {"description": "AI tools"}
            }
        }
        self.create_test_config(config_data)
        
        registry = ConfigurableNamespaceRegistry(str(self.config_file))
        
        # Register some tools
        registry.register_tool({"name": "scraper", "namespace": "web", "category": "scraping"})
        registry.register_tool({"name": "classifier", "namespace": "ai", "category": "ml"})
        registry.register_tool({"name": "validator", "namespace": "web", "category": "validation"})
        
        capabilities = registry.get_tool_capabilities()
        
        assert "namespaces" in capabilities
        assert "categories" in capabilities
        assert "total_tools" in capabilities
        
        # Check namespace summary
        assert "web" in capabilities["namespaces"]
        assert capabilities["namespaces"]["web"]["tool_count"] == 2
        assert "scraper" in capabilities["namespaces"]["web"]["tools"]
        assert "validator" in capabilities["namespaces"]["web"]["tools"]
        
        assert "ai" in capabilities["namespaces"]
        assert capabilities["namespaces"]["ai"]["tool_count"] == 1
        assert "classifier" in capabilities["namespaces"]["ai"]["tools"]
        
        # Check category summary
        assert capabilities["categories"]["scraping"] == 1
        assert capabilities["categories"]["ml"] == 1
        assert capabilities["categories"]["validation"] == 1
        
        # Check total
        assert capabilities["total_tools"] == 3
    
    def test_empty_registry(self):
        """Test behavior with empty registry."""
        registry = ConfigurableNamespaceRegistry(str(self.config_file))
        
        capabilities = registry.get_tool_capabilities()
        
        assert capabilities["total_tools"] == 0
        assert len(capabilities["namespaces"]) >= 1  # At least default
        assert len(capabilities["categories"]) == 0


class TestNamespaceUtilityFunctions:
    """Unit tests for namespace utility functions."""
    
    def test_get_namespaced_tool_name(self):
        """Test creating namespaced tool names."""
        # Default namespace should not add prefix
        assert get_namespaced_tool_name("default", "tool") == "tool"
        
        # Other namespaces should add prefix
        assert get_namespaced_tool_name("web", "scraper") == "web/scraper"
        assert get_namespaced_tool_name("ai.ml", "classifier") == "ai.ml/classifier"
    
    def test_parse_namespaced_tool_name(self):
        """Test parsing namespaced tool names."""
        # Simple tool name (no namespace)
        namespace, name = parse_namespaced_tool_name("simple_tool")
        assert namespace == "default"
        assert name == "simple_tool"
        
        # Namespaced tool name
        namespace, name = parse_namespaced_tool_name("web/scraper")
        assert namespace == "web"
        assert name == "scraper"
        
        # Complex namespace
        namespace, name = parse_namespaced_tool_name("ai.ml.vision/image_classifier")
        assert namespace == "ai.ml.vision"
        assert name == "image_classifier"
        
        # Multiple slashes (only first one counts)
        namespace, name = parse_namespaced_tool_name("web/api/rest/client")
        assert namespace == "web"
        assert name == "api/rest/client"


class TestGlobalNamespaceRegistry:
    """Test the global namespace registry instance."""
    
    def test_global_registry_exists(self):
        """Test that global registry instance exists and works."""
        # The global registry should be available
        assert namespace_registry is not None
        
        # Should have at least default namespace
        namespaces = namespace_registry.list_namespaces()
        assert "default" in namespaces
        
        # Should be able to register tools
        test_tool = {"name": "global_test", "category": "test"}
        namespace_registry.register_tool(test_tool)
        
        default_tools = namespace_registry.get_tools_by_namespace("default")
        tool_names = [tool["name"] for tool in default_tools]
        assert "global_test" in tool_names
