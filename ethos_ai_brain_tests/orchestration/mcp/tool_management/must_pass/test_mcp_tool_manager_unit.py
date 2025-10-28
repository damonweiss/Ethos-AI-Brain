#!/usr/bin/env python3
"""
Unit Tests for MCPToolManager

Tests the core tool discovery and management functionality in isolation.
"""

import pytest
import tempfile
import os
from pathlib import Path

from ethos_ai_brain.orchestration.mcp.tool_management.mcp_tool_manager import MCPToolManager


class TestMCPToolManagerUnit:
    """Unit tests for MCPToolManager."""
    
    def setup_method(self):
        """Set up for each test."""
        # Reset singleton for clean testing
        MCPToolManager.reset_instance()
        
        # Create temporary directory for test tools
        self.temp_dir = tempfile.mkdtemp()
        self.tools_dir = Path(self.temp_dir)
    
    def teardown_method(self):
        """Clean up after each test."""
        MCPToolManager.reset_instance()
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_singleton_pattern(self):
        """Test that MCPToolManager follows singleton pattern."""
        manager1 = MCPToolManager.get_instance(self.tools_dir)
        manager2 = MCPToolManager.get_instance(self.tools_dir)
        
        assert manager1 is manager2
        assert MCPToolManager._instance is manager1
    
    def test_initialization(self):
        """Test MCPToolManager initialization."""
        manager = MCPToolManager(self.tools_dir)
        
        assert manager.tools_dir == self.tools_dir
        assert isinstance(manager.discovered_tools, dict)
        assert len(manager.discovered_tools) == 0
    
    def test_discover_tools_empty_directory(self):
        """Test tool discovery with empty directory."""
        manager = MCPToolManager(self.tools_dir)
        tools = manager.discover_tools()
        
        assert isinstance(tools, list)
        assert len(tools) == 0
    
    def test_discover_tools_with_pyproject_only(self):
        """Test tool discovery with pyproject.toml but no tool.py."""
        # Create a tool directory with pyproject.toml
        tool_dir = self.tools_dir / "test_tool"
        tool_dir.mkdir()
        
        pyproject_content = """
[project]
name = "test_tool"
description = "A test tool"
dependencies = ["requests"]
"""
        (tool_dir / "pyproject.toml").write_text(pyproject_content)
        
        manager = MCPToolManager(self.tools_dir)
        tools = manager.discover_tools()
        
        # Should find the directory but no tools since no tool.py
        assert isinstance(tools, list)
        assert len(tools) == 0
    
    def test_discover_tools_with_complete_tool(self):
        """Test tool discovery with complete tool setup."""
        # Create a tool directory with both pyproject.toml and tool.py
        tool_dir = self.tools_dir / "test_tool"
        tool_dir.mkdir()
        
        pyproject_content = """
[project]
name = "test_tool"
description = "A test tool"
dependencies = ["requests"]
"""
        (tool_dir / "pyproject.toml").write_text(pyproject_content)
        
        tool_py_content = """
def test_function(params):
    return {"message": "Hello from test tool"}

TOOLS = [
    {
        "name": "test_function",
        "description": "A test function",
        "function": test_function,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]
"""
        (tool_dir / "tool.py").write_text(tool_py_content)
        
        manager = MCPToolManager(self.tools_dir)
        tools = manager.discover_tools()
        
        assert isinstance(tools, list)
        assert len(tools) == 1
        
        tool = tools[0]
        assert tool["name"] == "test_function"
        assert tool["description"] == "A test function"
        assert "tool_dir" in tool
        assert "dependencies" in tool
        assert "requests" in tool["dependencies"]
    
    def test_extract_dependencies_project_format(self):
        """Test dependency extraction from project.dependencies."""
        manager = MCPToolManager(self.tools_dir)
        
        pyproject_data = {
            "project": {
                "dependencies": ["requests>=2.0", "beautifulsoup4"]
            }
        }
        
        deps, source = manager._extract_dependencies(pyproject_data)
        
        assert deps == ["requests>=2.0", "beautifulsoup4"]
        assert source == "project.dependencies"
    
    def test_extract_dependencies_tool_uv_format(self):
        """Test dependency extraction from tool.uv.dependencies."""
        manager = MCPToolManager(self.tools_dir)
        
        pyproject_data = {
            "tool": {
                "uv": {
                    "dependencies": ["pandas", "numpy"]
                }
            }
        }
        
        deps, source = manager._extract_dependencies(pyproject_data)
        
        assert deps == ["pandas", "numpy"]
        assert source == "tool.uv.dependencies"
    
    def test_extract_dependencies_priority(self):
        """Test that project.dependencies takes priority over tool.uv.dependencies."""
        manager = MCPToolManager(self.tools_dir)
        
        pyproject_data = {
            "project": {
                "dependencies": ["requests"]
            },
            "tool": {
                "uv": {
                    "dependencies": ["pandas"]
                }
            }
        }
        
        deps, source = manager._extract_dependencies(pyproject_data)
        
        assert deps == ["requests"]
        assert source == "project.dependencies"
    
    def test_assess_complexity(self):
        """Test complexity assessment based on dependencies."""
        manager = MCPToolManager(self.tools_dir)
        
        # Simple (no deps)
        assert manager._assess_complexity([]) == "simple"
        
        # Simple (basic deps)
        assert manager._assess_complexity(["json", "os"]) == "simple"
        
        # Medium (common libs)
        assert manager._assess_complexity(["requests", "beautifulsoup4"]) == "medium"
        
        # Complex (heavy libs)
        assert manager._assess_complexity(["torch", "tensorflow"]) == "complex"
    
    def test_determine_namespace_from_metadata(self):
        """Test namespace determination from tool metadata."""
        manager = MCPToolManager(self.tools_dir)
        
        tool_info = {"namespace": "custom.namespace"}
        tool_dir = self.tools_dir / "test_tool"
        
        namespace = manager._determine_namespace(tool_info, tool_dir)
        
        assert namespace == "custom.namespace"
    
    def test_determine_namespace_from_directory(self):
        """Test namespace determination from directory structure."""
        manager = MCPToolManager(self.tools_dir)
        
        # Create nested directory structure
        nested_dir = self.tools_dir / "utilities" / "web" / "scraper"
        nested_dir.mkdir(parents=True)
        
        tool_info = {}  # No explicit namespace
        
        namespace = manager._determine_namespace(tool_info, nested_dir)
        
        assert namespace == "utilities.web"
    
    def test_determine_namespace_default(self):
        """Test default namespace for root level tools."""
        manager = MCPToolManager(self.tools_dir)
        
        tool_dir = self.tools_dir / "simple_tool"
        tool_info = {}  # No explicit namespace
        
        namespace = manager._determine_namespace(tool_info, tool_dir)
        
        assert namespace == "default"
    
    def test_execute_tool_not_found(self):
        """Test executing a tool that doesn't exist."""
        manager = MCPToolManager(self.tools_dir)
        
        result = manager.execute_tool("nonexistent_tool", param="value")
        
        assert result["success"] is False
        assert "not found" in result["error"].lower()
        assert "available_tools" in result
    
    def test_execute_tool_success(self):
        """Test successful tool execution with real function."""
        # Create a real test function
        def test_function(params):
            return {"message": "success", "received_params": params}
        
        manager = MCPToolManager(self.tools_dir)
        
        # Add a real tool with function to discovered_tools
        manager.discovered_tools["test_tool"] = {
            "name": "test_tool",
            "function": test_function,
            "tool_dir": str(self.tools_dir / "test_tool")
        }
        
        result = manager.execute_tool("test_tool", param="value")
        
        assert result["success"] is True
        assert result["result"]["message"] == "success"
        assert result["result"]["received_params"]["param"] == "value"
        assert result["execution_method"] == "direct_function"
    
    def test_execute_tool_failure(self):
        """Test failed tool execution with real function that raises exception."""
        # Create a test function that raises an exception
        def failing_function(params):
            raise ValueError("Tool execution failed")
        
        manager = MCPToolManager(self.tools_dir)
        
        # Add a real tool with failing function to discovered_tools
        manager.discovered_tools["test_tool"] = {
            "name": "test_tool",
            "function": failing_function,
            "tool_dir": str(self.tools_dir / "test_tool")
        }
        
        result = manager.execute_tool("test_tool")
        
        assert result["success"] is False
        assert "function execution failed" in result["error"].lower()
        assert "ValueError" in result["error"]
        assert "Tool execution failed" in result["error"]
    
    def test_skip_shared_envs(self):
        """Test that _shared_envs directories are skipped."""
        # Create _shared_envs directory with pyproject.toml
        shared_env_dir = self.tools_dir / "_shared_envs" / "test"
        shared_env_dir.mkdir(parents=True)
        
        pyproject_content = """
[project]
name = "shared_env_tool"
"""
        (shared_env_dir / "pyproject.toml").write_text(pyproject_content)
        
        manager = MCPToolManager(self.tools_dir)
        tools = manager.discover_tools()
        
        # Should not find any tools from _shared_envs
        assert len(tools) == 0
