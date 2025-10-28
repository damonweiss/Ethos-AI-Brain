#!/usr/bin/env python3
"""
Unit Tests for Tool Validation

Tests the tool validation functionality in isolation.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from ethos_ai_brain.orchestration.mcp.tool_management.validate_tools import (
    validate_tool_metadata,
    generate_tool_documentation
)


class TestToolValidationUnit:
    """Unit tests for tool validation functionality."""
    
    def test_validate_complete_tool(self):
        """Test validation of a complete, valid tool."""
        tool = {
            "name": "test_tool",
            "description": "A test tool for validation",
            "function": lambda x: x,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Input parameter"
                    }
                },
                "required": ["input"]
            },
            "tags": ["test", "validation"],
            "author": "Test Author",
            "version": "1.0.0",
            "category": "testing",
            "examples": [
                {
                    "description": "Basic usage",
                    "input": {"input": "test"},
                    "output": {"result": "test"}
                }
            ]
        }
        
        issues = validate_tool_metadata(tool)
        
        assert len(issues) == 0
    
    def test_validate_minimal_tool(self):
        """Test validation of a tool with only required fields."""
        tool = {
            "name": "minimal_tool",
            "description": "A minimal tool",
            "function": lambda x: x,
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
        
        issues = validate_tool_metadata(tool)
        
        # Should have issues for missing recommended fields
        recommended_missing = [issue for issue in issues if "Missing recommended field" in issue]
        assert len(recommended_missing) == 5  # tags, author, version, category, examples
        
        # But no issues for missing required fields
        required_missing = [issue for issue in issues if "Missing required field" in issue]
        assert len(required_missing) == 0
    
    def test_validate_missing_required_fields(self):
        """Test validation of tool missing required fields."""
        tool = {
            "name": "incomplete_tool",
            # Missing: description, function, parameters
        }
        
        issues = validate_tool_metadata(tool)
        
        required_issues = [issue for issue in issues if "Missing required field" in issue]
        assert len(required_issues) == 3
        
        required_fields = ["description", "function", "parameters"]
        for field in required_fields:
            assert any(field in issue for issue in required_issues)
    
    def test_validate_invalid_parameters(self):
        """Test validation of tool with invalid parameters structure."""
        tool = {
            "name": "bad_params_tool",
            "description": "Tool with bad parameters",
            "function": lambda x: x,
            "parameters": "invalid_parameters"  # Should be dict
        }
        
        issues = validate_tool_metadata(tool)
        
        param_issues = [issue for issue in issues if "Parameters should be" in issue]
        assert len(param_issues) == 1
        assert "dictionary" in param_issues[0]
    
    def test_validate_parameters_missing_type(self):
        """Test validation of parameters missing type field."""
        tool = {
            "name": "no_type_tool",
            "description": "Tool with parameters missing type",
            "function": lambda x: x,
            "parameters": {
                "properties": {"input": {"type": "string"}}
                # Missing: type
            }
        }
        
        issues = validate_tool_metadata(tool)
        
        type_issues = [issue for issue in issues if "should have 'type' field" in issue]
        assert len(type_issues) == 1
    
    def test_validate_parameters_missing_properties(self):
        """Test validation of parameters missing properties field."""
        tool = {
            "name": "no_props_tool",
            "description": "Tool with parameters missing properties",
            "function": lambda x: x,
            "parameters": {
                "type": "object"
                # Missing: properties
            }
        }
        
        issues = validate_tool_metadata(tool)
        
        props_issues = [issue for issue in issues if "should have 'properties' field" in issue]
        assert len(props_issues) == 1
    
    def test_validate_examples_missing_fields(self):
        """Test validation of examples with missing fields."""
        tool = {
            "name": "bad_examples_tool",
            "description": "Tool with incomplete examples",
            "function": lambda x: x,
            "parameters": {"type": "object", "properties": {}},
            "examples": [
                {
                    "description": "Example with missing input/output"
                    # Missing: input, output
                },
                {
                    "input": {"test": "value"},
                    "output": {"result": "value"}
                    # Missing: description
                }
            ]
        }
        
        issues = validate_tool_metadata(tool)
        
        example_issues = [issue for issue in issues if "Example" in issue]
        assert len(example_issues) == 3  # Example 0 missing input/output, Example 1 missing description
        
        # Check specific issues
        assert any("Example 0 missing input" in issue for issue in example_issues)
        assert any("Example 0 missing output" in issue for issue in example_issues)
        assert any("Example 1 missing description" in issue for issue in example_issues)


class TestToolDocumentationGeneration:
    """Unit tests for tool documentation generation."""
    
    def test_generate_documentation_single_tool(self):
        """Test generating documentation for a single tool."""
        tools = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "category": "testing",
                "tags": ["test", "demo"],
                "author": "Test Author",
                "version": "1.0.0",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Input parameter"
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of iterations"
                        }
                    }
                },
                "examples": [
                    {
                        "description": "Basic usage",
                        "input": {"input": "test", "count": 1},
                        "output": {"result": "test"}
                    }
                ]
            }
        ]
        
        doc = generate_tool_documentation(tools)
        
        # Check basic structure
        assert "# MCP Tools Documentation" in doc
        assert "Total tools: 1" in doc
        assert "## Testing Tools" in doc
        assert "### test_tool" in doc
        
        # Check content
        assert "**Description:** A test tool" in doc
        assert "**Tags:** test, demo" in doc
        assert "**Author:** Test Author" in doc
        assert "**Version:** 1.0.0" in doc
        
        # Check parameters
        assert "**Parameters:**" in doc
        assert "- `input` (string): Input parameter" in doc
        assert "- `count` (integer): Number of iterations" in doc
        
        # Check examples
        assert "**Examples:**" in doc
        assert "*Basic usage*" in doc
        assert "Input: `{'input': 'test', 'count': 1}`" in doc
        assert "Output: `{'result': 'test'}`" in doc
    
    def test_generate_documentation_multiple_categories(self):
        """Test generating documentation for tools in multiple categories."""
        tools = [
            {
                "name": "web_tool",
                "description": "Web scraping tool",
                "category": "web"
            },
            {
                "name": "ai_tool",
                "description": "AI processing tool",
                "category": "ai"
            },
            {
                "name": "util_tool",
                "description": "Utility tool",
                "category": "utilities"
            }
        ]
        
        doc = generate_tool_documentation(tools)
        
        # Check that all categories are present
        assert "## Web Tools" in doc
        assert "## Ai Tools" in doc
        assert "## Utilities Tools" in doc
        
        # Check that all tools are documented
        assert "### web_tool" in doc
        assert "### ai_tool" in doc
        assert "### util_tool" in doc
        
        assert "Total tools: 3" in doc
    
    def test_generate_documentation_uncategorized_tools(self):
        """Test generating documentation for uncategorized tools."""
        tools = [
            {
                "name": "mystery_tool",
                "description": "Tool without category"
                # No category field
            }
        ]
        
        doc = generate_tool_documentation(tools)
        
        # Should be placed in "uncategorized" category
        assert "## Uncategorized Tools" in doc
        assert "### mystery_tool" in doc
    
    def test_generate_documentation_minimal_tool(self):
        """Test generating documentation for tool with minimal metadata."""
        tools = [
            {
                "name": "minimal_tool",
                "description": "Minimal tool"
                # Only required fields
            }
        ]
        
        doc = generate_tool_documentation(tools)
        
        # Should still generate basic documentation
        assert "### minimal_tool" in doc
        assert "**Description:** Minimal tool" in doc
        
        # Should not have sections for missing optional fields
        assert "**Tags:**" not in doc
        assert "**Author:**" not in doc
        assert "**Parameters:**" not in doc
        assert "**Examples:**" not in doc
    
    def test_generate_documentation_empty_list(self):
        """Test generating documentation for empty tool list."""
        tools = []
        
        doc = generate_tool_documentation(tools)
        
        assert "# MCP Tools Documentation" in doc
        assert "Total tools: 0" in doc
        
        # Should not have any tool categories
        assert "##" not in doc.split('\n')[2:]  # Skip the main header


class TestMainFunction:
    """Unit tests for the main validation function."""
    
    @patch('ethos_ai_brain.orchestration.mcp.tool_management.validate_tools.MCPToolManager')
    def test_main_function_with_tools(self, mock_manager_class):
        """Test main function with discovered tools."""
        # Mock the tool manager
        mock_manager = MagicMock()
        mock_manager.discover_tools.return_value = [
            {
                "name": "test_tool",
                "description": "Test tool",
                "function": lambda x: x,
                "parameters": {"type": "object", "properties": {}}
            }
        ]
        mock_manager_class.get_instance.return_value = mock_manager
        
        # Import and run main (we can't easily test the actual main() due to print statements)
        from ethos_ai_brain.orchestration.mcp.tool_management.validate_tools import main
        
        # This would normally print output, but we're just testing it doesn't crash
        try:
            # We can't easily capture print output in unit tests, so just ensure no exceptions
            with patch('builtins.print'):
                with patch('pathlib.Path.write_text'):
                    main()
        except SystemExit:
            pass  # main() might call sys.exit, which is fine
    
    @patch('ethos_ai_brain.orchestration.mcp.tool_management.validate_tools.MCPToolManager')
    def test_main_function_no_tools(self, mock_manager_class):
        """Test main function with no discovered tools."""
        # Mock the tool manager to return no tools
        mock_manager = MagicMock()
        mock_manager.discover_tools.return_value = []
        mock_manager_class.get_instance.return_value = mock_manager
        
        from ethos_ai_brain.orchestration.mcp.tool_management.validate_tools import main
        
        try:
            with patch('builtins.print'):
                with patch('pathlib.Path.write_text'):
                    main()
        except SystemExit:
            pass
