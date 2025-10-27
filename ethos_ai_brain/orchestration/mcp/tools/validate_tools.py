#!/usr/bin/env python3
"""
Tool Validation Utility

Validates tool metadata and provides insights into the tool ecosystem.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# from mcp_tools import discover_tools
from . import discover_tools


def validate_tool_metadata(tool: Dict[str, Any]) -> List[str]:
    """
    Validate tool metadata completeness and consistency.
    
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    required_fields = ['name', 'description', 'function', 'parameters']
    recommended_fields = ['tags', 'author', 'version', 'category', 'examples']
    
    # Check required fields
    for field in required_fields:
        if field not in tool:
            issues.append(f"Missing required field: {field}")
    
    # Check recommended fields
    for field in recommended_fields:
        if field not in tool:
            issues.append(f"Missing recommended field: {field}")
    
    # Validate parameter schema
    if 'parameters' in tool:
        params = tool['parameters']
        if isinstance(params, dict):
            if 'type' not in params:
                issues.append("Parameters should have 'type' field")
            if 'properties' not in params:
                issues.append("Parameters should have 'properties' field")
        else:
            issues.append("Parameters should be a dictionary")
    
    # Validate examples
    if 'examples' in tool and tool['examples']:
        for i, example in enumerate(tool['examples']):
            if 'description' not in example:
                issues.append(f"Example {i} missing description")
            if 'input' not in example:
                issues.append(f"Example {i} missing input")
            if 'output' not in example:
                issues.append(f"Example {i} missing output")
    
    return issues


def generate_tool_documentation(tools: List[Dict[str, Any]]) -> str:
    """Generate markdown documentation for all tools."""
    doc = "# MCP Tools Documentation\n\n"
    doc += f"Total tools: {len(tools)}\n\n"
    
    # Group by category
    categories = {}
    for tool in tools:
        category = tool.get('category', 'uncategorized')
        if category not in categories:
            categories[category] = []
        categories[category].append(tool)
    
    for category, category_tools in categories.items():
        doc += f"## {category.title()} Tools\n\n"
        
        for tool in category_tools:
            doc += f"### {tool['name']}\n\n"
            doc += f"**Description:** {tool['description']}\n\n"
            
            if 'tags' in tool:
                doc += f"**Tags:** {', '.join(tool['tags'])}\n\n"
            
            if 'author' in tool:
                doc += f"**Author:** {tool['author']}\n\n"
            
            if 'version' in tool:
                doc += f"**Version:** {tool['version']}\n\n"
            
            # Parameters
            if 'parameters' in tool and tool['parameters'].get('properties'):
                doc += "**Parameters:**\n\n"
                for param_name, param_info in tool['parameters']['properties'].items():
                    doc += f"- `{param_name}` ({param_info.get('type', 'unknown')}): {param_info.get('description', 'No description')}\n"
                doc += "\n"
            
            # Examples
            if 'examples' in tool and tool['examples']:
                doc += "**Examples:**\n\n"
                for example in tool['examples']:
                    doc += f"*{example['description']}*\n\n"
                    doc += f"Input: `{example['input']}`\n\n"
                    doc += f"Output: `{example['output']}`\n\n"
            
            doc += "---\n\n"
    
    return doc


def main():
    """Main validation and documentation generation."""
    print("🔍 Discovering MCP Tools...")
    tools = discover_tools()
    print(f"Found {len(tools)} tools\n")
    
    # Validate each tool
    all_valid = True
    for tool in tools:
        print(f"Validating: {tool['name']}")
        issues = validate_tool_metadata(tool)
        
        if issues:
            all_valid = False
            print(f"  Issues found:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"  Valid")
        print()
    
    if all_valid:
        print("All tools have valid metadata!")
    else:
        print("Some tools have metadata issues")
    
    # Generate documentation
    print("\nGenerating documentation...")
    doc = generate_tool_documentation(tools)
    
    doc_path = Path(__file__).parent / "TOOLS_DOCUMENTATION.md"
    doc_path.write_text(doc)
    print(f"Documentation written to: {doc_path}")
    
    # Summary statistics
    print(f"\nTool Statistics:")
    categories = {}
    tags = set()
    authors = set()
    
    for tool in tools:
        category = tool.get('category', 'uncategorized')
        categories[category] = categories.get(category, 0) + 1
        
        if 'tags' in tool:
            tags.update(tool['tags'])
        
        if 'author' in tool:
            authors.add(tool['author'])
    
    print(f"Categories: {dict(categories)}")
    print(f"Unique tags: {len(tags)} ({', '.join(sorted(tags))})")
    print(f"Authors: {len(authors)} ({', '.join(sorted(authors))})")


if __name__ == "__main__":
    main()
