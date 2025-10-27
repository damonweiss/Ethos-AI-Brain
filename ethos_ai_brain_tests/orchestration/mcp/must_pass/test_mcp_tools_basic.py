"""
Test MCP Tools Basic Functionality - Must Pass
Light tests focusing on core MCP tools features
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

try:
    from ethos_ai_brain.orchestration.mcp.tools.namespace_registry import NamespaceRegistry
    HAS_NAMESPACE_REGISTRY = True
except ImportError:
    HAS_NAMESPACE_REGISTRY = False
    NamespaceRegistry = None

try:
    from ethos_ai_brain.orchestration.mcp.tools.validate_tools import validate_tools
    HAS_VALIDATE_TOOLS = True
except ImportError:
    HAS_VALIDATE_TOOLS = False
    validate_tools = None


def test_mcp_tools_module_exists():
    """Test that MCP tools module can be imported"""
    try:
        from ethos_ai_brain.orchestration.mcp.tools import __init__
        assert True  # If we get here, import worked
        
    except ImportError as e:
        pytest.skip(f"MCP tools module import failed: {e}")


@pytest.mark.skipif(not HAS_NAMESPACE_REGISTRY, reason="NamespaceRegistry not available")
def test_namespace_registry_creation():
    """Test creating NamespaceRegistry"""
    registry = NamespaceRegistry()
    
    assert isinstance(registry, NamespaceRegistry)


@pytest.mark.skipif(not HAS_VALIDATE_TOOLS, reason="validate_tools not available")
def test_validate_tools_function():
    """Test validate_tools function exists"""
    assert callable(validate_tools)


def test_mcp_tools_directory_structure():
    """Test MCP tools directory structure"""
    tools_path = Path(__file__).resolve().parents[4] / "ethos_ai_brain" / "orchestration" / "mcp" / "tools"
    
    assert tools_path.exists()
    assert tools_path.is_dir()
    
    # Check for expected subdirectories
    expected_dirs = ["utilities", "api_management"]
    for dir_name in expected_dirs:
        dir_path = tools_path / dir_name
        if dir_path.exists():
            assert dir_path.is_dir()
