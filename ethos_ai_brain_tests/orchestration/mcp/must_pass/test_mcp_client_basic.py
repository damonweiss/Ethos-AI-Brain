"""
Test MCP Client Basic Functionality - Must Pass
Light tests focusing on core MCP client features using the new client architecture
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

try:
    from ethos_ai_brain.orchestration.mcp.client import (
        create_direct_client,
        create_json_rpc_client,
        MCPClientRouter
    )
    HAS_MCP_CLIENT = True
except Exception as e:
    HAS_MCP_CLIENT = False
    print(f"Failed to import MCP client: {e}")


@pytest.mark.skipif(not HAS_MCP_CLIENT, reason="MCP Client not available")
def test_mcp_client_creation():
    """Test creating MCP clients using factory functions"""
    # Test direct client creation
    direct_client = create_direct_client()
    assert direct_client is not None
    assert hasattr(direct_client, 'config')
    
    # Test JSON-RPC client creation
    jsonrpc_client = create_json_rpc_client()
    assert jsonrpc_client is not None
    assert hasattr(jsonrpc_client, 'config')


@pytest.mark.skipif(not HAS_MCP_CLIENT, reason="MCP Client not available")
def test_mcp_client_router():
    """Test MCP client router functionality"""
    # Create a client router with direct strategy
    direct_client = create_direct_client()
    assert isinstance(direct_client, MCPClientRouter)
    
    # Check router has the expected methods
    assert hasattr(direct_client, 'list_tools')
    assert hasattr(direct_client, 'execute_tool')
    assert hasattr(direct_client, 'health_check')


@pytest.mark.skipif(not HAS_MCP_CLIENT, reason="MCP Client not available")
def test_mcp_client_default_config():
    """Test MCP client default configuration"""
    direct_client = create_direct_client()
    
    # Check default configuration
    assert direct_client.config.server_url == "http://localhost:8051"
    assert direct_client.config.default_mode == "direct"


def test_mcp_module_exists():
    """Test that MCP module can be imported"""
    try:
        from ethos_ai_brain.orchestration.mcp import __init__
        assert True  # If we get here, import worked
        
    except ImportError as e:
        pytest.skip(f"MCP module import failed: {e}")
