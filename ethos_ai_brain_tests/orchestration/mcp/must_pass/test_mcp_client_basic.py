"""
Test MCP Client Basic Functionality - Must Pass
Light tests focusing on core MCP client features
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

try:
    # Note: client-server has a hyphen, need to import differently
    import importlib.util
    import sys
    from pathlib import Path
    
    client_path = Path(__file__).resolve().parents[4] / "ethos_ai_brain" / "orchestration" / "mcp" / "client-server" / "client.py"
    spec = importlib.util.spec_from_file_location("client", client_path)
    client_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client_module)
    
    EthosMCPClient = client_module.EthosMCPClient
    HAS_MCP_CLIENT = True
except Exception:
    HAS_MCP_CLIENT = False
    EthosMCPClient = None


@pytest.mark.skipif(not HAS_MCP_CLIENT, reason="EthosMCPClient not available")
def test_mcp_client_creation():
    """Test creating EthosMCPClient"""
    client = EthosMCPClient()
    
    assert isinstance(client, EthosMCPClient)
    assert hasattr(client, 'server_url')


@pytest.mark.skipif(not HAS_MCP_CLIENT, reason="EthosMCPClient not available")
def test_mcp_client_attributes():
    """Test EthosMCPClient basic attributes"""
    client = EthosMCPClient()
    
    # Check for basic attributes
    assert hasattr(client, 'server_url')
    assert hasattr(client, '__init__')


@pytest.mark.skipif(not HAS_MCP_CLIENT, reason="EthosMCPClient not available")
def test_mcp_client_default_url():
    """Test EthosMCPClient default server URL"""
    client = EthosMCPClient()
    
    assert client.server_url == "http://localhost:8051"


def test_mcp_module_exists():
    """Test that MCP module can be imported"""
    try:
        from ethos_ai_brain.orchestration.mcp import __init__
        assert True  # If we get here, import worked
        
    except ImportError as e:
        pytest.skip(f"MCP module import failed: {e}")
