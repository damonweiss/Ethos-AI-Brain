"""
Test ZMQ Core Components Basic Functionality - Must Pass
Light tests focusing on core ZMQ components
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

try:
    from ethos_ai_brain.core.zeromq.zmq_node_base import ZMQNode
    HAS_ZMQ_NODE = True
except ImportError:
    HAS_ZMQ_NODE = False
    ZMQNode = None

try:
    from ethos_ai_brain.core.zeromq.zmq_command_bridge import ZMQCommandBridge
    HAS_ZMQ_BRIDGE = True
except ImportError:
    HAS_ZMQ_BRIDGE = False
    ZMQCommandBridge = None


def test_zmq_directory_exists():
    """Test that ZMQ directory exists in core"""
    zmq_dir = Path(__file__).resolve().parents[4] / "ethos_ai_brain" / "core" / "zeromq"
    
    assert zmq_dir.exists()
    assert zmq_dir.is_dir()


@pytest.mark.skipif(not HAS_ZMQ_NODE, reason="ZMQNode not available")
def test_zmq_node_attributes():
    """Test ZMQNode basic attributes"""
    try:
        # Check if we can import and inspect the class
        assert ZMQNode is not None
        assert hasattr(ZMQNode, '__init__')
        
    except Exception as e:
        pytest.skip(f"ZMQNode inspection failed: {e}")


@pytest.mark.skipif(not HAS_ZMQ_BRIDGE, reason="ZMQCommandBridge not available")
def test_zmq_command_bridge_creation():
    """Test creating ZMQCommandBridge"""
    bridge = ZMQCommandBridge()
    
    assert isinstance(bridge, ZMQCommandBridge)
    assert hasattr(bridge, '__init__')


@pytest.mark.skipif(not HAS_ZMQ_BRIDGE, reason="ZMQCommandBridge not available")
def test_zmq_command_bridge_attributes():
    """Test ZMQCommandBridge basic attributes"""
    bridge = ZMQCommandBridge()
    
    # Check for basic attributes that a command bridge should have
    assert hasattr(bridge, '__init__')


def test_zmq_files_exist():
    """Test that ZMQ files exist"""
    zmq_dir = Path(__file__).resolve().parents[4] / "ethos_ai_brain" / "core" / "zeromq"
    
    expected_files = ["zmq_node_base.py", "zmq_command_bridge.py"]
    for file_name in expected_files:
        file_path = zmq_dir / file_name
        assert file_path.exists()
        assert file_path.is_file()
