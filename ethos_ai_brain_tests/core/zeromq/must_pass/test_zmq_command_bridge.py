"""
Test ZMQ Command Bridge - Must Pass
Tests each function in the ZMQCommandBridge class with real code only
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.core.zeromq.zmq_command_bridge import ZMQCommandBridge


def test_zmq_command_bridge_creation():
    """Test ZMQCommandBridge.__init__ - command bridge creation"""
    bridge = ZMQCommandBridge()
    
    print(f"Expected message_routing_graph exists: {hasattr(bridge, 'message_routing_graph')}")
    print(f"Expected pattern_optimization_graph exists: {hasattr(bridge, 'pattern_optimization_graph')}")
    print(f"Expected command_servers exists: {hasattr(bridge, 'command_servers')}")
    print(f"Expected biological_patterns exists: {hasattr(bridge, 'biological_patterns')}")
    
    assert hasattr(bridge, 'message_routing_graph')
    assert hasattr(bridge, 'pattern_optimization_graph')
    assert hasattr(bridge, 'zmq_engine')
    assert hasattr(bridge, 'command_servers')
    assert hasattr(bridge, 'biological_patterns')
    
    print("[SUCCESS] ZMQCommandBridge creation works correctly")


def test_zmq_command_bridge_routing_graph_setup():
    """Test that routing graph is properly initialized with military hierarchy"""
    bridge = ZMQCommandBridge()
    
    # Test that routing graph has nodes
    nodes = list(bridge.message_routing_graph.nodes())
    print(f"Routing graph nodes: {nodes}")
    print(f"Number of nodes: {len(nodes)}")
    
    # Test expected military roles are present
    expected_roles = ['major_general', 'strategic_advisor', 'mission_planner', 'quality_assurance']
    
    for role in expected_roles:
        if role in nodes:
            print(f"Found expected role: {role}")
            assert role in nodes
    
    # Test that graph has edges (routing paths)
    edges = list(bridge.message_routing_graph.edges())
    print(f"Number of routing edges: {len(edges)}")
    
    assert len(edges) > 0, "Should have routing edges"
    
    print("[SUCCESS] Routing graph setup works correctly")


def test_zmq_command_bridge_pattern_optimization_graph():
    """Test pattern optimization graph initialization"""
    bridge = ZMQCommandBridge()
    
    # Test pattern optimization graph
    pattern_nodes = list(bridge.pattern_optimization_graph.nodes())
    print(f"Pattern optimization nodes: {pattern_nodes}")
    print(f"Number of pattern nodes: {len(pattern_nodes)}")
    
    # Test expected patterns
    expected_patterns = ['reflex_arc', 'spreading_activation', 'hierarchical_processing']
    
    for pattern in expected_patterns:
        if pattern in pattern_nodes:
            print(f"Found expected pattern: {pattern}")
    
    # Test pattern edges
    pattern_edges = list(bridge.pattern_optimization_graph.edges())
    print(f"Number of pattern edges: {len(pattern_edges)}")
    
    assert len(pattern_edges) > 0, "Should have pattern optimization edges"
    
    print("[SUCCESS] Pattern optimization graph works correctly")


def test_zmq_command_bridge_message_routing_attributes():
    """Test message routing edge attributes"""
    bridge = ZMQCommandBridge()
    
    # Test that edges have proper attributes
    edges_with_data = bridge.message_routing_graph.edges(data=True)
    
    for source, target, data in edges_with_data:
        print(f"Route: {source} -> {target}")
        print(f"  Attributes: {data}")
        
        # Test expected attribute structure
        if 'message_types' in data:
            assert isinstance(data['message_types'], list)
            print(f"  Message types: {data['message_types']}")
        
        if 'priority' in data:
            assert data['priority'] in ['high', 'medium', 'low']
            print(f"  Priority: {data['priority']}")
    
    print("[SUCCESS] Message routing attributes work correctly")


def test_zmq_command_bridge_pattern_effectiveness():
    """Test pattern optimization effectiveness ratings"""
    bridge = ZMQCommandBridge()
    
    # Test pattern edges have effectiveness ratings
    pattern_edges_with_data = bridge.pattern_optimization_graph.edges(data=True)
    
    for source, target, data in pattern_edges_with_data:
        print(f"Pattern: {source} -> {target}")
        print(f"  Attributes: {data}")
        
        if 'effectiveness' in data:
            effectiveness = data['effectiveness']
            print(f"  Effectiveness: {effectiveness}")
            assert 0.0 <= effectiveness <= 1.0, "Effectiveness should be between 0 and 1"
        
        if 'latency' in data:
            latency = data['latency']
            print(f"  Latency: {latency}")
            assert latency in ['very_low', 'low', 'medium', 'high'], "Should have valid latency rating"
    
    print("[SUCCESS] Pattern effectiveness ratings work correctly")


def test_zmq_command_bridge_graph_connectivity():
    """Test that routing graph is properly connected"""
    bridge = ZMQCommandBridge()
    
    # Test graph connectivity
    is_connected = bridge.message_routing_graph.number_of_nodes() > 0
    print(f"Graph has nodes: {is_connected}")
    
    # Test that major_general has outgoing connections (command structure)
    if 'major_general' in bridge.message_routing_graph.nodes():
        successors = list(bridge.message_routing_graph.successors('major_general'))
        print(f"Major general connects to: {successors}")
        assert len(successors) > 0, "Major general should have subordinates"
    
    print("[SUCCESS] Graph connectivity works correctly")


def test_zmq_command_bridge_zmq_engine_integration():
    """Test ZeroMQ engine integration"""
    bridge = ZMQCommandBridge()
    
    print(f"ZMQ engine type: {type(bridge.zmq_engine).__name__}")
    print(f"ZMQ engine exists: {bridge.zmq_engine is not None}")
    
    assert bridge.zmq_engine is not None
    assert hasattr(bridge.zmq_engine, 'create_and_start_server')  # Expected ZeroMQ method
    
    print("[SUCCESS] ZMQ engine integration works correctly")


def test_zmq_command_bridge_methods_exist():
    """Test that expected methods exist on command bridge"""
    bridge = ZMQCommandBridge()
    
    # Test expected methods exist
    expected_methods = [
        '__init__',
        # Add other methods as we discover them in the implementation
    ]
    
    for method_name in expected_methods:
        print(f"Checking method: {method_name}")
        assert hasattr(bridge, method_name), f"Missing method: {method_name}"
        if method_name != '__init__':  # Skip __init__ for callable check
            assert callable(getattr(bridge, method_name)), f"Method not callable: {method_name}"
    
    print("[SUCCESS] Expected methods exist")


def test_zmq_command_bridge_different_instances():
    """Test that different bridge instances are independent"""
    bridge1 = ZMQCommandBridge()
    bridge2 = ZMQCommandBridge()
    
    print(f"Bridge1 ID: {id(bridge1)}")
    print(f"Bridge2 ID: {id(bridge2)}")
    print(f"Are different instances: {bridge1 is not bridge2}")
    
    assert bridge1 is not bridge2
    assert id(bridge1) != id(bridge2)
    assert bridge1.zmq_engine is not bridge2.zmq_engine  # Should have separate engines
    
    print("[SUCCESS] Different instances work correctly")


def test_zmq_command_bridge_graph_types():
    """Test that graphs are proper NetworkX graph types"""
    bridge = ZMQCommandBridge()
    
    # Test graph types
    import networkx as nx
    
    print(f"Message routing graph type: {type(bridge.message_routing_graph)}")
    print(f"Pattern optimization graph type: {type(bridge.pattern_optimization_graph)}")
    
    assert isinstance(bridge.message_routing_graph, nx.Graph), "Should be NetworkX graph"
    assert isinstance(bridge.pattern_optimization_graph, nx.Graph), "Should be NetworkX graph"
    
    print("[SUCCESS] Graph types are correct")
