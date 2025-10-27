"""
Test LayeredKnowledgeGraph System Integration - Must Pass
Tests the LayeredKnowledgeGraph functionality after inheriting from KnowledgeSystem
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.knowledge import KnowledgeGraph, LayeredKnowledgeGraph, GraphType


def test_layered_knowledge_graph_properties():
    """Test new abstract properties implementation"""
    lg = LayeredKnowledgeGraph("test_layered_props")
    
    # Test knowledge_id property
    print(f"Expected knowledge_id: test_layered_props, Actual: {lg.knowledge_id}")
    assert lg.knowledge_id == "test_layered_props", "[FAILURE] knowledge_id should match network_id"
    
    # Test knowledge_type property
    print(f"Expected knowledge_type: layered_knowledge_graph, Actual: {lg.knowledge_type}")
    assert lg.knowledge_type == "layered_knowledge_graph", "[FAILURE] knowledge_type should be 'layered_knowledge_graph'"
    
    # Test legacy compatibility
    print(f"Legacy network_id: {lg.network_id}")
    assert lg.network_id == "test_layered_props", "[FAILURE] Legacy network_id should work"
    
    print("[SUCCESS] LayeredKnowledgeGraph properties work correctly")


def test_layered_metadata_system():
    """Test enhanced metadata system with layer tracking"""
    lg = LayeredKnowledgeGraph("test_layered_metadata", z_separation=3.0)
    
    # Test initial metadata
    metadata = lg.get_metadata()
    print(f"Initial metadata keys: {list(metadata.keys())}")
    assert "created_at" in metadata, "[FAILURE] Should have created_at timestamp"
    assert "network_id" in metadata, "[FAILURE] Should have network_id"
    assert "z_separation" in metadata, "[FAILURE] Should have z_separation"
    assert metadata["layer_count"] == 0, "[FAILURE] Should start with 0 layers"
    
    # Add layers and test layer count tracking
    layer1 = KnowledgeGraph("layer1", GraphType.INTENT)
    layer2 = KnowledgeGraph("layer2", GraphType.EXECUTION)
    
    lg.add_layer(layer1)
    metadata_after_1 = lg.get_metadata()
    print(f"Layer count after adding 1 layer: {metadata_after_1['layer_count']}")
    assert metadata_after_1["layer_count"] == 1, "[FAILURE] Should track layer count"
    
    lg.add_layer(layer2)
    metadata_after_2 = lg.get_metadata()
    print(f"Layer count after adding 2 layers: {metadata_after_2['layer_count']}")
    assert metadata_after_2["layer_count"] == 2, "[FAILURE] Should track layer count"
    
    # Test custom metadata
    lg.set_metadata("complexity", "high")
    lg.set_metadata("domain", "machine_learning")
    
    final_metadata = lg.get_metadata()
    assert final_metadata["complexity"] == "high", "[FAILURE] Custom metadata should be set"
    assert final_metadata["domain"] == "machine_learning", "[FAILURE] Custom metadata should be set"
    
    print("[SUCCESS] LayeredKnowledgeGraph metadata system works correctly")


def test_layered_3d_positioning():
    """Test 3D positioning functionality for layered graphs"""
    lg = LayeredKnowledgeGraph("test_layered_3d")
    
    # Test global positioning
    lg.set_global_position(100.0, 200.0, 300.0)
    position = lg.get_global_position()
    print(f"Global position: {position}")
    assert position == (100.0, 200.0, 300.0), "[FAILURE] Global position should be set correctly"
    
    # Test 3D bounds (should be point bounds for empty layered graph)
    bounds = lg.get_3d_bounds()
    print(f"3D bounds: {bounds}")
    expected_bounds = ((100.0, 100.0), (200.0, 200.0), (300.0, 300.0))
    assert bounds == expected_bounds, "[FAILURE] 3D bounds should be point bounds"
    
    # Test capabilities
    assert lg.supports_3d_positioning(), "[FAILURE] Should support 3D positioning"
    assert lg.supports_physics(), "[FAILURE] Should support physics simulation"
    
    print("[SUCCESS] LayeredKnowledgeGraph 3D positioning works correctly")


def test_layered_serialization():
    """Test serialization with layers"""
    lg = LayeredKnowledgeGraph("test_layered_serial", z_separation=2.5)
    
    # Add layers with content
    layer1 = KnowledgeGraph("intent_layer", GraphType.INTENT)
    layer1.add_node("goal1", type="objective", priority="high")
    layer1.add_node("goal2", type="objective", priority="medium")
    
    layer2 = KnowledgeGraph("execution_layer", GraphType.EXECUTION)
    layer2.add_node("action1", type="step", status="ready")
    layer2.add_node("action2", type="step", status="pending")
    layer2.add_edge("action1", "action2", relationship="precedes")
    
    lg.add_layer(layer1, z_index=10.0)
    lg.add_layer(layer2, z_index=5.0)
    lg.set_global_position(50.0, 75.0, 100.0)
    lg.set_metadata("purpose", "test_serialization")
    
    # Test serialization
    data = lg.to_dict()
    print(f"Serialized keys: {list(data.keys())}")
    
    # Check required fields
    assert data["knowledge_id"] == "test_layered_serial", "[FAILURE] Should serialize knowledge_id"
    assert data["knowledge_type"] == "layered_knowledge_graph", "[FAILURE] Should serialize knowledge_type"
    assert data["global_position"] == (50.0, 75.0, 100.0), "[FAILURE] Should serialize position"
    
    # Check components (layers)
    components = data["components"]
    print(f"Components (layers): {components}")
    assert "intent_layer" in components, "[FAILURE] Should include intent_layer"
    assert "execution_layer" in components, "[FAILURE] Should include execution_layer"
    assert len(components) == 2, "[FAILURE] Should have 2 components"
    
    # Check relationships (cross-layer connections)
    relationships = data["relationships"]
    print(f"Cross-layer relationships: {len(relationships)}")
    # Should be empty since we didn't add cross-layer references
    assert isinstance(relationships, list), "[FAILURE] Relationships should be a list"
    
    # Test deserialization
    lg2 = LayeredKnowledgeGraph("test_deserial")
    lg2.from_dict(data)
    
    assert lg2.get_global_position() == (50.0, 75.0, 100.0), "[FAILURE] Should deserialize position"
    metadata = lg2.get_metadata()
    assert metadata.get("purpose") == "test_serialization", "[FAILURE] Should deserialize custom metadata"
    
    print("[SUCCESS] LayeredKnowledgeGraph serialization works correctly")


def test_layered_abstract_methods():
    """Test abstract methods implementation for layered graphs"""
    lg = LayeredKnowledgeGraph("test_layered_abstract")
    
    # Add layers with cross-layer references
    layer1 = KnowledgeGraph("input_layer", GraphType.INTENT)
    layer1.add_node("sensor1", type="input", location="room1")
    layer1.add_node("sensor2", type="input", location="room2")
    
    layer2 = KnowledgeGraph("processing_layer", GraphType.EXECUTION)
    layer2.add_node("processor1", type="compute", algorithm="ml")
    layer2.add_node("processor2", type="compute", algorithm="stats")
    
    layer3 = KnowledgeGraph("output_layer", GraphType.LOCAL_RAG)
    layer3.add_node("display1", type="visualization", format="web")
    layer3.add_node("alert1", type="notification", channel="email")
    
    lg.add_layer(layer1, z_index=0.0)
    lg.add_layer(layer2, z_index=5.0)
    lg.add_layer(layer3, z_index=10.0)
    
    # Add cross-layer references
    layer1.add_external_reference("sensor1", "processing_layer", "processor1", relationship="feeds_data")
    layer2.add_external_reference("processor1", "output_layer", "display1", relationship="outputs_to")
    
    # Test get_components (should return layer names)
    components = lg.get_components()
    print(f"Components (layers): {components}")
    assert len(components) == 3, "[FAILURE] Should have 3 components (layers)"
    assert "input_layer" in components, "[FAILURE] Should include input_layer"
    assert "processing_layer" in components, "[FAILURE] Should include processing_layer"
    assert "output_layer" in components, "[FAILURE] Should include output_layer"
    
    # Test get_relationships (should return cross-layer connections)
    relationships = lg.get_relationships()
    print(f"Cross-layer relationships: {len(relationships)}")
    assert len(relationships) == 2, "[FAILURE] Should have 2 cross-layer relationships"
    
    # Check relationship structure
    rel1 = relationships[0]
    print(f"First relationship: {rel1}")
    assert len(rel1) == 3, "[FAILURE] Relationship should be (source, target, metadata) tuple"
    assert ":" in rel1[0], "[FAILURE] Source should be layer:node format"
    assert ":" in rel1[1], "[FAILURE] Target should be layer:node format"
    assert isinstance(rel1[2], dict), "[FAILURE] Metadata should be dictionary"
    
    print("[SUCCESS] LayeredKnowledgeGraph abstract methods work correctly")


def test_layered_capabilities():
    """Test enhanced capabilities of layered graphs"""
    lg = LayeredKnowledgeGraph("test_capabilities")
    
    capabilities = lg.get_capabilities()
    print(f"LayeredKnowledgeGraph capabilities: {capabilities}")
    
    # Test expected capabilities
    assert capabilities["semantic_search"] == True, "[FAILURE] Should support semantic search"
    assert capabilities["relationship_traversal"] == True, "[FAILURE] Should support relationship traversal"
    assert capabilities["cross_layer_queries"] == True, "[FAILURE] Should support cross-layer queries"
    assert capabilities["3d_positioning"] == True, "[FAILURE] Should support 3D positioning"
    assert capabilities["physics_simulation"] == True, "[FAILURE] Should support physics simulation"
    assert capabilities["cross_graph_analysis"] == True, "[FAILURE] Should support cross-graph analysis"
    
    # Test capabilities that should be false
    assert capabilities["temporal_queries"] == False, "[FAILURE] Should not support temporal queries yet"
    assert capabilities["multi_dimensional_clustering"] == False, "[FAILURE] Should not support multi-dimensional clustering yet"
    
    print("[SUCCESS] LayeredKnowledgeGraph capabilities are correct")


def test_layer_management_integration():
    """Test that layer management integrates with enhanced interface"""
    lg = LayeredKnowledgeGraph("test_layer_mgmt")
    
    # Test empty state
    assert len(lg.get_components()) == 0, "[FAILURE] Should start with no components"
    assert len(lg.get_relationships()) == 0, "[FAILURE] Should start with no relationships"
    
    # Add layers and test component tracking
    layer1 = KnowledgeGraph("layer_a", GraphType.INTENT)
    layer1.add_node("node_a1", type="concept")
    
    layer2 = KnowledgeGraph("layer_b", GraphType.EXECUTION)
    layer2.add_node("node_b1", type="action")
    
    lg.add_layer(layer1)
    components_after_1 = lg.get_components()
    print(f"Components after adding 1 layer: {components_after_1}")
    assert len(components_after_1) == 1, "[FAILURE] Should have 1 component"
    assert "layer_a" in components_after_1, "[FAILURE] Should include layer_a"
    
    lg.add_layer(layer2)
    components_after_2 = lg.get_components()
    print(f"Components after adding 2 layers: {components_after_2}")
    assert len(components_after_2) == 2, "[FAILURE] Should have 2 components"
    assert "layer_b" in components_after_2, "[FAILURE] Should include layer_b"
    
    # Test metadata integration
    metadata = lg.get_metadata()
    assert metadata["layer_count"] == 2, "[FAILURE] Metadata should track layer count"
    
    print("[SUCCESS] Layer management integrates correctly with enhanced interface")


# Converted to pytest format - no manual main needed
