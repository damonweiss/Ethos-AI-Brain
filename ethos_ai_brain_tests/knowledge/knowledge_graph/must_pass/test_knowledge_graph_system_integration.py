"""
Test KnowledgeGraph System Integration - Must Pass
Tests the KnowledgeGraph functionality after inheriting from KnowledgeSystem
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.knowledge import KnowledgeGraph, GraphType


def test_knowledge_graph_properties():
    """Test new abstract properties implementation"""
    kg = KnowledgeGraph("test_props", GraphType.INTENT)
    
    # Test knowledge_id property
    print(f"Expected knowledge_id: test_props, Actual: {kg.knowledge_id}")
    assert kg.knowledge_id == "test_props", "[FAILURE] knowledge_id should match graph_id"
    
    # Test knowledge_type property
    print(f"Expected knowledge_type: knowledge_graph, Actual: {kg.knowledge_type}")
    assert kg.knowledge_type == "knowledge_graph", "[FAILURE] knowledge_type should be 'knowledge_graph'"
    
    # Test legacy compatibility
    print(f"Legacy graph_id: {kg.graph_id}, graph_type: {kg.graph_type}")
    assert kg.graph_id == "test_props", "[FAILURE] Legacy graph_id should work"
    assert kg.graph_type == GraphType.INTENT, "[FAILURE] Legacy graph_type should work"
    
    print("[SUCCESS] KnowledgeGraph properties work correctly")


def test_metadata_system():
    """Test enhanced metadata system"""
    kg = KnowledgeGraph("test_metadata", GraphType.INTENT)
    
    # Test initial metadata
    metadata = kg.get_metadata()
    print(f"Initial metadata keys: {list(metadata.keys())}")
    assert "created_at" in metadata, "[FAILURE] Should have created_at timestamp"
    assert "last_modified" in metadata, "[FAILURE] Should have last_modified timestamp"
    assert "version" in metadata, "[FAILURE] Should have version"
    
    # Test setting custom metadata
    kg.set_metadata("custom_field", "test_value")
    kg.set_metadata("priority", "high")
    
    updated_metadata = kg.get_metadata()
    print(f"Custom field: {updated_metadata.get('custom_field')}")
    print(f"Priority: {updated_metadata.get('priority')}")
    assert updated_metadata["custom_field"] == "test_value", "[FAILURE] Custom metadata should be set"
    assert updated_metadata["priority"] == "high", "[FAILURE] Priority metadata should be set"
    
    # Test bulk metadata update
    kg.update_metadata({"batch_field1": "value1", "batch_field2": "value2"})
    final_metadata = kg.get_metadata()
    assert final_metadata["batch_field1"] == "value1", "[FAILURE] Batch update should work"
    assert final_metadata["batch_field2"] == "value2", "[FAILURE] Batch update should work"
    
    print("[SUCCESS] Metadata system works correctly")


def test_3d_positioning():
    """Test 3D positioning functionality"""
    kg = KnowledgeGraph("test_3d", GraphType.INTENT)
    
    # Test initial position (should be None)
    initial_pos = kg.get_global_position()
    print(f"Initial position: {initial_pos}")
    assert initial_pos is None, "[FAILURE] Initial position should be None"
    
    # Test setting global position
    kg.set_global_position(10.5, 20.3, 30.7)
    position = kg.get_global_position()
    print(f"Set position: {position}")
    assert position == (10.5, 20.3, 30.7), "[FAILURE] Position should be set correctly"
    
    # Test 3D bounds
    bounds = kg.get_3d_bounds()
    print(f"3D bounds: {bounds}")
    expected_bounds = ((10.5, 10.5), (20.3, 20.3), (30.7, 30.7))
    assert bounds == expected_bounds, "[FAILURE] 3D bounds should be point bounds for basic graph"
    
    # Test 3D positioning capability
    assert kg.supports_3d_positioning(), "[FAILURE] Should support 3D positioning"
    
    print("[SUCCESS] 3D positioning works correctly")


def test_serialization():
    """Test to_dict and from_dict serialization"""
    kg = KnowledgeGraph("test_serial", GraphType.EXECUTION)
    kg.add_node("node1", type="process", status="active")
    kg.add_node("node2", type="output", status="ready")
    kg.add_edge("node1", "node2", relationship="produces")
    kg.set_global_position(5.0, 10.0, 15.0)
    kg.set_metadata("custom", "serialization_test")
    
    # Test serialization
    data = kg.to_dict()
    print(f"Serialized keys: {list(data.keys())}")
    
    # Check required fields
    assert data["knowledge_id"] == "test_serial", "[FAILURE] Should serialize knowledge_id"
    assert data["knowledge_type"] == "knowledge_graph", "[FAILURE] Should serialize knowledge_type"
    assert data["global_position"] == (5.0, 10.0, 15.0), "[FAILURE] Should serialize position"
    assert "metadata" in data, "[FAILURE] Should serialize metadata"
    assert "components" in data, "[FAILURE] Should serialize components"
    assert "relationships" in data, "[FAILURE] Should serialize relationships"
    
    # Check components and relationships
    components = data["components"]
    relationships = data["relationships"]
    print(f"Components: {components}")
    print(f"Relationships: {relationships}")
    assert "node1" in components, "[FAILURE] Should include node1 in components"
    assert "node2" in components, "[FAILURE] Should include node2 in components"
    assert len(relationships) == 1, "[FAILURE] Should have one relationship"
    
    # Test deserialization
    kg2 = KnowledgeGraph("test_deserial", GraphType.INTENT)
    kg2.from_dict(data)
    
    # Check deserialized data
    assert kg2.get_global_position() == (5.0, 10.0, 15.0), "[FAILURE] Should deserialize position"
    metadata = kg2.get_metadata()
    assert metadata.get("custom") == "serialization_test", "[FAILURE] Should deserialize custom metadata"
    
    print("[SUCCESS] Serialization works correctly")


def test_abstract_methods():
    """Test new abstract methods implementation"""
    kg = KnowledgeGraph("test_abstract", GraphType.INTENT)
    kg.add_node("concept1", type="idea", importance="high")
    kg.add_node("concept2", type="idea", importance="medium")
    kg.add_node("concept3", type="action", status="pending")
    kg.add_edge("concept1", "concept2", relationship="relates_to", strength=0.8)
    kg.add_edge("concept2", "concept3", relationship="leads_to", strength=0.9)
    
    # Test get_components
    components = kg.get_components()
    print(f"Components: {components}")
    assert len(components) == 3, "[FAILURE] Should have 3 components (nodes)"
    assert "concept1" in components, "[FAILURE] Should include concept1"
    assert "concept2" in components, "[FAILURE] Should include concept2"
    assert "concept3" in components, "[FAILURE] Should include concept3"
    
    # Test get_relationships
    relationships = kg.get_relationships()
    print(f"Relationships count: {len(relationships)}")
    assert len(relationships) == 2, "[FAILURE] Should have 2 relationships (edges)"
    
    # Check relationship structure
    rel1 = relationships[0]
    print(f"First relationship: {rel1}")
    assert len(rel1) == 3, "[FAILURE] Relationship should be (source, target, metadata) tuple"
    assert rel1[0] in ["concept1", "concept2"], "[FAILURE] Source should be valid node"
    assert rel1[1] in ["concept2", "concept3"], "[FAILURE] Target should be valid node"
    assert isinstance(rel1[2], dict), "[FAILURE] Metadata should be dictionary"
    
    print("[SUCCESS] Abstract methods work correctly")


def test_visualization_integration():
    """Test visualization color and physics integration"""
    kg = KnowledgeGraph("test_viz", GraphType.INTENT)
    
    # Test initial color (should be None)
    initial_color = kg.get_visualization_color()
    print(f"Initial color: {initial_color}")
    assert initial_color is None, "[FAILURE] Initial color should be None"
    
    # Test setting color
    kg.set_visualization_color("#FF6B6B")
    color = kg.get_visualization_color()
    print(f"Set color: {color}")
    assert color == "#FF6B6B", "[FAILURE] Color should be set correctly"
    
    # Test physics support detection
    supports_physics = kg.supports_physics()
    print(f"Supports physics: {supports_physics}")
    # Basic KnowledgeGraph doesn't support physics by default
    assert not supports_physics, "[FAILURE] Basic graph should not support physics by default"
    
    # Test 3D positioning support
    supports_3d = kg.supports_3d_positioning()
    print(f"Supports 3D positioning: {supports_3d}")
    assert supports_3d, "[FAILURE] Should support 3D positioning"
    
    print("[SUCCESS] Visualization integration works correctly")


# Converted to pytest format - no manual main needed
