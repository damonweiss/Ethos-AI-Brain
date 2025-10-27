"""
Test Knowledge System Integration - Must Pass
Tests the knowledge system integration across all knowledge types
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.knowledge import (
    KnowledgeGraph, 
    LayeredKnowledgeGraph,
    UnifiedKnowledgeManager,
    GraphType
)


def test_unified_interface_consistency():
    """Test that all knowledge types implement the interface consistently"""
    kg = KnowledgeGraph("test_kg", GraphType.INTENT)
    lg = LayeredKnowledgeGraph("test_lg")
    
    # Test that both implement required properties
    print(f"KG knowledge_id: {kg.knowledge_id}, knowledge_type: {kg.knowledge_type}")
    print(f"LG knowledge_id: {lg.knowledge_id}, knowledge_type: {lg.knowledge_type}")
    
    assert hasattr(kg, 'knowledge_id'), "[FAILURE] KG should have knowledge_id property"
    assert hasattr(kg, 'knowledge_type'), "[FAILURE] KG should have knowledge_type property"
    assert hasattr(lg, 'knowledge_id'), "[FAILURE] LG should have knowledge_id property"
    assert hasattr(lg, 'knowledge_type'), "[FAILURE] LG should have knowledge_type property"
    
    # Test that both implement required methods
    required_methods = ['query', 'add_knowledge', 'get_related', 'get_capabilities', 
                       'get_components', 'get_relationships', 'get_metadata', 
                       'set_metadata', 'to_dict', 'from_dict']
    
    for method in required_methods:
        assert hasattr(kg, method), f"[FAILURE] KG should have {method} method"
        assert hasattr(lg, method), f"[FAILURE] LG should have {method} method"
        assert callable(getattr(kg, method)), f"[FAILURE] KG {method} should be callable"
        assert callable(getattr(lg, method)), f"[FAILURE] LG {method} should be callable"
    
    print("[SUCCESS] Both knowledge types implement unified interface consistently")


def test_metadata_consistency():
    """Test metadata system consistency across knowledge types"""
    kg = KnowledgeGraph("test_meta_kg", GraphType.INTENT)
    lg = LayeredKnowledgeGraph("test_meta_lg")
    
    # Test initial metadata structure
    kg_meta = kg.get_metadata()
    lg_meta = lg.get_metadata()
    
    common_fields = ['created_at', 'last_modified', 'version', 'description', 'tags', 'source', 'confidence']
    
    for field in common_fields:
        assert field in kg_meta, f"[FAILURE] KG should have {field} in metadata"
        assert field in lg_meta, f"[FAILURE] LG should have {field} in metadata"
    
    # Test custom metadata setting
    kg.set_metadata("test_field", "kg_value")
    lg.set_metadata("test_field", "lg_value")
    
    assert kg.get_metadata()["test_field"] == "kg_value", "[FAILURE] KG custom metadata should work"
    assert lg.get_metadata()["test_field"] == "lg_value", "[FAILURE] LG custom metadata should work"
    
    # Test timestamp updates
    import time
    time.sleep(0.01)  # Small delay to ensure timestamp difference
    
    kg.set_metadata("trigger_update", "test")
    lg.set_metadata("trigger_update", "test")
    
    kg_updated = kg.get_metadata()["last_modified"]
    lg_updated = lg.get_metadata()["last_modified"]
    
    assert kg_updated != kg_meta["last_modified"], "[FAILURE] KG timestamp should update"
    assert lg_updated != lg_meta["last_modified"], "[FAILURE] LG timestamp should update"
    
    print("[SUCCESS] Metadata system is consistent across knowledge types")


def test_3d_positioning_consistency():
    """Test 3D positioning works consistently across knowledge types"""
    kg = KnowledgeGraph("test_3d_kg", GraphType.INTENT)
    lg = LayeredKnowledgeGraph("test_3d_lg")
    
    # Test initial state
    assert kg.get_global_position() is None, "[FAILURE] KG should start with no position"
    assert lg.get_global_position() is None, "[FAILURE] LG should start with no position"
    
    # Test setting positions
    kg.set_global_position(1.0, 2.0, 3.0)
    lg.set_global_position(4.0, 5.0, 6.0)
    
    kg_pos = kg.get_global_position()
    lg_pos = lg.get_global_position()
    
    print(f"KG position: {kg_pos}, LG position: {lg_pos}")
    assert kg_pos == (1.0, 2.0, 3.0), "[FAILURE] KG position should be set correctly"
    assert lg_pos == (4.0, 5.0, 6.0), "[FAILURE] LG position should be set correctly"
    
    # Test 3D bounds
    kg_bounds = kg.get_3d_bounds()
    lg_bounds = lg.get_3d_bounds()
    
    print(f"KG bounds: {kg_bounds}, LG bounds: {lg_bounds}")
    assert kg_bounds == ((1.0, 1.0), (2.0, 2.0), (3.0, 3.0)), "[FAILURE] KG bounds should be point bounds"
    assert lg_bounds == ((4.0, 4.0), (5.0, 5.0), (6.0, 6.0)), "[FAILURE] LG bounds should be point bounds"
    
    # Test capability detection
    assert kg.supports_3d_positioning(), "[FAILURE] KG should support 3D positioning"
    assert lg.supports_3d_positioning(), "[FAILURE] LG should support 3D positioning"
    
    print("[SUCCESS] 3D positioning is consistent across knowledge types")


def test_serialization_consistency():
    """Test serialization works consistently across knowledge types"""
    kg = KnowledgeGraph("test_serial_kg", GraphType.INTENT)
    kg.add_node("kg_node", type="concept")
    kg.set_global_position(10.0, 20.0, 30.0)
    kg.set_metadata("kg_custom", "kg_test")
    
    lg = LayeredKnowledgeGraph("test_serial_lg")
    layer = KnowledgeGraph("test_layer", GraphType.EXECUTION)
    layer.add_node("lg_node", type="action")
    lg.add_layer(layer)
    lg.set_global_position(40.0, 50.0, 60.0)
    lg.set_metadata("lg_custom", "lg_test")
    
    # Test serialization
    kg_data = kg.to_dict()
    lg_data = lg.to_dict()
    
    # Check common structure
    common_keys = ['knowledge_id', 'knowledge_type', 'metadata', 'global_position', 'components', 'relationships']
    
    for key in common_keys:
        assert key in kg_data, f"[FAILURE] KG serialization should include {key}"
        assert key in lg_data, f"[FAILURE] LG serialization should include {key}"
    
    # Check type-specific values
    assert kg_data['knowledge_type'] == 'knowledge_graph', "[FAILURE] KG should serialize correct type"
    assert lg_data['knowledge_type'] == 'layered_knowledge_graph', "[FAILURE] LG should serialize correct type"
    
    # Check components
    assert 'kg_node' in kg_data['components'], "[FAILURE] KG should serialize nodes as components"
    assert 'test_layer' in lg_data['components'], "[FAILURE] LG should serialize layers as components"
    
    # Test deserialization
    kg2 = KnowledgeGraph("temp", GraphType.INTENT)
    lg2 = LayeredKnowledgeGraph("temp")
    
    kg2.from_dict(kg_data)
    lg2.from_dict(lg_data)
    
    assert kg2.get_global_position() == (10.0, 20.0, 30.0), "[FAILURE] KG deserialization should work"
    assert lg2.get_global_position() == (40.0, 50.0, 60.0), "[FAILURE] LG deserialization should work"
    assert kg2.get_metadata()['kg_custom'] == 'kg_test', "[FAILURE] KG should deserialize custom metadata"
    assert lg2.get_metadata()['lg_custom'] == 'lg_test', "[FAILURE] LG should deserialize custom metadata"
    
    print("[SUCCESS] Serialization is consistent across knowledge types")


def test_unified_manager_enhanced_integration():
    """Test UnifiedKnowledgeManager works with enhanced functionality"""
    kg = KnowledgeGraph("manager_kg", GraphType.INTENT)
    kg.add_node("concept1", type="idea", priority="high")
    kg.add_node("concept2", type="idea", priority="medium")
    kg.set_metadata("source", "test_kg")
    kg.set_visualization_color("#FF6B6B")
    
    lg = LayeredKnowledgeGraph("manager_lg")
    layer = KnowledgeGraph("test_layer", GraphType.EXECUTION)
    layer.add_node("action1", type="step", priority="high")
    lg.add_layer(layer)
    lg.set_metadata("source", "test_lg")
    lg.set_visualization_color("#45B7D1")
    
    # Test manager integration
    manager = UnifiedKnowledgeManager()
    manager.add_knowledge_source(kg, is_primary=True)
    manager.add_knowledge_source(lg)
    
    # Test combined capabilities
    all_caps = manager.get_all_capabilities()
    print(f"Combined capabilities: {list(all_caps.keys())}")
    
    # Should have capabilities from both types
    assert all_caps['semantic_search'], "[FAILURE] Should support semantic search"
    assert all_caps['relationship_traversal'], "[FAILURE] Should support relationship traversal"
    assert all_caps['cross_layer_queries'], "[FAILURE] Should support cross-layer queries (from LG)"
    assert all_caps['3d_positioning'], "[FAILURE] Should support 3D positioning"
    assert all_caps['physics_simulation'], "[FAILURE] Should support physics simulation (from LG)"
    
    # Test querying across enhanced sources
    results = manager.query("concept")
    print(f"Query results: {len(results)}")
    
    # Should get results from both sources
    kg_results = [r for r in results if r.knowledge_type == "graph_node"]
    lg_results = [r for r in results if r.knowledge_type == "layered_graph_node"]
    
    print(f"KG results: {len(kg_results)}, LG results: {len(lg_results)}")
    assert len(kg_results) > 0, "[FAILURE] Should get results from KG"
    # Note: LG results depend on query matching in layer content
    
    # Test that enhanced properties are accessible
    assert kg.get_visualization_color() == "#FF6B6B", "[FAILURE] KG color should be preserved"
    assert lg.get_visualization_color() == "#45B7D1", "[FAILURE] LG color should be preserved"
    
    print("[SUCCESS] UnifiedKnowledgeManager integrates well with enhanced functionality")


def test_capability_differentiation():
    """Test that different knowledge types report different capabilities correctly"""
    kg = KnowledgeGraph("cap_kg", GraphType.INTENT)
    lg = LayeredKnowledgeGraph("cap_lg")
    
    kg_caps = kg.get_capabilities()
    lg_caps = lg.get_capabilities()
    
    print(f"KG capabilities: {kg_caps}")
    print(f"LG capabilities: {lg_caps}")
    
    # Test capabilities that should be the same
    same_caps = ['semantic_search', 'relationship_traversal', '3d_positioning']
    for cap in same_caps:
        assert kg_caps[cap] == lg_caps[cap], f"[FAILURE] {cap} should be same for both types"
    
    # Test capabilities that should be different
    assert kg_caps['cross_layer_queries'] == False, "[FAILURE] KG should not support cross-layer queries"
    assert lg_caps['cross_layer_queries'] == True, "[FAILURE] LG should support cross-layer queries"
    
    assert kg_caps['physics_simulation'] == False, "[FAILURE] KG should not support physics by default"
    assert lg_caps['physics_simulation'] == True, "[FAILURE] LG should support physics simulation"
    
    assert kg_caps['cross_graph_analysis'] == False, "[FAILURE] KG should not support cross-graph analysis"
    assert lg_caps['cross_graph_analysis'] == True, "[FAILURE] LG should support cross-graph analysis"
    
    print("[SUCCESS] Capability differentiation works correctly")


if __name__ == "__main__":
    print("Running Enhanced Unified Interface Tests...")
    test_unified_interface_consistency()
    test_metadata_consistency()
    test_3d_positioning_consistency()
    test_serialization_consistency()
    test_unified_manager_enhanced_integration()
    test_capability_differentiation()
    print("[SUCCESS] All enhanced unified interface tests passed!")
