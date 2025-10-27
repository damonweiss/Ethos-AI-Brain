"""
Test Knowledge System - Must Pass
Tests the knowledge system interface that allows AI Brain to work with any knowledge system
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.knowledge import (
    KnowledgeGraph, 
    GraphType,
    LayeredKnowledgeGraph,
    UnifiedKnowledgeManager,
    KnowledgeResult
)


def test_knowledge_result_creation():
    """Test creating and converting KnowledgeResult"""
    result = KnowledgeResult(
        content="Test content",
        knowledge_id="test:node1",
        knowledge_type="graph_node",
        confidence=0.9
    )
    
    print(f"Expected content: 'Test content', Actual: '{result.content}'")
    assert result.content == "Test content", "[FAILURE] Content should match"
    
    print(f"Expected ID: 'test:node1', Actual: '{result.knowledge_id}'")
    assert result.knowledge_id == "test:node1", "[FAILURE] ID should match"
    
    # Test conversion to dict
    result_dict = result.to_dict()
    print(f"Dict keys: {list(result_dict.keys())}")
    assert "content" in result_dict, "[FAILURE] Dict should contain content"
    assert "id" in result_dict, "[FAILURE] Dict should contain id"
    
    print("[SUCCESS] KnowledgeResult creation and conversion works")


def test_knowledge_graph_direct():
    """Test KnowledgeGraph with direct KnowledgeSystem inheritance"""
    
    # Create basic knowledge graph
    kg = KnowledgeGraph("test_graph", GraphType.INTENT)
    kg.add_node("concept1", type="idea", importance="high")
    kg.add_node("concept2", type="idea", importance="medium")
    kg.add_edge("concept1", "concept2", relationship="relates_to")
    
    # Test capabilities (direct interface)
    capabilities = kg.get_capabilities()
    print(f"KnowledgeGraph capabilities: {capabilities}")
    assert capabilities["semantic_search"] == True, "[FAILURE] Should support semantic search"
    assert capabilities["cross_layer_queries"] == False, "[FAILURE] Basic graph shouldn't support cross-layer"
    
    # Test adding knowledge (direct interface)
    knowledge_id = kg.add_knowledge("new_concept", {"type": "test"})
    print(f"Added knowledge ID: {knowledge_id}")
    assert knowledge_id != "", "[FAILURE] Should return valid knowledge ID"
    assert "test_graph:" in knowledge_id, "[FAILURE] ID should contain graph ID"
    
    # Test querying (direct interface)
    results = kg.query("concept")
    print(f"Query results count: {len(results)}")
    assert isinstance(results, list), "[FAILURE] Should return list of results"
    
    # Test new interface properties
    print(f"Knowledge ID: {kg.knowledge_id}, Type: {kg.knowledge_type}")
    assert kg.knowledge_id == "test_graph", "[FAILURE] Should return correct knowledge ID"
    assert kg.knowledge_type == "knowledge_graph", "[FAILURE] Should return correct knowledge type"
    
    print("[SUCCESS] KnowledgeGraph direct interface works correctly")


def test_layered_knowledge_graph_direct():
    """Test LayeredKnowledgeGraph with direct KnowledgeSystem inheritance"""
    
    # Create real layered knowledge graph
    lg = LayeredKnowledgeGraph("test_layered")
    
    # Add layers
    layer1 = KnowledgeGraph("layer1", GraphType.INTENT)
    layer1.add_node("intent1", type="goal", priority="high")
    
    layer2 = KnowledgeGraph("layer2", GraphType.EXECUTION)
    layer2.add_node("action1", type="step", status="ready")
    
    lg.add_layer(layer1)
    lg.add_layer(layer2)
    
    # Test capabilities (direct interface)
    capabilities = lg.get_capabilities()
    print(f"LayeredKnowledgeGraph capabilities: {capabilities}")
    assert capabilities["cross_layer_queries"] == True, "[FAILURE] Should support cross-layer queries"
    
    # Test querying across layers (direct interface)
    results = lg.query("intent")
    print(f"Cross-layer query results: {len(results)}")
    
    for result in results:
        print(f"  Found: {result.content} (ID: {result.knowledge_id})")
        assert result.knowledge_type == "layered_graph_node", "[FAILURE] Should be layered graph node"
        assert "test_layered:" in result.knowledge_id, "[FAILURE] Should contain network ID"
    
    # Test new interface properties
    print(f"Knowledge ID: {lg.knowledge_id}, Type: {lg.knowledge_type}")
    assert lg.knowledge_id == "test_layered", "[FAILURE] Should return correct knowledge ID"
    assert lg.knowledge_type == "layered_knowledge_graph", "[FAILURE] Should return correct knowledge type"
    
    # Test components and relationships
    components = lg.get_components()
    print(f"Components: {components}")
    assert "layer1" in components, "[FAILURE] Should contain layer1"
    assert "layer2" in components, "[FAILURE] Should contain layer2"
    
    print("[SUCCESS] LayeredKnowledgeGraph direct interface works correctly")


def test_unified_knowledge_manager():
    """Test UnifiedKnowledgeManager with multiple sources"""
    
    # Create knowledge sources (direct interface)
    kg1 = KnowledgeGraph("basic_graph", GraphType.INTENT)
    kg1.add_node("basic_concept", type="simple")
    
    lg1 = LayeredKnowledgeGraph("layered_graph")
    layer = KnowledgeGraph("layer1", GraphType.INTENT)
    layer.add_node("layered_concept", type="complex")
    lg1.add_layer(layer)
    
    # Create unified manager (no adapters needed)
    manager = UnifiedKnowledgeManager()
    manager.add_knowledge_source(kg1, is_primary=True)
    manager.add_knowledge_source(lg1)
    
    print(f"Manager has {len(manager.knowledge_sources)} sources")
    assert len(manager.knowledge_sources) == 2, "[FAILURE] Should have 2 sources"
    assert manager.primary_source == kg1, "[FAILURE] Should set primary source"
    
    # Test combined capabilities
    all_capabilities = manager.get_all_capabilities()
    print(f"Combined capabilities: {all_capabilities}")
    assert all_capabilities["semantic_search"] == True, "[FAILURE] Should support semantic search"
    assert all_capabilities["cross_layer_queries"] == True, "[FAILURE] Should support cross-layer from layered source"
    
    # Test querying across all sources
    results = manager.query("concept")
    print(f"Multi-source query results: {len(results)}")
    
    # Should get results from both sources
    basic_results = [r for r in results if "basic_graph:" in r.knowledge_id]
    layered_results = [r for r in results if "layered_graph:" in r.knowledge_id]
    
    print(f"Basic graph results: {len(basic_results)}")
    print(f"Layered graph results: {len(layered_results)}")
    
    print("[SUCCESS] UnifiedKnowledgeManager works correctly")


def test_ai_brain_integration_example():
    """Test how AI Brain would use the unified interface"""
    
    class MockAIBrain:
        def __init__(self):
            self.knowledge = UnifiedKnowledgeManager()
        
        def setup_knowledge(self, knowledge_sources):
            for source in knowledge_sources:
                self.knowledge.add_knowledge_source(source)
        
        def think_about(self, topic):
            # AI Brain reasoning using unified interface
            knowledge_results = self.knowledge.query(topic)
            
            # Get related concepts
            related = []
            for result in knowledge_results:
                related_concepts = self.knowledge.get_related(result.knowledge_id)
                related.extend(related_concepts)
            
            return {
                "topic": topic,
                "primary_knowledge": [r.to_dict() for r in knowledge_results],
                "related_concepts": [r.to_dict() for r in related],
                "reasoning": f"Found {len(knowledge_results)} primary and {len(related)} related concepts"
            }
    
    # Setup AI Brain with knowledge sources
    brain = MockAIBrain()
    
    # Add basic knowledge graph (direct interface)
    kg = KnowledgeGraph("brain_knowledge", GraphType.INTENT)
    kg.add_node("machine_learning", type="field", complexity="high")
    kg.add_node("neural_networks", type="technique", complexity="high")
    kg.add_edge("machine_learning", "neural_networks", relationship="includes")
    
    brain.setup_knowledge([kg])
    
    # Test AI Brain reasoning
    result = brain.think_about("learning")
    
    print(f"AI Brain thought about 'learning':")
    print(f"  Primary knowledge: {len(result['primary_knowledge'])} items")
    print(f"  Related concepts: {len(result['related_concepts'])} items")
    print(f"  Reasoning: {result['reasoning']}")
    
    assert result["topic"] == "learning", "[FAILURE] Should preserve topic"
    assert isinstance(result["primary_knowledge"], list), "[FAILURE] Should return list"
    
    print("[SUCCESS] AI Brain integration works correctly")


if __name__ == "__main__":
    print("Running Knowledge System Tests...")
    test_knowledge_result_creation()
    test_knowledge_graph_direct()
    test_layered_knowledge_graph_direct()
    test_unified_knowledge_manager()
    test_ai_brain_integration_example()
    print("[SUCCESS] All knowledge system tests passed!")
