"""
Test Knowledge Graph Edge Operations - Must Pass
Tests adding and managing edges in the base knowledge graph
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.knowledge import KnowledgeGraph, GraphType


def test_add_single_edge():
    """Test adding a single edge between two nodes"""
    kg = KnowledgeGraph("test_edges", "intent")
    
    # Add nodes first
    node1 = "source_node"
    node2 = "target_node"
    kg.add_node(node1)
    kg.add_node(node2)
    
    # Add edge
    result = kg.add_edge(node1, node2)
    
    assert result == True
    assert kg.graph.number_of_edges() == 1
    assert kg.graph.has_edge(node1, node2)


def test_add_edge_with_attributes():
    """Test adding an edge with custom attributes"""
    kg = KnowledgeGraph("test_edge_attrs", "execution")
    
    # Add nodes
    source = "action_start"
    target = "action_end"
    kg.add_node(source)
    kg.add_node(target)
    
    # Add edge with attributes
    relationship = "leads_to"
    weight = 0.8
    
    result = kg.add_edge(source, target, relationship=relationship, weight=weight)
    
    print(f"Expected add_edge result: True, Actual: {result}")
    assert result == True, "[FAILURE] add_edge with attributes should return True"
    
    # Check attributes
    edge_data = kg.graph.edges[source, target]
    actual_relationship = edge_data.get("relationship")
    actual_weight = edge_data.get("weight")
    
    print(f"Expected relationship: {relationship}, Actual: {actual_relationship}")
    print(f"Expected weight: {weight}, Actual: {actual_weight}")
    
    assert actual_relationship == relationship, f"[FAILURE] Edge relationship should be {relationship}"
    assert actual_weight == weight, f"[FAILURE] Edge weight should be {weight}"
    
    print("[SUCCESS] Edge with attributes added correctly")


def test_edge_count_tracking():
    """Test that edge count is tracked correctly"""
    kg = KnowledgeGraph("edge_count_test", "local_rag")
    
    # Add nodes
    nodes = ["node_1", "node_2", "node_3"]
    for node in nodes:
        kg.add_node(node)
    
    # Start with 0 edges
    initial_count = kg.graph.number_of_edges()
    print(f"Initial edge count: {initial_count}")
    assert initial_count == 0, "[FAILURE] Should start with 0 edges"
    
    # Add edges one by one
    edges = [("node_1", "node_2"), ("node_2", "node_3")]
    
    for i, (source, target) in enumerate(edges):
        kg.add_edge(source, target)
        current_count = kg.graph.number_of_edges()
        expected_count = i + 1
        
        print(f"After adding edge {source}->{target} - Expected: {expected_count}, Actual: {current_count}")
        assert current_count == expected_count, f"[FAILURE] Count should be {expected_count}"
    
    print("[SUCCESS] Edge count tracking works correctly")


# Converted to pytest format - no manual main needed
