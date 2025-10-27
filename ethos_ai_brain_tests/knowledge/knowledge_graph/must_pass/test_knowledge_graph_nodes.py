"""
Test Knowledge Graph Node Operations - Must Pass
Tests adding and managing nodes in the base knowledge graph
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.knowledge import KnowledgeGraph, GraphType


def test_add_single_node():
    """Test adding a single node to the graph"""
    kg = KnowledgeGraph("test_nodes", "intent")
    
    node_id = "test_node_1"
    result = kg.add_node(node_id)
    
    # Human-readable output
    print(f"Expected add_node result: True, Actual: {result}")
    
    assert result == True, "[FAILURE] add_node should return True on success"
    
    # Verify node exists
    node_exists = kg.graph.has_node(node_id)
    print(f"Expected node exists: True, Actual: {node_exists}")
    
    assert node_exists == True, f"[FAILURE] Node {node_id} should exist in graph"
    
    print("[SUCCESS] Single node added successfully")


def test_add_node_with_attributes():
    """Test adding a node with custom attributes"""
    kg = KnowledgeGraph("test_attrs", "intent")
    
    node_id = "attr_node"
    name = "Test Node"
    node_type = "concept"
    
    result = kg.add_node(node_id, name=name, type=node_type)
    
    print(f"Expected add_node result: True, Actual: {result}")
    assert result == True, "[FAILURE] add_node with attributes should return True"
    
    # Check attributes
    node_data = kg.graph.nodes[node_id]
    actual_name = node_data.get("name")
    actual_type = node_data.get("type")
    
    print(f"Expected name: {name}, Actual: {actual_name}")
    print(f"Expected type: {node_type}, Actual: {actual_type}")
    
    assert actual_name == name, f"[FAILURE] Node name should be {name}"
    assert actual_type == node_type, f"[FAILURE] Node type should be {node_type}"
    
    print("[SUCCESS] Node with attributes added correctly")


def test_node_count_tracking():
    """Test that node count is tracked correctly"""
    kg = KnowledgeGraph("count_test", "local_rag")
    
    # Start with 0
    initial_count = kg.graph.number_of_nodes()
    print(f"Initial count: {initial_count}")
    assert initial_count == 0, "[FAILURE] Should start with 0 nodes"
    
    # Add nodes one by one
    for i in range(3):
        kg.add_node(f"node_{i}")
        current_count = kg.graph.number_of_nodes()
        expected_count = i + 1
        
        print(f"After adding node_{i} - Expected: {expected_count}, Actual: {current_count}")
        assert current_count == expected_count, f"[FAILURE] Count should be {expected_count}"
    
    print("[SUCCESS] Node count tracking works correctly")


# Converted to pytest format - no manual main needed
