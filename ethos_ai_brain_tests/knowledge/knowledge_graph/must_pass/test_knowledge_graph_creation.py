"""
Test Knowledge Graph Creation - Must Pass
Tests basic graph instantiation and initialization
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.knowledge import KnowledgeGraph, GraphType


def test_create_intent_graph():
    """Test creating an intent graph with basic properties"""
    graph_id = "test_intent_graph"
    graph_type = "intent"
    
    kg = KnowledgeGraph(graph_id, graph_type)
    
    assert kg.graph_id == graph_id
    assert kg.graph_type == graph_type
    assert kg.graph is not None


def test_create_execution_graph():
    """Test creating an execution graph"""
    graph_id = "test_execution_graph"
    graph_type = "execution"
    
    kg = KnowledgeGraph(graph_id, graph_type)
    
    assert kg.graph_type == graph_type


def test_graph_has_empty_nodes_initially():
    """Test that new graph starts with no nodes"""
    kg = KnowledgeGraph("empty_test", "intent")
    
    node_count = kg.graph.number_of_nodes()
    
    assert node_count == 0
