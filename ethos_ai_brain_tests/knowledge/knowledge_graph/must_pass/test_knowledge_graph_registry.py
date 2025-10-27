"""
Test Knowledge Graph Registry - Must Pass
Tests the global registry for knowledge graphs
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.knowledge import KnowledgeGraph, KnowledgeGraphRegistry, GraphType


def test_registry_singleton():
    """Test that registry follows singleton pattern"""
    registry1 = KnowledgeGraphRegistry()
    registry2 = KnowledgeGraphRegistry()
    
    # Human-readable output
    same_instance = registry1 is registry2
    print(f"Expected same instance: True, Actual: {same_instance}")
    
    assert same_instance == True, "[FAILURE] Registry should be singleton"
    
    print("[SUCCESS] Registry singleton pattern works")


def test_register_single_graph():
    """Test registering a single graph"""
    registry = KnowledgeGraphRegistry()
    
    # Clear any existing graphs
    registry._graphs.clear()
    
    # Create and register graph
    kg = KnowledgeGraph("test_register", "intent")
    registry.register(kg)
    
    # Check registration
    retrieved = registry.get("test_register")
    
    print(f"Expected retrieved graph ID: test_register, Actual: {retrieved.graph_id if retrieved else None}")
    
    assert retrieved is not None, "[FAILURE] Should retrieve registered graph"
    assert retrieved.graph_id == "test_register", "[FAILURE] Retrieved graph should have correct ID"
    assert retrieved is kg, "[FAILURE] Should retrieve same graph instance"
    
    print("[SUCCESS] Single graph registered successfully")


def test_register_multiple_graphs():
    """Test registering multiple graphs"""
    registry = KnowledgeGraphRegistry()
    registry._graphs.clear()
    
    # Create multiple graphs
    graphs = [
        KnowledgeGraph("graph_1", "intent"),
        KnowledgeGraph("graph_2", "execution"),
        KnowledgeGraph("graph_3", "local_rag")
    ]
    
    # Register all graphs
    for graph in graphs:
        registry.register(graph)
    
    # Verify all are registered
    for graph in graphs:
        retrieved = registry.get(graph.graph_id)
        
        print(f"Checking graph {graph.graph_id} - Retrieved: {retrieved is not None}")
        
        assert retrieved is not None, f"[FAILURE] Graph {graph.graph_id} should be registered"
        assert retrieved is graph, f"[FAILURE] Should retrieve same instance for {graph.graph_id}"
    
    print("[SUCCESS] Multiple graphs registered successfully")


def test_unregister_graph():
    """Test unregistering a graph"""
    registry = KnowledgeGraphRegistry()
    registry._graphs.clear()
    
    # Register graph
    kg = KnowledgeGraph("test_unregister", "intent")
    registry.register(kg)
    
    # Verify it's registered
    retrieved_before = registry.get("test_unregister")
    assert retrieved_before is not None, "[FAILURE] Graph should be registered before unregister"
    
    # Unregister
    registry.unregister("test_unregister")
    
    # Verify it's gone
    retrieved_after = registry.get("test_unregister")
    
    print(f"Expected after unregister: None, Actual: {retrieved_after}")
    
    assert retrieved_after is None, "[FAILURE] Graph should be None after unregister"
    
    print("[SUCCESS] Graph unregistered successfully")


# Converted to pytest format - no manual main needed
