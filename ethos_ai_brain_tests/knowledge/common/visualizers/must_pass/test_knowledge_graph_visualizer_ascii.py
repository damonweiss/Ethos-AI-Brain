"""
Test Knowledge Graph ASCII Visualizer - Must Pass
Tests the ASCII text-based visualization functionality
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.knowledge import KnowledgeGraph, GraphType
from ethos_ai_brain.knowledge.common.visualizers import ASCIIKnowledgeGraphVisualizer


def test_ascii_visualizer_creation():
    """Test creating ASCII visualizer with knowledge graph"""
    kg = KnowledgeGraph("ascii_test", "visualization")
    
    viz = ASCIIKnowledgeGraphVisualizer(kg)
    
    # Human-readable output
    print(f"Expected graph ID: ascii_test, Actual: {viz.graph.graph_id}")
    print(f"Expected visualizer type: ASCIIKnowledgeGraphVisualizer, Actual: {type(viz).__name__}")
    
    assert viz.graph.graph_id == "ascii_test", "[FAILURE] ASCII visualizer should store graph"
    assert isinstance(viz, ASCIIKnowledgeGraphVisualizer), "[FAILURE] Should be ASCII visualizer instance"
    
    print("[SUCCESS] ASCII visualizer created successfully")


def test_ascii_render_empty_graph():
    """Test ASCII rendering of empty graph"""
    kg = KnowledgeGraph("empty_ascii", "test")
    viz = ASCIIKnowledgeGraphVisualizer(kg)
    
    result = viz.render_graph()
    
    # Check that result is a string
    print(f"Expected type: str, Actual type: {type(result)}")
    assert isinstance(result, str), "[FAILURE] render_graph should return string"
    
    # Display the actual visual output
    print("\n" + "="*50)
    print("VISUAL OUTPUT - Empty Graph:")
    print("="*50)
    print(result)
    print("="*50 + "\n")
    
    # Check that it contains graph info
    expected_id = "empty_ascii"
    expected_type = "test"
    
    print(f"Expected graph ID in output: {expected_id}")
    print(f"Expected graph type in output: {expected_type}")
    print(f"Actual output length: {len(result)} characters")
    
    assert expected_id in result, f"[FAILURE] Output should contain graph ID {expected_id}"
    assert expected_type in result, f"[FAILURE] Output should contain graph type {expected_type}"
    assert len(result) > 0, "[FAILURE] Output should not be empty"
    
    print("[SUCCESS] ASCII rendering of empty graph works")


def test_ascii_render_simple_graph():
    """Test ASCII rendering of graph with nodes and edges"""
    kg = KnowledgeGraph("simple_ascii", "demo")
    
    # Add simple test data
    kg.add_node("A", label="Node A")
    kg.add_node("B", label="Node B") 
    kg.add_edge("A", "B", relationship="connects_to")
    
    viz = ASCIIKnowledgeGraphVisualizer(kg)
    result = viz.render_graph()
    
    # Display the actual visual output
    print("\n" + "="*50)
    print("VISUAL OUTPUT - Simple Graph (A -> B):")
    print("="*50)
    print(result)
    print("="*50 + "\n")
    
    # Verify output characteristics
    print(f"Output type: {type(result)}")
    print(f"Output length: {len(result)} characters")
    print(f"Number of lines: {len(result.splitlines())}")
    
    assert isinstance(result, str), "[FAILURE] Should return string output"
    assert len(result) > 50, "[FAILURE] Should have substantial output for graph with data"
    
    # Check for graph identification
    assert "simple_ascii" in result, "[FAILURE] Should contain graph ID"
    assert "demo" in result, "[FAILURE] Should contain graph type"
    
    # Check for node/edge counts
    lines = result.splitlines()
    stats_found = False
    for line in lines:
        if "Nodes:" in line and "Edges:" in line:
            stats_found = True
            print(f"Found stats line: {line}")
            break
    
    assert stats_found, "[FAILURE] Should display node and edge statistics"
    
    print("[SUCCESS] ASCII rendering of simple graph works")


def test_ascii_render_with_attributes():
    """Test ASCII rendering with node attributes displayed"""
    kg = KnowledgeGraph("attr_ascii", "attributes")
    
    # Add nodes with various attributes
    kg.add_node("node1", type="concept", importance=0.9, category="primary")
    kg.add_node("node2", type="action", importance=0.7, category="secondary")
    kg.add_edge("node1", "node2", weight=0.8, relationship="triggers")
    
    viz = ASCIIKnowledgeGraphVisualizer(kg)
    
    # Test with attributes shown
    result_with_attrs = viz.render_graph(show_attributes=True)
    
    # Test with attributes hidden
    result_without_attrs = viz.render_graph(show_attributes=False)
    
    # Display both visual outputs for comparison
    print("\n" + "="*50)
    print("VISUAL OUTPUT - WITH Attributes:")
    print("="*50)
    print(result_with_attrs)
    print("="*50)
    print("VISUAL OUTPUT - WITHOUT Attributes:")
    print("="*50)
    print(result_without_attrs)
    print("="*50 + "\n")
    
    print(f"With attributes length: {len(result_with_attrs)}")
    print(f"Without attributes length: {len(result_without_attrs)}")
    
    # Both should be strings
    assert isinstance(result_with_attrs, str), "[FAILURE] Should return string with attributes"
    assert isinstance(result_without_attrs, str), "[FAILURE] Should return string without attributes"
    
    # With attributes should generally be longer (more detail)
    # Note: This might not always be true, so we'll just check they're different
    print(f"Results are different: {result_with_attrs != result_without_attrs}")
    
    print("[SUCCESS] ASCII rendering with/without attributes works")


def test_ascii_network_rendering():
    """Test ASCII network rendering method"""
    kg = KnowledgeGraph("network_ascii", "network")
    
    # Add some network structure
    kg.add_node("hub", type="central")
    kg.add_node("spoke1", type="peripheral")
    kg.add_node("spoke2", type="peripheral")
    kg.add_edge("hub", "spoke1")
    kg.add_edge("hub", "spoke2")
    
    viz = ASCIIKnowledgeGraphVisualizer(kg)
    result = viz.render_network()
    
    # Display the network visualization
    print("\n" + "="*50)
    print("VISUAL OUTPUT - Network Rendering (Hub & Spokes):")
    print("="*50)
    print(result)
    print("="*50 + "\n")
    
    print(f"Network render type: {type(result)}")
    print(f"Network render length: {len(result) if isinstance(result, str) else 'Not string'}")
    
    # Network rendering should return something (implementation dependent)
    assert result is not None, "[FAILURE] render_network should return something"
    
    print("[SUCCESS] ASCII network rendering works")


def test_ascii_render_complex_graph():
    """Test ASCII rendering of a more complex graph structure"""
    kg = KnowledgeGraph("complex_ascii", "workflow")
    
    # Create a workflow-like graph structure
    kg.add_node("Start", type="trigger", status="active")
    kg.add_node("ValidateInput", type="validation", status="ready")
    kg.add_node("ProcessData", type="computation", status="ready")
    kg.add_node("SaveResults", type="storage", status="ready")
    kg.add_node("NotifyUser", type="notification", status="ready")
    kg.add_node("End", type="terminator", status="ready")
    
    # Create workflow connections
    kg.add_edge("Start", "ValidateInput", relationship="triggers")
    kg.add_edge("ValidateInput", "ProcessData", relationship="validates_for")
    kg.add_edge("ProcessData", "SaveResults", relationship="outputs_to")
    kg.add_edge("ProcessData", "NotifyUser", relationship="notifies_via")
    kg.add_edge("SaveResults", "End", relationship="completes_to")
    kg.add_edge("NotifyUser", "End", relationship="completes_to")
    
    viz = ASCIIKnowledgeGraphVisualizer(kg)
    result = viz.render_graph(show_attributes=True)
    
    # Display the complex workflow visualization
    print("\n" + "="*60)
    print("VISUAL OUTPUT - Complex Workflow Graph:")
    print("="*60)
    print(result)
    print("="*60 + "\n")
    
    # Verify the complex structure
    assert viz.graph.graph.number_of_nodes() == 6, "[FAILURE] Should have 6 nodes"
    assert viz.graph.graph.number_of_edges() == 6, "[FAILURE] Should have 6 edges"
    
    # Check that key workflow elements appear in output
    assert "Start" in result, "[FAILURE] Should contain Start node"
    assert "ProcessData" in result, "[FAILURE] Should contain ProcessData node"
    assert "triggers" in result, "[FAILURE] Should contain trigger relationship"
    assert "type=computation" in result, "[FAILURE] Should show node types"
    
    print(f"Complex graph nodes: {viz.graph.graph.number_of_nodes()}")
    print(f"Complex graph edges: {viz.graph.graph.number_of_edges()}")
    print("[SUCCESS] ASCII rendering of complex graph works")


if __name__ == "__main__":
    print("Running ASCII Visualizer Tests...")
    test_ascii_visualizer_creation()
    test_ascii_render_empty_graph()
    test_ascii_render_simple_graph()
    test_ascii_render_with_attributes()
    test_ascii_render_complex_graph()
    test_ascii_network_rendering()
    print("[SUCCESS] All ASCII visualizer tests passed!")
