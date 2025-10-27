"""
Test Knowledge Graph Visualizer Base - Must Pass
Tests the abstract base visualizer class functionality
"""

import pytest
import sys
import matplotlib
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.knowledge import KnowledgeGraph, GraphType
from ethos_ai_brain.knowledge.common.visualizers import KnowledgeGraphVisualizer


def test_visualizer_initialization():
    """Test that visualizer can be initialized with a knowledge graph"""
    kg = KnowledgeGraph("test_viz", "intent")
    
    # Add some test data
    kg.add_node("node1", type="concept")
    kg.add_node("node2", type="action")
    kg.add_edge("node1", "node2", relationship="triggers")
    
    # Create a concrete visualizer class for testing
    class TestVisualizer(KnowledgeGraphVisualizer):
        def render_graph(self, **kwargs):
            return "test_render"
        
        def render_network(self, **kwargs):
            return "test_network"
    
    viz = TestVisualizer(kg)
    
    # Human-readable output
    print(f"Expected graph ID: test_viz, Actual: {viz.graph.graph_id}")
    print(f"Expected graph type: intent, Actual: {viz.graph.graph_type}")
    
    assert viz.graph.graph_id == "test_viz", "[FAILURE] Visualizer should store graph reference"
    assert viz.graph.graph_type == "intent", "[FAILURE] Graph type should be preserved"
    assert viz.style is not None, "[FAILURE] Visualizer should have default style"
    
    print("[SUCCESS] Visualizer initialized with graph successfully")


def test_visualizer_abstract_methods():
    """Test that abstract methods are properly defined"""
    kg = KnowledgeGraph("abstract_test", "generic")
    
    # Verify abstract class cannot be instantiated directly
    try:
        viz = KnowledgeGraphVisualizer(kg)
        assert False, "[FAILURE] Should not be able to instantiate abstract class"
    except TypeError as e:
        expected_error = "abstract"
        actual_error = str(e).lower()
        print(f"Expected error contains: {expected_error}, Actual: {actual_error}")
        assert expected_error in actual_error, "[FAILURE] Should get abstract class error"
    
    print("[SUCCESS] Abstract class properly prevents direct instantiation")


def test_visualizer_with_empty_graph():
    """Test visualizer behavior with empty graph"""
    kg = KnowledgeGraph("empty_viz", "test")
    
    class EmptyGraphVisualizer(KnowledgeGraphVisualizer):
        def render_graph(self, **kwargs):
            node_count = len(self.graph.graph.nodes())
            edge_count = len(self.graph.graph.edges())
            return f"Empty graph: {node_count} nodes, {edge_count} edges"
        
        def render_network(self, **kwargs):
            return "Empty network"
    
    viz = EmptyGraphVisualizer(kg)
    result = viz.render_graph()
    
    expected_content = "0 nodes, 0 edges"
    print(f"Expected content: {expected_content}, Actual result: {result}")
    
    assert expected_content in result, "[FAILURE] Should handle empty graph correctly"
    
    print("[SUCCESS] Visualizer handles empty graph correctly")


def test_visualizer_with_populated_graph():
    """Test visualizer with graph containing nodes and edges"""
    kg = KnowledgeGraph("populated_viz", "test")
    
    # Add test data
    kg.add_node("start", type="entry")
    kg.add_node("middle", type="process") 
    kg.add_node("end", type="exit")
    kg.add_edge("start", "middle", weight=1.0)
    kg.add_edge("middle", "end", weight=0.8)
    
    class PopulatedGraphVisualizer(KnowledgeGraphVisualizer):
        def render_graph(self, **kwargs):
            nodes = list(self.graph.graph.nodes())
            edges = list(self.graph.graph.edges())
            return f"Graph: {len(nodes)} nodes, {len(edges)} edges"
        
        def render_network(self, **kwargs):
            return "Network rendered"
    
    viz = PopulatedGraphVisualizer(kg)
    result = viz.render_graph()
    
    expected_nodes = 3
    expected_edges = 2
    
    print(f"Expected: {expected_nodes} nodes, {expected_edges} edges")
    print(f"Actual result: {result}")
    
    assert "3 nodes" in result, "[FAILURE] Should show correct node count"
    assert "2 edges" in result, "[FAILURE] Should show correct edge count"
    
    print("[SUCCESS] Visualizer handles populated graph correctly")


if __name__ == "__main__":
    print("Running Knowledge Graph Visualizer Base Tests...")
    test_visualizer_initialization()
    test_visualizer_abstract_methods()
    test_visualizer_with_empty_graph()
    test_visualizer_with_populated_graph()
    print("[SUCCESS] All visualizer base tests passed!")
