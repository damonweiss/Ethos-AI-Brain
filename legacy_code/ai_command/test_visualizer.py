#!/usr/bin/env python3
"""
Quick test of the visualizer system
"""

from knowledge_graph import KnowledgeGraph, GraphType
from knowledge_graph_visualizer import ASCIIKnowledgeGraphVisualizer, MatplotlibKnowledgeGraphVisualizer
from knowledge_graph_style import StylePresets

def test_basic_visualization():
    """Test basic visualization functionality"""
    
    print("🧪 TESTING VISUALIZER SYSTEM")
    print("=" * 40)
    
    # Create a simple test graph
    test_graph = KnowledgeGraph(GraphType.TACTICAL, "test_graph")
    
    # Add some nodes
    test_graph.add_node("node1", type="task", status="active")
    test_graph.add_node("node2", type="status", health="good")
    test_graph.add_node("node3", type="output", ready=True)
    
    # Add some edges
    test_graph.add_edge("node1", "node2", relationship="monitors")
    test_graph.add_edge("node1", "node3", relationship="produces")
    
    print(f"Created graph with {len(test_graph.nodes())} nodes and {len(test_graph.edges())} edges")
    
    # Test ASCII visualization
    print("\n📝 ASCII VISUALIZATION:")
    ascii_viz = ASCIIKnowledgeGraphVisualizer(test_graph, StylePresets.professional())
    ascii_viz.print_graph(show_attributes=True)
    
    # Test matplotlib visualization
    print("\n📊 MATPLOTLIB VISUALIZATION:")
    try:
        matplotlib_viz = MatplotlibKnowledgeGraphVisualizer(test_graph, StylePresets.professional())
        print("✅ MatplotlibKnowledgeGraphVisualizer created successfully")
        
        # Test render_graph method
        matplotlib_viz.render_graph(figsize=(8, 6), show_labels=True, show_attributes=True)
        print("✅ render_graph() completed successfully")
        
    except Exception as e:
        print(f"❌ Matplotlib visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_visualization()
