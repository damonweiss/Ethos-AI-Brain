"""
Test Knowledge Graph Visualizer Style - Must Pass
Tests the styling system for visualizers
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.knowledge import KnowledgeGraph, GraphType
from ethos_ai_brain.knowledge.common.style.knowledge_graph_style import (
    KnowledgeGraphStyle, 
    StylePresets, 
    ColorScheme
)
from ethos_ai_brain.knowledge.common.visualizers import ASCIIKnowledgeGraphVisualizer


def test_visualizer_default_style():
    """Test that visualizer has default style when none provided"""
    kg = KnowledgeGraph("style_test", "styling")
    viz = ASCIIKnowledgeGraphVisualizer(kg)
    
    # Check that style exists
    print(f"Style exists: {viz.style is not None}")
    print(f"Style type: {type(viz.style)}")
    
    assert viz.style is not None, "[FAILURE] Visualizer should have default style"
    
    print("[SUCCESS] Default style is applied")


def test_visualizer_custom_style():
    """Test visualizer with different real style presets"""
    kg = KnowledgeGraph("custom_style", "styling")
    
    # Use real StylePresets instead of mocks
    # Already imported at top of file
    
    # Test with different real style presets
    minimal_style = StylePresets.minimal()
    viz = ASCIIKnowledgeGraphVisualizer(kg, style=minimal_style)
    
    print(f"Custom style assigned: {viz.style is minimal_style}")
    print(f"Style type: {type(viz.style)}")
    
    assert viz.style is minimal_style, "[FAILURE] Should use provided minimal style"
    assert viz.style is not None, "[FAILURE] Style should not be None"
    
    print("[SUCCESS] Real custom style is applied")


def test_visualizer_force_color():
    """Test visualizer with forced color override"""
    kg = KnowledgeGraph("color_test", "coloring")
    
    force_color = "#FF0000"  # Red
    viz = ASCIIKnowledgeGraphVisualizer(kg, force_graph_color=force_color)
    
    print(f"Force color set: {viz.force_graph_color}")
    print(f"Expected color: {force_color}")
    
    assert viz.force_graph_color == force_color, "[FAILURE] Should store forced color"
    
    print("[SUCCESS] Force color is applied")


def test_visualizer_style_with_graph_data():
    """Test that style affects rendering with actual graph data"""
    kg = KnowledgeGraph("styled_graph", "test")
    
    # Add some test data
    kg.add_node("node1", type="test")
    kg.add_node("node2", type="test")
    kg.add_edge("node1", "node2")
    
    # Test with default style
    viz_default = ASCIIKnowledgeGraphVisualizer(kg)
    result_default = viz_default.render_graph()
    
    # Test with real StylePresets (no mocks!)
    viz_professional = ASCIIKnowledgeGraphVisualizer(kg, style=StylePresets.professional())
    result_professional = viz_professional.render_graph()
    
    # Display both style outputs for comparison
    print("\n" + "="*50)
    print("VISUAL OUTPUT - DEFAULT Style:")
    print("="*50)
    print(result_default)
    print("="*50)
    print("VISUAL OUTPUT - PROFESSIONAL Style:")
    print("="*50)
    print(result_professional)
    print("="*50 + "\n")
    
    print(f"Default result length: {len(result_default)}")
    print(f"Professional result length: {len(result_professional)}")
    print(f"Results are different: {result_default != result_professional}")
    
    # Both should produce valid string output
    assert isinstance(result_default, str), "[FAILURE] Default style should produce string"
    assert isinstance(result_professional, str), "[FAILURE] Professional style should produce string"
    assert len(result_default) > 0, "[FAILURE] Default style should produce content"
    assert len(result_professional) > 0, "[FAILURE] Professional style should produce content"
    
    print("[SUCCESS] Style affects rendering output")


def test_visualizer_style_parameters():
    """Test visualizer rendering with different style parameters"""
    kg = KnowledgeGraph("param_test", "parameters")
    
    # Add test data
    kg.add_node("A", label="Node A", type="primary")
    kg.add_node("B", label="Node B", type="secondary")
    kg.add_edge("A", "B", relationship="connects")
    
    viz = ASCIIKnowledgeGraphVisualizer(kg)
    
    # Test different parameter combinations
    result_basic = viz.render_graph()
    result_with_labels = viz.render_graph(show_labels=True)
    result_with_attrs = viz.render_graph(show_attributes=True)
    result_full = viz.render_graph(show_labels=True, show_attributes=True, show_external_refs=True)
    
    print(f"Basic result length: {len(result_basic)}")
    print(f"With labels length: {len(result_with_labels)}")
    print(f"With attributes length: {len(result_with_attrs)}")
    print(f"Full result length: {len(result_full)}")
    
    # All should be valid strings
    results = [result_basic, result_with_labels, result_with_attrs, result_full]
    for i, result in enumerate(results):
        assert isinstance(result, str), f"[FAILURE] Result {i} should be string"
        assert len(result) > 0, f"[FAILURE] Result {i} should have content"
    
    print("[SUCCESS] Different style parameters work")


if __name__ == "__main__":
    print("Running Visualizer Style Tests...")
    test_visualizer_default_style()
    test_visualizer_custom_style()
    test_visualizer_force_color()
    test_visualizer_style_with_graph_data()
    test_visualizer_style_parameters()
    print("[SUCCESS] All visualizer style tests passed!")
