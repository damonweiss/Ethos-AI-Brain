#!/usr/bin/env python3
"""
Knowledge Graph Visualizer - Hierarchical visualization system for knowledge graphs
Base class with ASCII and Matplotlib subclasses using centralized styling
"""

import logging
import networkx as nx
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from datetime import datetime

logger = logging.getLogger(__name__)

class LayeredKnowledgeGraphVisualizer:
    """
    Specialized visualizer for LayeredKnowledgeGraph instances
    Handles unified network visualization with proper Z-layering
    """

    def __init__(self, layered_graph, style_preset=None):
        """
        Initialize visualizer for a LayeredKnowledgeGraph

        Args:
            layered_graph: LayeredKnowledgeGraph instance
            style_preset: Style preset to use
        """
        from ..style.knowledge_graph_style import StylePresets
        self.layered_graph = layered_graph
        self.style = style_preset or StylePresets.professional()

    def render_unified_3d_network(self, figsize: Tuple[int, int] = (18, 14),
                                  save_path: str = None, **kwargs) -> None:
        """
        Render the complete layered network in 3D with proper Z-layering

        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np
        except ImportError:
            print("[ERROR] matplotlib not available. Install with: pip install matplotlib")
            return

        if not self.layered_graph.graphs:
            print("[WARNING] No graphs in layered network to visualize")
            return

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Get color scheme from layered network
        layer_colors = self.layered_graph.get_all_graph_colors()

        # Get network info
        network_info = self.layered_graph.get_unified_network_info()
        all_positions = self.layered_graph.get_all_nodes_3d()

        # Collect visualization data
        all_node_colors = []
        all_node_sizes = []
        all_labels = {}

        for i, graph_info in enumerate(network_info['graphs']):
            graph_id = graph_info['graph_id']
            graph = self.layered_graph.get_graph(graph_id)
            base_color = layer_colors.get(graph_id, '#808080')  # Default gray if color not found

            for node_id in graph.nodes():
                unified_node_id = f"{graph_id}:{node_id}"
                if unified_node_id in all_positions:
                    # Get visual properties
                    node_attrs = graph.get_node_attributes(node_id)
                    degree = graph.degree(node_id)
                    all_node_colors.append(base_color)
                    all_node_sizes.append(self.style.get_node_size(degree))

                    # Create label
                    label = f"{graph_id}:{node_id}"
                    all_labels[unified_node_id] = label

        # Draw all nodes
        if all_positions:
            xs, ys, zs = zip(*all_positions.values())
            scatter = ax.scatter(xs, ys, zs, c=all_node_colors, s=all_node_sizes, alpha=0.8)

            # Draw internal edges for each graph
            for graph_info in network_info['graphs']:
                graph_id = graph_info['graph_id']
                graph = self.layered_graph.get_graph(graph_id)

                for source, target in graph.edges():
                    unified_source = f"{graph_id}:{source}"
                    unified_target = f"{graph_id}:{target}"

                    if unified_source in all_positions and unified_target in all_positions:
                        source_pos = all_positions[unified_source]
                        target_pos = all_positions[unified_target]

                        ax.plot([source_pos[0], target_pos[0]],
                                [source_pos[1], target_pos[1]],
                                [source_pos[2], target_pos[2]],
                                'gray', alpha=0.6, linewidth=1.5)

            # Draw cross-graph connections
            for connection in network_info['cross_graph_connections']:
                unified_source = f"{connection['source_graph']}:{connection['source_node']}"
                unified_target = f"{connection['target_graph']}:{connection['target_node']}"

                if unified_source in all_positions and unified_target in all_positions:
                    source_pos = all_positions[unified_source]
                    target_pos = all_positions[unified_target]

                    # Draw cross-graph connection as red dashed line
                    ax.plot([source_pos[0], target_pos[0]],
                            [source_pos[1], target_pos[1]],
                            [source_pos[2], target_pos[2]],
                            'red', alpha=0.8, linewidth=2.5, linestyle='--')

            # Draw labels
            for unified_node_id, pos in all_positions.items():
                if unified_node_id in all_labels:
                    ax.text(pos[0], pos[1], pos[2], all_labels[unified_node_id], fontsize=7)

        # Completely remove all axes for pure floating effect
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Remove axis lines and panes completely
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')

        # Hide the axes completely
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.zaxis.set_visible(False)

        # Remove the 3D box outline
        ax.set_axis_off()

        # Set title
        font_sizes = self.style.get_font_sizes()
        ax.set_title(f"Layered Knowledge Network: {self.layered_graph.network_id}",
                     fontsize=font_sizes['title'], fontweight='bold')

        # Add legend for graphs
        try:
            import matplotlib.patches as patches
            legend_elements = []
            for i, graph_info in enumerate(network_info['graphs']):
                graph_id = graph_info['graph_id']
                color = layer_colors.get(graph_id, '#808080')
                graph_type = graph_info['graph_type']
                z_index = graph_info['z_index']
                legend_elements.append(patches.Patch(color=color,
                                                     label=f"{graph_id} ({graph_type}) Z={z_index:.1f}"))

            # Add cross-graph connection legend
            if network_info['cross_connections'] > 0:
                legend_elements.append(patches.Patch(color='red', label='Cross-graph References'))
            ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
        except ImportError:
            pass

        # Add comprehensive network stats
        stats_lines = [
            f"Network: {network_info['network_id']}",
            f"Graphs: {network_info['total_graphs']}, Nodes: {network_info['total_nodes']}, Edges: {network_info['total_edges']}",
            f"Cross-connections: {network_info['cross_connections']}, Z-separation: {network_info['z_separation']:.1f}",
            f"Physics: {'ON' if network_info['physics_enabled'] else 'OFF'}"
        ]
        stats_text = "\n".join(stats_lines)

        fig.text(0.02, 0.02, stats_text, fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SUCCESS] Layered Network saved to: {save_path}")

        plt.show()
