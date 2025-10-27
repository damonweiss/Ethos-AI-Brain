"""
Matplotlib Knowledge Graph Visualizer - Publication-quality visualization
Creates 2D and 3D visualizations without emojis
"""

import logging
import networkx as nx
from typing import Dict, Any, List, Optional, Union, Tuple, Set

from .knowledge_graph_visualizer_base import KnowledgeGraphVisualizer

logger = logging.getLogger(__name__)


class MatplotlibKnowledgeGraphVisualizer(KnowledgeGraphVisualizer):
    """
    Matplotlib-based knowledge graph visualizer
    Creates publication-quality 2D, 3D, and isometric visualizations
    """

    def render_graph(self, show_labels: bool = True, show_attributes: bool = False,
                     show_external_refs: bool = True, figsize: Tuple[int, int] = (12, 8),
                     layout: str = "spring", save_path: str = None, **kwargs) -> None:
        """
        Create a matplotlib visualization of the graph

        Args:
            figsize: Figure size (width, height)
            show_labels: Whether to show node labels
            show_attributes: Whether to show node attributes in labels
            show_external_refs: Whether to show external references
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'shell')
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("[ERROR] matplotlib not available. Install with: pip install matplotlib")
            return

        fig, ax = plt.subplots(figsize=figsize)

        # Choose layout algorithm
        layout_funcs = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'shell': nx.shell_layout
        }

        if layout not in layout_funcs:
            layout = 'spring'

        pos = layout_funcs[layout](self.graph.graph)

        # Draw nodes
        node_colors = []
        node_sizes = []
        labels = {}

        for node_id in self.graph.graph.nodes():
            node_props = self.get_node_visual_properties(node_id)
            node_colors.append(node_props['color'])
            node_sizes.append(node_props['size'])

            # Create labels
            if show_labels:
                label = node_id
                if show_attributes and node_props['attributes']:
                    # Add key attributes to label
                    attr_parts = []
                    for key in ['status', 'priority', 'health', 'load']:
                        if key in node_props['attributes']:
                            attr_parts.append(f"{key}:{node_props['attributes'][key]}")
                    if attr_parts:
                        label += f"\\n({', '.join(attr_parts)})"
                labels[node_id] = label

        # Draw the graph
        nx.draw_networkx_nodes(self.graph.graph, pos, node_color=node_colors,
                               node_size=node_sizes, alpha=self.style.node_styles['alpha'], ax=ax)

        # Draw edges
        edge_style = self.style.get_edge_style()
        nx.draw_networkx_edges(self.graph.graph, pos, edge_color=edge_style['color'],
                               arrows=True, arrowsize=edge_style['arrow_size'],
                               alpha=edge_style['alpha'], width=edge_style['width'], ax=ax)

        # Draw labels
        if show_labels:
            font_sizes = self.style.get_font_sizes()
            nx.draw_networkx_labels(self.graph.graph, pos, labels,
                                    font_size=font_sizes['node_label'], ax=ax)

        # Draw external references if requested
        if show_external_refs and self.graph.external_references:
            self._draw_external_references(ax, pos)

        # Set title and formatting
        font_sizes = self.style.get_font_sizes()
        ax.set_title(f"Knowledge Graph: {self.graph.graph_id} ({self.graph.graph_type})",
                     fontsize=font_sizes['title'], fontweight='bold')

        # Add stats and legend
        self._add_stats_overlay(ax)
        self._add_node_type_legend(ax)

        ax.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SUCCESS] Graph saved to: {save_path}")
            plt.close()  # Close the figure to free memory
        else:
            plt.show()

    def render_network(self, figsize: Tuple[int, int] = (20, 12), layout: str = "spring",
                       save_path: str = None, **kwargs) -> None:
        """
        Render the entire connected knowledge network with cross-graph interconnects

        Args:
            figsize: Figure size (width, height)
            layout: Layout algorithm
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import numpy as np
        except ImportError:
            print("[ERROR] matplotlib not available. Install with: pip install matplotlib")
            return

        # Get all connected graphs
        network = self.graph.get_connected_knowledge_network()
        from ..knowledge_graph import KnowledgeGraphRegistry
        registry = KnowledgeGraphRegistry()

        # Create a single large plot for the unified network
        fig, ax = plt.subplots(figsize=figsize)

        # Collect all graphs
        all_graphs = [(self.graph.graph_id, self.graph)]
        for graph_id in network['connected_graphs'].keys():
            target_graph = registry.get(graph_id)
            if target_graph:
                all_graphs.append((graph_id, target_graph))

        # Create a combined graph for layout purposes
        combined_graph = nx.Graph()

        # Color scheme for different graphs
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']

        # Add all nodes with graph prefixes
        for i, (graph_id, graph) in enumerate(all_graphs):
            for node_id in graph.nodes():
                prefixed_node = f"{graph_id}:{node_id}"
                combined_graph.add_node(prefixed_node)

                # Add internal edges
                for successor in graph.successors(node_id):
                    prefixed_successor = f"{graph_id}:{successor}"
                    combined_graph.add_edge(prefixed_node, prefixed_successor)

        # Calculate layout for the combined graph
        pos = nx.spring_layout(combined_graph, k=3, iterations=50)

        # Draw each graph's nodes and internal edges
        for i, (graph_id, graph) in enumerate(all_graphs):
            color = colors[i % len(colors)]
            self._draw_graph_cluster(ax, graph_id, graph, pos, color)

        # Draw cross-graph connections (dashed lines)
        cross_connections = self._draw_cross_graph_connections(ax, pos, network)

        # Create legend and add title
        self._add_network_legend(ax, all_graphs, colors, cross_connections)
        self._add_network_title_and_stats(ax, all_graphs, cross_connections)

        ax.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SUCCESS] Network plot saved to: {save_path}")
            plt.close()  # Close the figure to free memory
        else:
            plt.show()

    def _draw_external_references(self, ax, pos):
        """Helper method to draw external references"""
        for local_node, refs in self.graph.external_references.items():
            if local_node in pos:
                local_pos = pos[local_node]

                for ref_key, ref_data in refs.items():
                    target_graph_id = ref_data['target_graph']
                    target_node = ref_data['target_node']
                    relationship = ref_data.get('relationship', 'references')

                    # Draw external reference as dashed arrow pointing outward
                    offset_x = 0.3 if local_pos[0] > 0 else -0.3
                    offset_y = 0.3 if local_pos[1] > 0 else -0.3
                    external_pos = (local_pos[0] + offset_x, local_pos[1] + offset_y)

                    # Draw dashed arrow
                    ax.annotate('', xy=external_pos, xytext=local_pos,
                                arrowprops=dict(arrowstyle='->', linestyle='--',
                                                color=self.style.get_edge_color(True), alpha=0.7))

                    # Add external reference label
                    ax.text(external_pos[0], external_pos[1],
                            f"{target_graph_id}:\\n{target_node}\\n[{relationship}]",
                            fontsize=6, ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))

    def _add_stats_overlay(self, ax):
        """Helper method to add stats overlay"""
        stats = self.graph.get_stats()
        stats_text = f"Nodes: {stats['node_count']}, Edges: {stats['edge_count']}"
        if stats.get('is_dag') is not None:
            stats_text += f", DAG: {stats['is_dag']}"
        if self.graph.external_references:
            total_refs = sum(len(refs) for refs in self.graph.external_references.values())
            stats_text += f", External Refs: {total_refs}"

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                                                   facecolor='lightblue', alpha=0.7))

    def _add_node_type_legend(self, ax):
        """Helper method to add node type legend"""
        unique_types = set()
        for node_id in self.graph.graph.nodes():
            node_type = self.graph.get_node_attributes(node_id).get('type', 'default')
            unique_types.add(node_type)

        if len(unique_types) > 1:
            try:
                import matplotlib.patches as patches
                legend_data = self.style.create_legend_data()
                legend_elements = []

                for item in legend_data:
                    if item['type'] == 'node' and item['label'].lower() in unique_types:
                        legend_elements.append(patches.Patch(color=item['color'], label=item['label']))

                if legend_elements:
                    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            except ImportError:
                pass

    def _draw_graph_cluster(self, ax, graph_id, graph, pos, color):
        """Helper method to draw a single graph cluster"""
        # Get positions for this graph's nodes (with prefixes)
        prefixed_nodes = [f"{graph_id}:{node}" for node in graph.nodes()]
        available_nodes = [node for node in prefixed_nodes if node in pos]

        if not available_nodes:
            return

        # Extract original node names and create position mapping
        original_nodes = []
        cluster_pos = {}
        node_sizes = []

        for prefixed_node in available_nodes:
            original_node = prefixed_node.split(':', 1)[1]
            original_nodes.append(original_node)
            cluster_pos[original_node] = pos[prefixed_node]

            # Calculate node size based on degree
            degree = graph.degree(original_node)
            node_sizes.append(self.style.get_node_size(degree, 400))

        if original_nodes and cluster_pos:
            # Create subgraph with original node names
            subgraph = graph.graph.subgraph(original_nodes)

            # Draw nodes
            nx.draw_networkx_nodes(subgraph, cluster_pos,
                                   node_color=color, node_size=node_sizes,
                                   alpha=0.8, ax=ax)

            # Draw internal edges (solid lines)
            nx.draw_networkx_edges(subgraph, cluster_pos,
                                   edge_color=color, alpha=0.6, width=2, ax=ax)

            # Draw labels
            nx.draw_networkx_labels(subgraph, cluster_pos,
                                    font_size=8, ax=ax)

    def _draw_cross_graph_connections(self, ax, pos, network):
        """Helper method to draw cross-graph connections"""
        cross_connections = []
        for connection in network['cross_graph_connections']:
            source_node = f"{connection['source_graph']}:{connection['source_node']}"
            target_node = f"{connection['target_graph']}:{connection['target_node']}"

            if source_node in pos and target_node in pos:
                # Draw dashed line
                source_pos = pos[source_node]
                target_pos = pos[target_node]

                ax.plot([source_pos[0], target_pos[0]], [source_pos[1], target_pos[1]],
                        'r--', alpha=0.7, linewidth=2)

                # Add arrow
                ax.annotate('', xy=target_pos, xytext=source_pos,
                            arrowprops=dict(arrowstyle='->', color='red',
                                            linestyle='--', alpha=0.7, lw=2))

                cross_connections.append((source_node, target_node))

        return cross_connections

    def _add_network_legend(self, ax, all_graphs, colors, cross_connections):
        """Helper method to add network legend"""
        try:
            import matplotlib.patches as patches
            legend_elements = []
            for i, (graph_id, graph) in enumerate(all_graphs):
                color = colors[i % len(colors)]
                legend_elements.append(patches.Patch(color=color, label=f"{graph_id} ({graph.graph_type})"))

            if cross_connections:
                legend_elements.append(patches.Patch(color='red', label='Cross-graph References'))

            ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        except ImportError:
            pass

    def _add_network_title_and_stats(self, ax, all_graphs, cross_connections):
        """Helper method to add network title and stats"""
        # Add title
        ax.set_title(f"Unified Knowledge Network: {len(all_graphs)} Graphs, {len(cross_connections)} Cross-connections",
                     fontsize=16, fontweight='bold')

        # Add network stats
        total_nodes = sum(len(graph.nodes()) for _, graph in all_graphs)
        total_edges = sum(len(graph.edges()) for _, graph in all_graphs)
        stats_text = f"Total: {total_nodes} nodes, {total_edges} internal edges, {len(cross_connections)} cross-references"

        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3",
                                                      facecolor='lightblue', alpha=0.7))
