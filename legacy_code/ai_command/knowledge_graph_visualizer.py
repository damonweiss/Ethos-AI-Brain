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

from knowledge_graph_style import KnowledgeGraphStyle, StylePresets, ColorScheme

logger = logging.getLogger(__name__)

class KnowledgeGraphVisualizer(ABC):
    """
    Abstract base class for all knowledge graph visualizers
    Provides common functionality and enforces interface consistency
    """
    
    def __init__(self, graph: 'KnowledgeGraph', style: 'KnowledgeGraphStyle' = None, 
                 force_graph_color: str = None):
        """
        Initialize visualizer with a knowledge graph and style
        
        Args:
            graph: KnowledgeGraph instance to visualize
            style: Style configuration (uses default if None)
            force_graph_color: Override all node colors with this single color
        """
        from knowledge_graph_style import StylePresets
        self.graph = graph
        self.style = style or StylePresets.professional()
        self.force_graph_color = force_graph_color
    
    @abstractmethod
    def render_graph(self, show_labels: bool = True, show_attributes: bool = False, 
                    show_external_refs: bool = True, **kwargs) -> Any:
        """
        Abstract method to render the graph
        Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def render_network(self, **kwargs) -> Any:
        """
        Abstract method to render the entire connected network
        Must be implemented by subclasses
        """
        pass
    
    def get_node_visual_properties(self, node_id: str) -> Dict[str, Any]:
        """
        Get visual properties for a node based on its attributes and style
        
        Args:
            node_id: The node to get properties for
            
        Returns:
            Dictionary of visual properties
        """
        node_attrs = self.graph.get_node_attributes(node_id)
        node_type = node_attrs.get('type', 'default')
        degree = self.graph.degree(node_id)
        
        # Use forced color if specified, otherwise use style-based color
        node_color = self.force_graph_color if self.force_graph_color else self.style.get_node_color(node_type)
        
        return {
            'color': node_color,
            'size': self.style.get_node_size(degree),
            'shape': self.style.get_node_shape(node_type),
            'type': node_type,
            'attributes': node_attrs
        }
    
    def get_edge_visual_properties(self, source: str, target: str, is_external_ref: bool = False) -> Dict[str, Any]:
        """
        Get visual properties for an edge
        
        Args:
            source: Source node ID
            target: Target node ID
            is_external_ref: Whether this is an external reference
            
        Returns:
            Dictionary of visual properties
        """
        edge_attrs = self.graph.get_edge_attributes(source, target) if not is_external_ref else {}
        edge_style = self.style.get_edge_style(is_external_ref)
        
        return {
            **edge_style,
            'relationship': edge_attrs.get('relationship', 'connects'),
            'attributes': edge_attrs
        }
    
    def set_style(self, style: KnowledgeGraphStyle):
        """Change the visualization style"""
        self.style = style
    
class ASCIIKnowledgeGraphVisualizer(KnowledgeGraphVisualizer):
    """
    ASCII-based knowledge graph visualizer
    Creates text-based representations using Unicode symbols and styling
    """
    
    def render_graph(self, show_labels: bool = True, show_attributes: bool = False, 
                    show_external_refs: bool = True, **kwargs) -> str:
        """
        Create a quick ASCII visualization of the graph
        
        Args:
            show_attributes: Whether to show node attributes
            show_external_refs: Whether to show external references
            
        Returns:
            ASCII art representation of the graph
        """
        symbols = self.style.get_ascii_symbols()
        lines = []
        
        # Header with styled title
        lines.append(f"📊 {self.graph.graph_id} ({self.graph.graph_type.value})")
        lines.append("=" * (len(self.graph.graph_id) + len(self.graph.graph_type.value) + 5))
        
        # Show basic stats
        stats = self.graph.get_stats()
        lines.append(f"Nodes: {stats['node_count']}, Edges: {stats['edge_count']}")
        if stats.get('is_dag') is not None:
            lines.append(f"DAG: {stats['is_dag']}, Connected: {stats['is_connected']}")
        lines.append("")
        
        # Show nodes with their connections
        if len(self.graph.nodes()) == 0:
            lines.append("(Empty graph)")
        else:
            for node_id in self.graph.nodes():
                # Get visual properties for this node
                node_props = self.get_node_visual_properties(node_id)
                
                # Node representation with styled symbol
                node_symbol = symbols.get(node_props['type'], symbols['default'])
                node_line = f"{node_symbol} {node_id}"
                
                if show_attributes and node_props['attributes']:
                    # Show key attributes
                    key_attrs = []
                    for key in ['type', 'status', 'priority', 'health', 'load']:
                        if key in node_props['attributes']:
                            key_attrs.append(f"{key}={node_props['attributes'][key]}")
                    if key_attrs:
                        node_line += f" ({', '.join(key_attrs)})"
                
                lines.append(node_line)
                
                # Show outgoing edges with styled symbols
                successors = list(self.graph.successors(node_id) if isinstance(self.graph.graph, nx.DiGraph) else self.graph.neighbors(node_id))
                
                for successor in successors:
                    edge_props = self.get_edge_visual_properties(node_id, successor)
                    edge_line = f"  {symbols['corner']}{symbols['edge']}{symbols['arrow']} {successor}"
                    if edge_props['relationship']:
                        edge_line += f" [{edge_props['relationship']}]"
                    lines.append(edge_line)
                
                # Show external references with special styling
                if show_external_refs and node_id in self.graph.external_references:
                    for ref_key, ref_data in self.graph.external_references[node_id].items():
                        ref_line = f"  {symbols['external_ref']} {ref_data['target_graph']}:{ref_data['target_node']}"
                        if ref_data.get('relationship'):
                            ref_line += f" [{ref_data['relationship']}]"
                        lines.append(ref_line)
                
                lines.append("")  # Empty line between nodes
        
        # Show external reference summary
        if show_external_refs and self.graph.external_references:
            lines.append("🔗 External References:")
            total_refs = sum(len(refs) for refs in self.graph.external_references.values())
            connected_graphs = set()
            for refs in self.graph.external_references.values():
                for ref_data in refs.values():
                    connected_graphs.add(ref_data['target_graph'])
            lines.append(f"  {total_refs} references to {len(connected_graphs)} graphs")
            for graph_id in connected_graphs:
                lines.append(f"  - {graph_id}")
        
        return "\n".join(lines)
    
    def render_network(self, **kwargs) -> str:
        """
        Render the entire connected knowledge network in ASCII
        
        Returns:
            ASCII representation of the network
        """
        # Get all connected graphs
        network = self.graph.get_connected_knowledge_network()
        lines = []
        
        lines.append("🌐 CONNECTED KNOWLEDGE NETWORK")
        lines.append("=" * 40)
        
        # Show root graph
        lines.append(f"ROOT: {self.graph.graph_id}")
        lines.append(self.render_graph(**kwargs))
        
        # Show connected graphs
        from knowledge_graph import KnowledgeGraphRegistry
        registry = KnowledgeGraphRegistry()
        
        for graph_id in network['connected_graphs'].keys():
            target_graph = registry.get(graph_id)
            if target_graph:
                lines.append(f"\nCONNECTED: {graph_id}")
                lines.append("-" * 20)
                # Create visualizer for connected graph
                connected_viz = ASCIIKnowledgeGraphVisualizer(target_graph, self.style)
                lines.append(connected_viz.render_graph(show_external_refs=False, **kwargs))
        
        return "\n".join(lines)
    
    def print_graph(self, **kwargs):
        """Print the graph visualization to console"""
        print(self.render_graph(**kwargs))
    
    def print_network(self, **kwargs):
        """Print the network visualization to console"""
        print(self.render_network(**kwargs))

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
            print("❌ matplotlib not available. Install with: pip install matplotlib")
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
        ax.set_title(f"Knowledge Graph: {self.graph.graph_id} ({self.graph.graph_type.value})", 
                    fontsize=font_sizes['title'], fontweight='bold')
        
        # Add stats and legend
        self._add_stats_overlay(ax)
        self._add_node_type_legend(ax)
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Graph saved to: {save_path}")
        
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
            print("❌ matplotlib not available. Install with: pip install matplotlib")
            return
        
        # Get all connected graphs
        network = self.graph.get_connected_knowledge_network()
        from knowledge_graph import KnowledgeGraphRegistry
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
            print(f"📊 Network plot saved to: {save_path}")
        
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
                legend_elements.append(patches.Patch(color=color, label=f"{graph_id} ({graph.graph_type.value})"))
            
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
    
    def render_3d_graph(self, figsize: Tuple[int, int] = (14, 10), show_labels: bool = True, 
                       show_attributes: bool = False, positioning_strategy: str = "layered",
                       save_path: str = None, **kwargs) -> None:
        """
        Create a 3D matplotlib visualization of the graph
        
        Args:
            figsize: Figure size (width, height)
            show_labels: Whether to show node labels
            show_attributes: Whether to show node attributes in labels
            positioning_strategy: 3D positioning strategy
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np
        except ImportError:
            print("❌ matplotlib or mpl_toolkits not available. Install with: pip install matplotlib")
            return
        
        # Ensure 3D positions are assigned
        if not self.graph.node_3d_positions:
            self.graph.auto_assign_3d_positions(positioning_strategy)
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare node data
        node_positions = {}
        node_colors = []
        node_sizes = []
        labels = {}
        
        for node_id in self.graph.nodes():
            pos_3d = self.graph.get_node_3d_position(node_id)
            if pos_3d:
                node_positions[node_id] = pos_3d
                
                # Get visual properties
                node_props = self.get_node_visual_properties(node_id)
                node_colors.append(node_props['color'])
                node_sizes.append(node_props['size'])
                
                # Labels
                if show_labels:
                    label = node_id
                    if show_attributes and node_props['attributes']:
                        attr_parts = []
                        for key in ['status', 'priority', 'health', 'load']:
                            if key in node_props['attributes']:
                                attr_parts.append(f"{key}:{node_props['attributes'][key]}")
                        if attr_parts:
                            label += f"\\n({', '.join(attr_parts)})"
                    labels[node_id] = label
        
        # Draw all nodes (all should have positions now)
        if node_positions:
            xs, ys, zs = zip(*node_positions.values())
            scatter = ax.scatter(xs, ys, zs, c=node_colors, s=node_sizes, alpha=0.8)
        else:
            print("⚠️ Warning: No nodes have 3D positions!")
        
        # Draw edges (after all nodes are positioned)
        for source, target in self.graph.edges():
            if source in node_positions and target in node_positions:
                source_pos = node_positions[source]
                target_pos = node_positions[target]
                
                ax.plot([source_pos[0], target_pos[0]], 
                       [source_pos[1], target_pos[1]], 
                       [source_pos[2], target_pos[2]], 
                       'gray', alpha=0.6, linewidth=2)
        
        # Draw labels for positioned nodes
        if show_labels:
            for node_id, pos in node_positions.items():
                if node_id in labels:
                    ax.text(pos[0], pos[1], pos[2], labels[node_id], fontsize=8)
        
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
        ax.set_title(f"3D Knowledge Graph: {self.graph.graph_id} ({self.graph.graph_type.value})", 
                    fontsize=font_sizes['title'], fontweight='bold')
        
        # Add stats
        stats = self.graph.get_stats()
        stats_text = f"Nodes: {stats['node_count']}, Edges: {stats['edge_count']}, Strategy: {positioning_strategy}"
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 3D Graph saved to: {save_path}")
        
        plt.show()
    
    def render_3d_network(self, figsize: Tuple[int, int] = (16, 12), 
                         positioning_strategy: str = "layered", save_path: str = None, **kwargs) -> None:
        """
        Create a unified 3D visualization showing all connected graphs in one space
        
        Args:
            figsize: Figure size (width, height)
            positioning_strategy: 3D positioning strategy for all graphs
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np
        except ImportError:
            print("❌ matplotlib or mpl_toolkits not available. Install with: pip install matplotlib")
            return
        
        # Get all connected graphs
        network = self.graph.get_connected_knowledge_network()
        from knowledge_graph import KnowledgeGraphRegistry
        registry = KnowledgeGraphRegistry()
        
        # Collect all graphs with their instances
        all_graphs = [(self.graph.graph_id, self.graph)]
        for graph_id in network['connected_graphs'].keys():
            target_graph = registry.get(graph_id)
            if target_graph:
                all_graphs.append((graph_id, target_graph))
        
        # Ensure all graphs have 3D positions
        for graph_id, graph in all_graphs:
            if not graph.node_3d_positions:
                graph.auto_assign_3d_positions(positioning_strategy)
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Color scheme for different graphs
        graph_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
        
        # Collect all nodes and their positions
        all_node_positions = {}
        all_node_colors = []
        all_node_sizes = []
        all_labels = {}
        
        for i, (graph_id, graph) in enumerate(all_graphs):
            base_color = graph_colors[i % len(graph_colors)]
            
            for node_id in graph.nodes():
                pos_3d = graph.get_node_3d_position(node_id)
                if pos_3d:
                    # Create unique node identifier for the unified space
                    unified_node_id = f"{graph_id}:{node_id}"
                    all_node_positions[unified_node_id] = pos_3d
                    
                    # Get visual properties but use graph-specific coloring
                    node_attrs = graph.get_node_attributes(node_id)
                    degree = graph.degree(node_id)
                    all_node_colors.append(base_color)  # Use graph color instead of type color
                    all_node_sizes.append(self.style.get_node_size(degree))
                    
                    # Create label
                    label = f"{graph_id}:\\n{node_id}"
                    all_labels[unified_node_id] = label
        
        # Draw all nodes
        if all_node_positions:
            xs, ys, zs = zip(*all_node_positions.values())
            scatter = ax.scatter(xs, ys, zs, c=all_node_colors, s=all_node_sizes, alpha=0.8)
            
            # Draw internal edges for each graph
            for graph_id, graph in all_graphs:
                for source, target in graph.edges():
                    unified_source = f"{graph_id}:{source}"
                    unified_target = f"{graph_id}:{target}"
                    
                    if unified_source in all_node_positions and unified_target in all_node_positions:
                        source_pos = all_node_positions[unified_source]
                        target_pos = all_node_positions[unified_target]
                        
                        ax.plot([source_pos[0], target_pos[0]], 
                               [source_pos[1], target_pos[1]], 
                               [source_pos[2], target_pos[2]], 
                               'gray', alpha=0.6, linewidth=1.5)
            
            # Draw cross-graph connections (external references)
            for connection in network['cross_graph_connections']:
                unified_source = f"{connection['source_graph']}:{connection['source_node']}"
                unified_target = f"{connection['target_graph']}:{connection['target_node']}"
                
                if unified_source in all_node_positions and unified_target in all_node_positions:
                    source_pos = all_node_positions[unified_source]
                    target_pos = all_node_positions[unified_target]
                    
                    # Draw cross-graph connection as red dashed line
                    ax.plot([source_pos[0], target_pos[0]], 
                           [source_pos[1], target_pos[1]], 
                           [source_pos[2], target_pos[2]], 
                           'red', alpha=0.8, linewidth=2.5, linestyle='--')
            
            # Draw labels
            for unified_node_id, pos in all_node_positions.items():
                if unified_node_id in all_labels:
                    ax.text(pos[0], pos[1], pos[2], all_labels[unified_node_id], fontsize=7)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        font_sizes = self.style.get_font_sizes()
        ax.set_title(f"Unified 3D Knowledge Network: {len(all_graphs)} Graphs, {len(network['cross_graph_connections'])} Cross-connections", 
                    fontsize=font_sizes['title'], fontweight='bold')
        
        # Add legend for graphs
        try:
            import matplotlib.patches as patches
            legend_elements = []
            for i, (graph_id, graph) in enumerate(all_graphs):
                color = graph_colors[i % len(graph_colors)]
                legend_elements.append(patches.Patch(color=color, label=f"{graph_id} ({graph.graph_type.value})"))
            
            # Add cross-graph connection legend
            legend_elements.append(patches.Patch(color='red', label='Cross-graph References'))
            ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
        except ImportError:
            pass
        
        # Add network stats
        total_nodes = sum(len(graph.nodes()) for _, graph in all_graphs)
        total_edges = sum(len(graph.edges()) for _, graph in all_graphs)
        stats_text = f"Total: {total_nodes} nodes, {total_edges} internal edges, {len(network['cross_graph_connections'])} cross-references"
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 3D Network saved to: {save_path}")
        
        plt.show()

class BrainKnowledgeGraphVisualizer:
    """
    Specialized visualizer for BrainKnowledgeGraph instances
    Handles unified network visualization with proper Z-layering
    """
    
    def __init__(self, brain_graph, style_preset=None):
        """
        Initialize visualizer for a BrainKnowledgeGraph
        
        Args:
            brain_graph: BrainKnowledgeGraph instance
            style_preset: Style preset to use
        """
        from knowledge_graph_style import StylePresets
        self.brain_graph = brain_graph
        self.style = style_preset or StylePresets.professional()
    
    def render_unified_3d_network(self, figsize: Tuple[int, int] = (18, 14), 
                                 save_path: str = None, **kwargs) -> None:
        """
        Render the complete brain network in 3D with proper Z-layering
        
        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np
        except ImportError:
            print("❌ matplotlib or mpl_toolkits not available. Install with: pip install matplotlib")
            return
        
        if not self.brain_graph.graphs:
            print("⚠️ No graphs in brain network to visualize")
            return
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get color scheme from brain network
        brain_colors = self.brain_graph.get_all_graph_colors()
        
        # Get network info
        network_info = self.brain_graph.get_unified_network_info()
        all_positions = self.brain_graph.get_all_nodes_3d()
        
        # Collect visualization data
        all_node_colors = []
        all_node_sizes = []
        all_labels = {}
        
        for i, graph_info in enumerate(network_info['graphs']):
            graph_id = graph_info['graph_id']
            graph = self.brain_graph.get_graph(graph_id)
            base_color = brain_colors.get(graph_id, '#808080')  # Default gray if color not found
            
            for node_id in graph.nodes():
                unified_node_id = f"{graph_id}:{node_id}"
                if unified_node_id in all_positions:
                    # Get visual properties
                    node_attrs = graph.get_node_attributes(node_id)
                    degree = graph.degree(node_id)
                    all_node_colors.append(base_color)
                    all_node_sizes.append(self.style.get_node_size(degree))
                    
                    # Create label
                    label = f"{graph_id}:\\n{node_id}"
                    all_labels[unified_node_id] = label
        
        # Draw all nodes
        if all_positions:
            xs, ys, zs = zip(*all_positions.values())
            scatter = ax.scatter(xs, ys, zs, c=all_node_colors, s=all_node_sizes, alpha=0.8)
            
            # Draw internal edges for each graph
            for graph_info in network_info['graphs']:
                graph_id = graph_info['graph_id']
                graph = self.brain_graph.get_graph(graph_id)
                
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
        ax.set_title(f"Brain Knowledge Network: {self.brain_graph.brain_id}", 
                    fontsize=font_sizes['title'], fontweight='bold')
        
        # Add legend for graphs
        try:
            import matplotlib.patches as patches
            legend_elements = []
            for i, graph_info in enumerate(network_info['graphs']):
                graph_id = graph_info['graph_id']
                color = brain_colors.get(graph_id, '#808080')
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
            f"Brain: {network_info['brain_id']}",
            f"Graphs: {network_info['total_graphs']}, Nodes: {network_info['total_nodes']}, Edges: {network_info['total_edges']}",
            f"Cross-connections: {network_info['cross_connections']}, Z-separation: {network_info['z_separation']:.1f}",
            f"Physics: {'ON' if network_info['physics_enabled'] else 'OFF'}"
        ]
        stats_text = "\\n".join(stats_lines)
        
        fig.text(0.02, 0.02, stats_text, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Brain Network saved to: {save_path}")
        
        plt.show()
