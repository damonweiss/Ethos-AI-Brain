#!/usr/bin/env python3
"""
Knowledge Graph Base Class - Foundation for all knowledge graph types
Part of Sprint B: Knowledge Graph Foundation
"""

import json
import logging
import networkx as nx
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum

# Import the visualizer from separate file
from knowledge_graph_visualizer import KnowledgeGraphVisualizer

logger = logging.getLogger(__name__)

class GraphType(str, Enum):
    """Types of knowledge graphs"""
    MISSION = "mission"
    TACTICAL = "tactical"
    LOCAL_RAG = "local_rag"
    CLOUD_RAG = "cloud_rag"

class KnowledgeGraphRegistry:
    """
    Global registry for knowledge graphs to enable cross-graph operations
    Singleton pattern for shared access across all graphs
    """
    _instance = None
    _graphs: Dict[str, 'KnowledgeGraph'] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(self, graph: 'KnowledgeGraph'):
        """Register a knowledge graph"""
        self._graphs[graph.graph_id] = graph
        logger.info(f"Registered graph: {graph.graph_id}")
    
    def unregister(self, graph_id: str):
        """Unregister a knowledge graph"""
        if graph_id in self._graphs:
            del self._graphs[graph_id]
            logger.info(f"Unregistered graph: {graph_id}")
    
    def get(self, graph_id: str) -> Optional['KnowledgeGraph']:
        """Get a registered graph by ID"""
        return self._graphs.get(graph_id)
    
    def list_graphs(self) -> List[str]:
        """List all registered graph IDs"""
        return list(self._graphs.keys())
    
    def clear(self):
        """Clear all registered graphs"""
        self._graphs.clear()
        logger.info("Cleared all registered graphs")

class KnowledgeGraph:
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
    
    def create_matplotlib_plot(self, figsize: Tuple[int, int] = (12, 8), show_labels: bool = True, 
                              show_attributes: bool = False, show_external_refs: bool = True,
                              layout: str = "spring", save_path: str = None) -> None:
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
            node_attrs = self.graph.get_node_attributes(node_id)
            
            # Color based on node type
            node_type = node_attrs.get('type', 'default')
            color_map = {
                'mission': '#FF6B6B',      # Red
                'coordination': '#4ECDC4',  # Teal
                'task': '#45B7D1',         # Blue
                'status': '#96CEB4',       # Green
                'requirement': '#FECA57',   # Yellow
                'resource': '#E17055',     # Orange
                'output': '#A29BFE',       # Purple
                'default': '#95A5A6'       # Gray
            }
            node_colors.append(color_map.get(node_type, color_map['default']))
            
            # Size based on importance or connections
            degree = self.graph.degree(node_id)
            node_sizes.append(300 + degree * 100)
            
            # Create labels
            if show_labels:
                label = node_id
                if show_attributes and node_attrs:
                    # Add key attributes to label
                    attr_parts = []
                    for key in ['status', 'priority', 'health', 'load']:
                        if key in node_attrs:
                            attr_parts.append(f"{key}:{node_attrs[key]}")
                    if attr_parts:
                        label += f"\n({', '.join(attr_parts)})"
                labels[node_id] = label
        
        # Draw the graph
        nx.draw_networkx_nodes(self.graph.graph, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.8, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph.graph, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, alpha=0.6, ax=ax)
        
        # Draw labels
        if show_labels:
            nx.draw_networkx_labels(self.graph.graph, pos, labels, font_size=8, ax=ax)
        
        # Draw external references if requested
        if show_external_refs and self.graph.external_references:
            self._draw_external_references(ax, pos)
        
        # Set title and formatting
        ax.set_title(f"Knowledge Graph: {self.graph.graph_id} ({self.graph.graph_type.value})", 
                    fontsize=14, fontweight='bold')
        
        # Add stats and legend
        self._add_stats_overlay(ax)
        self._add_node_type_legend(ax)
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Graph saved to: {save_path}")
        
        plt.show()
    
    def create_network_plot(self, figsize: Tuple[int, int] = (20, 12), layout: str = "spring", 
                           save_path: str = None) -> None:
        """
        Plot the entire connected knowledge network with cross-graph interconnects
        
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
        registry = KnowledgeGraphRegistry()
        
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
                                             color='red', alpha=0.7))
                    
                    # Add external reference label
                    ax.text(external_pos[0], external_pos[1], 
                           f"{target_graph_id}:\n{target_node}\n[{relationship}]",
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
                legend_elements = []
                color_map = {
                    'mission': '#FF6B6B', 'coordination': '#4ECDC4', 'task': '#45B7D1',
                    'status': '#96CEB4', 'requirement': '#FECA57', 'resource': '#E17055',
                    'output': '#A29BFE', 'default': '#95A5A6'
                }
                for node_type in sorted(unique_types):
                    color = color_map.get(node_type, color_map['default'])
                    legend_elements.append(patches.Patch(color=color, label=node_type))
                
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            except ImportError:
                pass
    
    def _draw_graph_cluster(self, ax, graph_id, graph, pos, color):
        """Helper method to draw a single graph cluster"""
        # Get positions for this graph's nodes
        graph_nodes = [f"{graph_id}:{node}" for node in graph.nodes()]
        graph_pos = {node: pos[node] for node in graph_nodes if node in pos}
        
        # Draw nodes
        node_sizes = []
        for node in graph_nodes:
            if node in pos:
                original_node = node.split(':', 1)[1]
                degree = graph.degree(original_node)
                node_sizes.append(400 + degree * 100)
        
        nx.draw_networkx_nodes(graph.graph.subgraph([n.split(':', 1)[1] for n in graph_nodes]), 
                              {n.split(':', 1)[1]: pos[n] for n in graph_nodes if n in pos}, 
                              node_color=color, node_size=node_sizes, 
                              alpha=0.8, ax=ax)
        
        # Draw internal edges (solid lines)
        internal_edges = []
        for node in graph.nodes():
            for successor in graph.successors(node):
                prefixed_edge = (f"{graph_id}:{node}", f"{graph_id}:{successor}")
                internal_edges.append(prefixed_edge)
        
        if internal_edges:
            # Create a temporary graph for edge drawing
            temp_graph = nx.Graph()
            temp_graph.add_edges_from(internal_edges)
            nx.draw_networkx_edges(temp_graph, pos, edgelist=internal_edges,
                                 edge_color=color, alpha=0.6, width=2, ax=ax)
        
        # Draw labels
        labels = {f"{graph_id}:{node}": node for node in graph.nodes()}
        nx.draw_networkx_labels(graph.graph, graph_pos, 
                               {n.split(':', 1)[1]: n.split(':', 1)[1] for n in graph_nodes if n in pos}, 
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
    
    def create_3d_plot(self, figsize: Tuple[int, int] = (14, 10), show_labels: bool = True, 
                      show_attributes: bool = False, positioning_strategy: str = "layered",
                      save_path: str = None) -> None:
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
                
                # Color and size based on node attributes
                node_attrs = self.graph.get_node_attributes(node_id)
                node_type = node_attrs.get('type', 'default')
                
                color_map = {
                    'mission': '#FF6B6B',      # Red
                    'coordination': '#4ECDC4',  # Teal
                    'task': '#45B7D1',         # Blue
                    'status': '#96CEB4',       # Green
                    'requirement': '#FECA57',   # Yellow
                    'resource': '#E17055',     # Orange
                    'output': '#A29BFE',       # Purple
                    'default': '#95A5A6'       # Gray
                }
                node_colors.append(color_map.get(node_type, color_map['default']))
                
                # Size based on connections
                degree = self.graph.degree(node_id)
                node_sizes.append(100 + degree * 50)
                
                # Labels
                if show_labels:
                    label = node_id
                    if show_attributes and node_attrs:
                        attr_parts = []
                        for key in ['status', 'priority', 'health', 'load']:
                            if key in node_attrs:
                                attr_parts.append(f"{key}:{node_attrs[key]}")
                        if attr_parts:
                            label += f"\n({', '.join(attr_parts)})"
                    labels[node_id] = label
        
        # Draw nodes
        if node_positions:
            xs, ys, zs = zip(*node_positions.values())
            scatter = ax.scatter(xs, ys, zs, c=node_colors, s=node_sizes, alpha=0.8)
            
            # Draw edges
            for source, target in self.graph.edges():
                if source in node_positions and target in node_positions:
                    source_pos = node_positions[source]
                    target_pos = node_positions[target]
                    
                    ax.plot([source_pos[0], target_pos[0]], 
                           [source_pos[1], target_pos[1]], 
                           [source_pos[2], target_pos[2]], 
                           'gray', alpha=0.6, linewidth=1)
                    
                    # Add arrow (simplified)
                    ax.quiver(source_pos[0], source_pos[1], source_pos[2],
                             target_pos[0] - source_pos[0],
                             target_pos[1] - source_pos[1], 
                             target_pos[2] - source_pos[2],
                             color='gray', alpha=0.4, arrow_length_ratio=0.1)
            
            # Draw labels
            if show_labels:
                for node_id, pos in node_positions.items():
                    if node_id in labels:
                        ax.text(pos[0], pos[1], pos[2], labels[node_id], fontsize=8)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"3D Knowledge Graph: {self.graph.graph_id} ({self.graph.graph_type.value})", 
                    fontsize=14, fontweight='bold')
        
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
    
    def create_isometric_plot(self, figsize: Tuple[int, int] = (14, 10), show_labels: bool = True, 
                             show_attributes: bool = False, positioning_strategy: str = "layered",
                             save_path: str = None) -> None:
        """
        Create an isometric (2.5D) visualization of the 3D graph
        
        Args:
            figsize: Figure size (width, height)
            show_labels: Whether to show node labels
            show_attributes: Whether to show node attributes in labels
            positioning_strategy: 3D positioning strategy
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("❌ matplotlib not available. Install with: pip install matplotlib")
            return
        
        # Ensure 3D positions are assigned
        if not self.graph.node_3d_positions:
            self.graph.auto_assign_3d_positions(positioning_strategy)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Isometric projection matrix
        # Standard isometric angles: 30° rotation around Y, then 35.264° around X
        def isometric_projection(x, y, z):
            """Convert 3D coordinates to 2D isometric projection"""
            # Isometric projection matrix
            iso_x = (x - z) * np.cos(np.pi/6)  # cos(30°)
            iso_y = y + (x + z) * np.sin(np.pi/6)  # sin(30°)
            return iso_x, iso_y
        
        # Prepare node data
        node_positions_2d = {}
        node_colors = []
        node_sizes = []
        labels = {}
        z_values = []  # For depth sorting
        
        for node_id in self.graph.nodes():
            pos_3d = self.graph.get_node_3d_position(node_id)
            if pos_3d:
                x, y, z = pos_3d
                iso_x, iso_y = isometric_projection(x, y, z)
                node_positions_2d[node_id] = (iso_x, iso_y)
                z_values.append((z, node_id))  # For depth sorting
                
                # Color and size based on node attributes
                node_attrs = self.graph.get_node_attributes(node_id)
                node_type = node_attrs.get('type', 'default')
                
                color_map = {
                    'mission': '#FF6B6B',      # Red
                    'coordination': '#4ECDC4',  # Teal
                    'task': '#45B7D1',         # Blue
                    'status': '#96CEB4',       # Green
                    'requirement': '#FECA57',   # Yellow
                    'resource': '#E17055',     # Orange
                    'output': '#A29BFE',       # Purple
                    'default': '#95A5A6'       # Gray
                }
                node_colors.append(color_map.get(node_type, color_map['default']))
                
                # Size based on connections and Z depth
                degree = self.graph.degree(node_id)
                base_size = 200 + degree * 100
                # Larger nodes appear closer (higher Z)
                depth_factor = 1.0 + (z / 5.0)  # Adjust scaling as needed
                node_sizes.append(base_size * depth_factor)
                
                # Labels
                if show_labels:
                    label = node_id
                    if show_attributes and node_attrs:
                        attr_parts = []
                        for key in ['status', 'priority', 'health', 'load']:
                            if key in node_attrs:
                                attr_parts.append(f"{key}:{node_attrs[key]}")
                        if attr_parts:
                            label += f"\n({', '.join(attr_parts)})"
                    labels[node_id] = label
        
        # Sort nodes by Z depth (back to front for proper rendering)
        z_values.sort(reverse=True)  # Furthest first
        
        # Draw edges first (behind nodes)
        for source, target in self.graph.edges():
            if source in node_positions_2d and target in node_positions_2d:
                source_pos = node_positions_2d[source]
                target_pos = node_positions_2d[target]
                
                # Get Z positions for depth-based styling
                source_z = self.graph.get_node_3d_position(source)[2]
                target_z = self.graph.get_node_3d_position(target)[2]
                avg_z = (source_z + target_z) / 2
                
                # Fade edges based on depth
                alpha = 0.3 + (avg_z / 5.0) * 0.4  # Closer edges more opaque
                alpha = max(0.1, min(0.7, alpha))
                
                ax.plot([source_pos[0], target_pos[0]], 
                       [source_pos[1], target_pos[1]], 
                       'gray', alpha=alpha, linewidth=1.5)
                
                # Add arrow
                dx = target_pos[0] - source_pos[0]
                dy = target_pos[1] - source_pos[1]
                ax.annotate('', xy=target_pos, xytext=source_pos,
                           arrowprops=dict(arrowstyle='->', color='gray', 
                                         alpha=alpha, lw=1))
        
        # Draw nodes (sorted by depth)
        for z, node_id in z_values:
            if node_id in node_positions_2d:
                pos = node_positions_2d[node_id]
                node_idx = list(self.graph.nodes()).index(node_id)
                
                # Adjust alpha based on depth
                alpha = 0.6 + (z / 5.0) * 0.3  # Closer nodes more opaque
                alpha = max(0.4, min(0.9, alpha))
                
                ax.scatter(pos[0], pos[1], 
                          c=[node_colors[node_idx]], 
                          s=[node_sizes[node_idx]], 
                          alpha=alpha, 
                          edgecolors='black', 
                          linewidth=0.5)
                
                # Draw labels
                if show_labels and node_id in labels:
                    ax.text(pos[0], pos[1], labels[node_id], 
                           fontsize=8, ha='center', va='bottom',
                           bbox=dict(boxstyle="round,pad=0.2", 
                                   facecolor='white', alpha=0.7))
        
        # Add depth grid lines for isometric effect
        self._add_isometric_grid(ax, node_positions_2d)
        
        # Set title and formatting
        ax.set_title(f"Isometric Knowledge Graph: {self.graph.graph_id} ({self.graph.graph_type.value})", 
                    fontsize=14, fontweight='bold')
        
        # Add stats
        stats = self.graph.get_stats()
        stats_text = f"Nodes: {stats['node_count']}, Edges: {stats['edge_count']}, Strategy: {positioning_strategy}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
               facecolor='lightblue', alpha=0.7))
        
        ax.set_aspect('equal')
        ax.axis('off')  # Hide axes for cleaner isometric look
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Isometric graph saved to: {save_path}")
        
        plt.show()
    
    def _add_isometric_grid(self, ax, node_positions_2d):
        """Add subtle grid lines to enhance isometric effect"""
        if not node_positions_2d:
            return
        
        # Get bounds
        xs, ys = zip(*node_positions_2d.values())
        min_x, max_x = min(xs) - 1, max(xs) + 1
        min_y, max_y = min(ys) - 1, max(ys) + 1
        
        # Add subtle grid lines
        import numpy as np
        
        # Horizontal lines
        for y in np.arange(min_y, max_y, 0.5):
            ax.plot([min_x, max_x], [y, y], 'lightgray', alpha=0.2, linewidth=0.5)
        
        # Diagonal lines (isometric grid)
        for offset in np.arange(-10, 10, 0.5):
            # Left-leaning diagonals
            x_start = min_x
            y_start = min_y + offset
            x_end = max_x
            y_end = min_y + offset + (max_x - min_x) * 0.5
            ax.plot([x_start, x_end], [y_start, y_end], 'lightgray', alpha=0.1, linewidth=0.5)
            
            # Right-leaning diagonals
            x_start = min_x
            y_start = max_y - offset
            x_end = max_x
            y_end = max_y - offset - (max_x - min_x) * 0.5
            ax.plot([x_start, x_end], [y_start, y_end], 'lightgray', alpha=0.1, linewidth=0.5)

class GraphType(str, Enum):
    """Types of knowledge graphs"""
    MISSION = "mission"
    TACTICAL = "tactical"
    LOCAL_RAG = "local_rag"
    CLOUD_RAG = "cloud_rag"

class KnowledgeGraphRegistry:
    """
    Global registry for knowledge graphs to enable cross-graph operations
    Singleton pattern for shared access across all graphs
    """
    _instance = None
    _graphs: Dict[str, 'KnowledgeGraph'] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(self, graph: 'KnowledgeGraph') -> None:
        """Register a knowledge graph"""
        self._graphs[graph.graph_id] = graph
        logger.info(f"KnowledgeGraph registered: {graph.graph_id}")
    
    def unregister(self, graph_id: str) -> None:
        """Unregister a knowledge graph"""
        if graph_id in self._graphs:
            del self._graphs[graph_id]
            logger.info(f"KnowledgeGraph unregistered: {graph_id}")
    
    def get(self, graph_id: str) -> Optional['KnowledgeGraph']:
        """Get a knowledge graph by ID"""
        return self._graphs.get(graph_id)
    
    def list_graphs(self) -> List[str]:
        """List all registered graph IDs"""
        return list(self._graphs.keys())
    
    def query_all(self, **filters) -> Dict[str, List[str]]:
        """Query nodes across all registered graphs"""
        results = {}
        for graph_id, graph in self._graphs.items():
            matching_nodes = graph.query_nodes(**filters)
            if matching_nodes:
                results[graph_id] = matching_nodes
        return results

class KnowledgeGraph:
    """
    Base class for all knowledge graphs using NetworkX
    Provides common functionality through composition and cross-graph operations
    """
    
    def __init__(self, graph_type: GraphType, graph_id: str = None, directed: bool = True, auto_register: bool = True):
        """
        Initialize knowledge graph
        
        Args:
            graph_type: Type of knowledge graph
            graph_id: Unique identifier for this graph
            directed: Whether to use directed or undirected graph
            auto_register: Whether to automatically register with global registry
        """
        self.graph_type = graph_type
        self.graph_id = graph_id or f"{graph_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create NetworkX graph via composition
        self.graph = nx.DiGraph() if directed else nx.Graph()
        
        # Cross-graph connections (external references)
        self.external_references: Dict[str, Dict[str, Any]] = {}  # {node_id: {target_graph: graph_id, target_node: node_id, relationship: str}}
        
        # 3D positioning for nodes
        self.node_3d_positions: Dict[str, Tuple[float, float, float]] = {}  # {node_id: (x, y, z)}
        
        # Registry for cross-graph operations
        self.registry = KnowledgeGraphRegistry()
        if auto_register:
            self.registry.register(self)
        
        # Visualization component (composition)
        self.visualizer = KnowledgeGraphVisualizer(self)
        
        # Metadata
        self.metadata = {
            "graph_type": graph_type.value,
            "graph_id": self.graph_id,
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "version": "1.0",
            "description": "",
            "tags": [],
            "source": "",
            "confidence": 1.0
        }
        
        logger.info(f"KnowledgeGraph created: {self.graph_id} ({graph_type.value})")
    
    # ========================================
    # Core NetworkX Delegation Methods
    # ========================================
    
    def add_node(self, node_id: str, **attributes) -> None:
        """Add a node to the graph with attributes"""
        self.graph.add_node(node_id, **attributes)
        self._update_modified_time()
    
    def add_edge(self, source: str, target: str, **attributes) -> None:
        """Add an edge to the graph with attributes"""
        self.graph.add_edge(source, target, **attributes)
        self._update_modified_time()
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the graph"""
        if self.has_node(node_id):
            self.graph.remove_node(node_id)
            self._update_modified_time()
    
    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge from the graph"""
        if self.has_edge(source, target):
            self.graph.remove_edge(source, target)
            self._update_modified_time()
    
    def has_node(self, node_id: str) -> bool:
        """Check if node exists in graph"""
        return self.graph.has_node(node_id)
    
    def has_edge(self, source: str, target: str) -> bool:
        """Check if edge exists in graph"""
        return self.graph.has_edge(source, target)
    
    def get_node_attributes(self, node_id: str) -> Dict[str, Any]:
        """Get all attributes for a node"""
        return dict(self.graph.nodes[node_id]) if self.has_node(node_id) else {}
    
    def get_edge_attributes(self, source: str, target: str) -> Dict[str, Any]:
        """Get all attributes for an edge"""
        return dict(self.graph.edges[source, target]) if self.has_edge(source, target) else {}
    
    def set_node_attribute(self, node_id: str, key: str, value: Any) -> None:
        """Set a single node attribute"""
        if self.has_node(node_id):
            self.graph.nodes[node_id][key] = value
            self._update_modified_time()
    
    def set_edge_attribute(self, source: str, target: str, key: str, value: Any) -> None:
        """Set a single edge attribute"""
        if self.has_edge(source, target):
            self.graph.edges[source, target][key] = value
            self._update_modified_time()
    
    # ========================================
    # Graph Analysis Methods
    # ========================================
    
    def nodes(self) -> List[str]:
        """Get all node IDs"""
        return list(self.graph.nodes())
    
    def edges(self) -> List[Tuple[str, str]]:
        """Get all edges as (source, target) tuples"""
        return list(self.graph.edges())
    
    def neighbors(self, node_id: str) -> List[str]:
        """Get neighbors of a node"""
        return list(self.graph.neighbors(node_id)) if self.has_node(node_id) else []
    
    def predecessors(self, node_id: str) -> List[str]:
        """Get predecessors of a node (for directed graphs)"""
        if isinstance(self.graph, nx.DiGraph) and self.has_node(node_id):
            return list(self.graph.predecessors(node_id))
        return []
    
    def successors(self, node_id: str) -> List[str]:
        """Get successors of a node (for directed graphs)"""
        if isinstance(self.graph, nx.DiGraph) and self.has_node(node_id):
            return list(self.graph.successors(node_id))
        return []
    
    def degree(self, node_id: str) -> int:
        """Get degree of a node"""
        return self.graph.degree(node_id) if self.has_node(node_id) else 0
    
    def in_degree(self, node_id: str) -> int:
        """Get in-degree of a node (for directed graphs)"""
        if isinstance(self.graph, nx.DiGraph) and self.has_node(node_id):
            return self.graph.in_degree(node_id)
        return 0
    
    def out_degree(self, node_id: str) -> int:
        """Get out-degree of a node (for directed graphs)"""
        if isinstance(self.graph, nx.DiGraph) and self.has_node(node_id):
            return self.graph.out_degree(node_id)
        return 0
    
    # ========================================
    # Cross-Graph Connection Methods
    # ========================================
    
    def add_external_reference(self, local_node: str, target_graph_id: str, target_node: str, relationship: str = "references", **attributes) -> None:
        """
        Create a reference from a local node to a node in another graph
        
        Args:
            local_node: Node ID in this graph
            target_graph_id: ID of the target graph
            target_node: Node ID in the target graph
            relationship: Type of relationship
            **attributes: Additional attributes for the reference
        """
        if not self.has_node(local_node):
            raise ValueError(f"Local node '{local_node}' does not exist")
        
        # Check if target graph and node exist
        target_graph = self.registry.get(target_graph_id)
        if not target_graph:
            raise ValueError(f"Target graph '{target_graph_id}' not found in registry")
        
        if not target_graph.has_node(target_node):
            raise ValueError(f"Target node '{target_node}' does not exist in graph '{target_graph_id}'")
        
        # Store the external reference
        if local_node not in self.external_references:
            self.external_references[local_node] = {}
        
        ref_key = f"{target_graph_id}:{target_node}"
        self.external_references[local_node][ref_key] = {
            "target_graph": target_graph_id,
            "target_node": target_node,
            "relationship": relationship,
            "created_at": datetime.now().isoformat(),
            **attributes
        }
        
        self._update_modified_time()
        logger.info(f"External reference added: {self.graph_id}:{local_node} -> {target_graph_id}:{target_node}")
    
    def remove_external_reference(self, local_node: str, target_graph_id: str, target_node: str) -> None:
        """Remove an external reference"""
        if local_node in self.external_references:
            ref_key = f"{target_graph_id}:{target_node}"
            if ref_key in self.external_references[local_node]:
                del self.external_references[local_node][ref_key]
                if not self.external_references[local_node]:
                    del self.external_references[local_node]
                self._update_modified_time()
    
    def get_external_references(self, local_node: str) -> Dict[str, Dict[str, Any]]:
        """Get all external references for a local node"""
        return self.external_references.get(local_node, {})
    
    def resolve_external_reference(self, local_node: str, target_graph_id: str, target_node: str) -> Optional[Dict[str, Any]]:
        """
        Resolve an external reference and get the target node's data
        
        Returns:
            Dictionary containing target node attributes and metadata
        """
        target_graph = self.registry.get(target_graph_id)
        if not target_graph or not target_graph.has_node(target_node):
            return None
        
        return {
            "graph_id": target_graph_id,
            "graph_type": target_graph.graph_type.value,
            "node_id": target_node,
            "attributes": target_graph.get_node_attributes(target_node),
            "metadata": target_graph.metadata
        }
    
    def hive_mind_query(self, **filters) -> Dict[str, List[Dict[str, Any]]]:
        """
        Query across this graph and all referenced graphs (hive mind)
        
        Returns:
            Dictionary mapping graph_id to list of matching nodes with full data
        """
        results = {}
        
        # Query local graph
        local_matches = self.query_nodes(**filters)
        if local_matches:
            results[self.graph_id] = [
                {
                    "node_id": node_id,
                    "attributes": self.get_node_attributes(node_id),
                    "graph_type": self.graph_type.value
                }
                for node_id in local_matches
            ]
        
        # Query all referenced graphs
        referenced_graphs = set()
        for node_refs in self.external_references.values():
            for ref_data in node_refs.values():
                referenced_graphs.add(ref_data["target_graph"])
        
        for graph_id in referenced_graphs:
            target_graph = self.registry.get(graph_id)
            if target_graph:
                matches = target_graph.query_nodes(**filters)
                if matches:
                    results[graph_id] = [
                        {
                            "node_id": node_id,
                            "attributes": target_graph.get_node_attributes(node_id),
                            "graph_type": target_graph.graph_type.value
                        }
                        for node_id in matches
                    ]
        
        return results
    
    def get_connected_knowledge_network(self) -> Dict[str, Any]:
        """
        Get the entire connected knowledge network from this graph's perspective
        
        Returns:
            Complete network view including all connected graphs
        """
        network = {
            "root_graph": {
                "graph_id": self.graph_id,
                "graph_type": self.graph_type.value,
                "node_count": len(self.nodes()),
                "edge_count": len(self.edges())
            },
            "connected_graphs": {},
            "cross_graph_connections": []
        }
        
        # Get all connected graphs
        connected_graph_ids = set()
        for node_refs in self.external_references.values():
            for ref_data in node_refs.values():
                connected_graph_ids.add(ref_data["target_graph"])
        
        # Add connected graph info
        for graph_id in connected_graph_ids:
            target_graph = self.registry.get(graph_id)
            if target_graph:
                network["connected_graphs"][graph_id] = {
                    "graph_type": target_graph.graph_type.value,
                    "node_count": len(target_graph.nodes()),
                    "edge_count": len(target_graph.edges()),
                    "metadata": target_graph.metadata
                }
        
        # Add cross-graph connections
        for local_node, node_refs in self.external_references.items():
            for ref_key, ref_data in node_refs.items():
                network["cross_graph_connections"].append({
                    "source_graph": self.graph_id,
                    "source_node": local_node,
                    "target_graph": ref_data["target_graph"],
                    "target_node": ref_data["target_node"],
                    "relationship": ref_data["relationship"]
                })
        
        return network
    
    # ========================================
    # Query Methods
    # ========================================
    
    def query_nodes(self, **filters) -> List[str]:
        """
        Query nodes based on attribute filters
        
        Args:
            **filters: Attribute key-value pairs to filter by
            
        Returns:
            List of node IDs matching the filters
        """
        matching_nodes = []
        
        for node_id in self.graph.nodes():
            node_attrs = self.graph.nodes[node_id]
            
            # Check if all filters match
            match = True
            for key, value in filters.items():
                if key not in node_attrs or node_attrs[key] != value:
                    match = False
                    break
            
            if match:
                matching_nodes.append(node_id)
        
        return matching_nodes
    
    def query_edges(self, **filters) -> List[Tuple[str, str]]:
        """
        Query edges based on attribute filters
        
        Args:
            **filters: Attribute key-value pairs to filter by
            
        Returns:
            List of (source, target) tuples matching the filters
        """
        matching_edges = []
        
        for source, target in self.graph.edges():
            edge_attrs = self.graph.edges[source, target]
            
            # Check if all filters match
            match = True
            for key, value in filters.items():
                if key not in edge_attrs or edge_attrs[key] != value:
                    match = False
                    break
            
            if match:
                matching_edges.append((source, target))
        
        return matching_edges
    
    def find_paths(self, source: str, target: str, max_length: int = None) -> List[List[str]]:
        """Find all simple paths between two nodes"""
        if not (self.has_node(source) and self.has_node(target)):
            return []
        
        try:
            if max_length:
                return list(nx.all_simple_paths(self.graph, source, target, cutoff=max_length))
            else:
                return list(nx.all_simple_paths(self.graph, source, target))
        except nx.NetworkXNoPath:
            return []
    
    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two nodes"""
        if not (self.has_node(source) and self.has_node(target)):
            return None
        
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None
    
    def connected_components(self) -> List[Set[str]]:
        """Get connected components"""
        if isinstance(self.graph, nx.DiGraph):
            return [set(component) for component in nx.weakly_connected_components(self.graph)]
        else:
            return [set(component) for component in nx.connected_components(self.graph)]
    
    def subgraph(self, nodes: List[str]) -> 'KnowledgeGraph':
        """Create a subgraph with specified nodes"""
        # Filter nodes that actually exist
        valid_nodes = [node for node in nodes if self.has_node(node)]
        
        if not valid_nodes:
            # Return empty graph of same type
            return KnowledgeGraph(self.graph_type, f"{self.graph_id}_subgraph", isinstance(self.graph, nx.DiGraph))
        
        # Create subgraph
        sub_nx_graph = self.graph.subgraph(valid_nodes).copy()
        
        # Create new KnowledgeGraph instance
        subgraph = KnowledgeGraph(self.graph_type, f"{self.graph_id}_subgraph", isinstance(self.graph, nx.DiGraph))
        subgraph.graph = sub_nx_graph
        
        # Copy relevant metadata
        subgraph.metadata.update({
            "parent_graph": self.graph_id,
            "subgraph_nodes": valid_nodes,
            "created_from": "subgraph_operation"
        })
        
        return subgraph
    
    # ========================================
    # Serialization Methods
    # ========================================
    
    def to_json(self, include_metadata: bool = True) -> str:
        """
        Convert graph to JSON string
        
        Args:
            include_metadata: Whether to include graph metadata
            
        Returns:
            JSON string representation of the graph
        """
        graph_data = {
            "nodes": [
                {
                    "id": node_id,
                    "attributes": dict(self.graph.nodes[node_id])
                }
                for node_id in self.graph.nodes()
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "attributes": dict(self.graph.edges[source, target])
                }
                for source, target in self.graph.edges()
            ]
        }
        
        if include_metadata:
            graph_data["metadata"] = self.metadata.copy()
        
        return json.dumps(graph_data, indent=2, default=str)
    
    def from_json(self, json_str: str) -> None:
        """
        Load graph from JSON string
        
        Args:
            json_str: JSON string representation of the graph
        """
        try:
            data = json.loads(json_str)
            
            # Clear existing graph
            self.graph.clear()
            
            # Add nodes
            for node_data in data.get("nodes", []):
                node_id = node_data["id"]
                attributes = node_data.get("attributes", {})
                self.graph.add_node(node_id, **attributes)
            
            # Add edges
            for edge_data in data.get("edges", []):
                source = edge_data["source"]
                target = edge_data["target"]
                attributes = edge_data.get("attributes", {})
                self.graph.add_edge(source, target, **attributes)
            
            # Update metadata if present
            if "metadata" in data:
                self.metadata.update(data["metadata"])
            
            self._update_modified_time()
            logger.info(f"KnowledgeGraph loaded from JSON: {self.graph_id}")
            
        except Exception as e:
            logger.error(f"Failed to load graph from JSON: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary"""
        return json.loads(self.to_json())
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load graph from dictionary"""
        self.from_json(json.dumps(data))
    
    # ========================================
    # Graph Statistics
    # ========================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        stats = {
            "graph_id": self.graph_id,
            "graph_type": self.graph_type.value,
            "node_count": len(self.graph.nodes()),
            "edge_count": len(self.graph.edges()),
            "is_directed": isinstance(self.graph, nx.DiGraph),
            "is_connected": nx.is_connected(self.graph) if not isinstance(self.graph, nx.DiGraph) else nx.is_weakly_connected(self.graph),
            "density": nx.density(self.graph),
            "created_at": self.metadata["created_at"],
            "last_modified": self.metadata["last_modified"]
        }
        
        # Add directed graph specific stats
        if isinstance(self.graph, nx.DiGraph):
            stats.update({
                "is_dag": nx.is_directed_acyclic_graph(self.graph),
                "strongly_connected_components": len(list(nx.strongly_connected_components(self.graph))),
                "weakly_connected_components": len(list(nx.weakly_connected_components(self.graph)))
            })
        else:
            stats.update({
                "connected_components": len(list(nx.connected_components(self.graph)))
            })
        
        return stats
    
    # ========================================
    # Utility Methods
    # ========================================
    
    def _update_modified_time(self) -> None:
        """Update the last modified timestamp"""
        self.metadata["last_modified"] = datetime.now().isoformat()
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value"""
        self.metadata[key] = value
        self._update_modified_time()
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value"""
        return self.metadata.get(key, default)
    
    def clear(self) -> None:
        """Clear all nodes and edges from the graph"""
        self.graph.clear()
        self._update_modified_time()
    
    def copy(self) -> 'KnowledgeGraph':
        """Create a deep copy of the knowledge graph"""
        new_graph = KnowledgeGraph(
            self.graph_type, 
            f"{self.graph_id}_copy", 
            isinstance(self.graph, nx.DiGraph)
        )
        new_graph.graph = self.graph.copy()
        new_graph.metadata = self.metadata.copy()
        return new_graph
    
    def merge(self, other: 'KnowledgeGraph') -> 'KnowledgeGraph':
        """
        Merge another knowledge graph into a new graph
        
        Args:
            other: Another KnowledgeGraph to merge
            
        Returns:
            New KnowledgeGraph containing nodes and edges from both graphs
        """
        merged = self.copy()
        merged.graph_id = f"{self.graph_id}_merged_{other.graph_id}"
        
        # Add nodes from other graph
        for node_id in other.nodes():
            if not merged.has_node(node_id):
                merged.add_node(node_id, **other.get_node_attributes(node_id))
        
        # Add edges from other graph
        for source, target in other.edges():
            if not merged.has_edge(source, target):
                merged.add_edge(source, target, **other.get_edge_attributes(source, target))
        
        # Merge metadata
        merged.set_metadata("merged_from", [self.graph_id, other.graph_id])
        
        return merged
    
    def __str__(self) -> str:
        """String representation of the graph"""
        return f"KnowledgeGraph(id={self.graph_id}, type={self.graph_type.value}, nodes={len(self.graph.nodes())}, edges={len(self.graph.edges())})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__()
    
    # ========================================
    # 3D Positioning Methods
    # ========================================
    
    def set_node_3d_position(self, node_id: str, x: float, y: float, z: float) -> None:
        """Set 3D position for a node"""
        if self.has_node(node_id):
            self.node_3d_positions[node_id] = (x, y, z)
            self._update_modified_time()
    
    def get_node_3d_position(self, node_id: str) -> Optional[Tuple[float, float, float]]:
        """Get 3D position for a node"""
        return self.node_3d_positions.get(node_id)
    
    def auto_assign_3d_positions(self, strategy: str = "layered") -> None:
        """
        Automatically assign 3D positions to nodes based on strategy
        
        Args:
            strategy: "layered", "hierarchical", "circular", or "random"
        """
        if strategy == "layered":
            self._assign_layered_3d_positions()
        elif strategy == "hierarchical":
            self._assign_hierarchical_3d_positions()
        elif strategy == "circular":
            self._assign_circular_3d_positions()
        elif strategy == "random":
            self._assign_random_3d_positions()
        else:
            self._assign_layered_3d_positions()  # Default
    
    def _assign_layered_3d_positions(self) -> None:
        """Assign 3D positions in layers based on node type"""
        import random
        
        # Group nodes by type
        type_layers = {}
        for node_id in self.nodes():
            node_type = self.get_node_attributes(node_id).get('type', 'default')
            if node_type not in type_layers:
                type_layers[node_type] = []
            type_layers[node_type].append(node_id)
        
        # Assign Z levels to types
        type_z_levels = {
            'mission': 3.0,
            'coordination': 2.5,
            'requirement': 2.0,
            'task': 1.5,
            'resource': 1.0,
            'status': 0.5,
            'output': 0.0,
            'default': 1.0
        }
        
        # Position nodes in each layer
        for node_type, nodes in type_layers.items():
            z_level = type_z_levels.get(node_type, 1.0)
            
            # Arrange nodes in a circle at this Z level
            import math
            num_nodes = len(nodes)
            for i, node_id in enumerate(nodes):
                angle = 2 * math.pi * i / num_nodes if num_nodes > 1 else 0
                radius = 2.0 + random.uniform(-0.5, 0.5)  # Add some randomness
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                self.set_node_3d_position(node_id, x, y, z_level)
    
    def _assign_hierarchical_3d_positions(self) -> None:
        """Assign 3D positions based on graph hierarchy"""
        if not isinstance(self.graph, nx.DiGraph):
            self._assign_layered_3d_positions()
            return
        
        try:
            # Calculate levels using topological sort
            levels = {}
            for node in nx.topological_sort(self.graph):
                predecessors = list(self.graph.predecessors(node))
                if not predecessors:
                    levels[node] = 0
                else:
                    levels[node] = max(levels[pred] for pred in predecessors) + 1
            
            # Position nodes by level
            import math
            level_groups = {}
            for node, level in levels.items():
                if level not in level_groups:
                    level_groups[level] = []
                level_groups[level].append(node)
            
            for level, nodes in level_groups.items():
                z_level = level * 1.0  # 1 unit per level
                num_nodes = len(nodes)
                
                for i, node_id in enumerate(nodes):
                    angle = 2 * math.pi * i / num_nodes if num_nodes > 1 else 0
                    radius = 1.5
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    self.set_node_3d_position(node_id, x, y, z_level)
                    
        except nx.NetworkXError:
            # Graph has cycles, fall back to layered
            self._assign_layered_3d_positions()
    
    def _assign_circular_3d_positions(self) -> None:
        """Assign 3D positions in a spiral"""
        import math
        
        nodes = list(self.nodes())
        num_nodes = len(nodes)
        
        for i, node_id in enumerate(nodes):
            # Create a spiral
            t = i / num_nodes * 4 * math.pi  # 2 full rotations
            radius = 1.0 + t / (4 * math.pi)  # Expanding radius
            x = radius * math.cos(t)
            y = radius * math.sin(t)
            z = i * 0.3  # Rising spiral
            self.set_node_3d_position(node_id, x, y, z)
    
    def _assign_random_3d_positions(self) -> None:
        """Assign random 3D positions"""
        import random
        
        for node_id in self.nodes():
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            z = random.uniform(0, 3)
            self.set_node_3d_position(node_id, x, y, z)
    
    # ========================================
    # Visualization Methods (Delegation to Visualizer)
    # ========================================
    
    def visualize(self, show_attributes: bool = False, show_external_refs: bool = True) -> str:
        """
        Create a quick ASCII visualization of the graph
        Delegates to KnowledgeGraphVisualizer for separation of concerns
        """
        return self.visualizer.create_ascii_visualization(show_attributes, show_external_refs)
    
    def print_viz(self, show_attributes: bool = False, show_external_refs: bool = True) -> None:
        """Print the graph visualization to console"""
        print(self.visualize(show_attributes, show_external_refs))
    
    def plot(self, figsize: Tuple[int, int] = (12, 8), show_labels: bool = True, 
             show_attributes: bool = False, show_external_refs: bool = True,
             layout: str = "spring", save_path: str = None) -> None:
        """
        Create a matplotlib visualization of the graph
        Delegates to KnowledgeGraphVisualizer for separation of concerns
        """
        self.visualizer.create_matplotlib_plot(figsize, show_labels, show_attributes, 
                                              show_external_refs, layout, save_path)
    
    def plot_network(self, figsize: Tuple[int, int] = (20, 12), layout: str = "spring", 
                    save_path: str = None) -> None:
        """
        Plot the entire connected knowledge network with cross-graph interconnects
        Delegates to KnowledgeGraphVisualizer for separation of concerns
        """
        self.visualizer.create_network_plot(figsize, layout, save_path)
    
    def plot_3d(self, figsize: Tuple[int, int] = (14, 10), show_labels: bool = True, 
               show_attributes: bool = False, positioning_strategy: str = "layered",
               save_path: str = None) -> None:
        """
        Create a 3D matplotlib visualization of the graph
        Delegates to KnowledgeGraphVisualizer for separation of concerns
        """
        self.visualizer.create_3d_plot(figsize, show_labels, show_attributes, 
                                      positioning_strategy, save_path)
    
    def plot_isometric(self, figsize: Tuple[int, int] = (14, 10), show_labels: bool = True, 
                      show_attributes: bool = False, positioning_strategy: str = "layered",
                      save_path: str = None) -> None:
        """
        Create an isometric (2.5D) visualization of the 3D graph
        Delegates to KnowledgeGraphVisualizer for separation of concerns
        """
        self.visualizer.create_isometric_plot(figsize, show_labels, show_attributes, 
                                             positioning_strategy, save_path)
    

if __name__ == "__main__":
    # Import and run demo from separate file
    from knowledge_graph_demo import demo_knowledge_graph
    demo_knowledge_graph()
