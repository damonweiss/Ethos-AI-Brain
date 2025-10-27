"""
ASCII Knowledge Graph Visualizer - Text-based visualization
Creates clean ASCII representations without emojis
"""

import logging
import networkx as nx
from typing import Dict, Any, List, Optional, Union, Tuple, Set

from .knowledge_graph_visualizer_base import KnowledgeGraphVisualizer

logger = logging.getLogger(__name__)


class ASCIIKnowledgeGraphVisualizer(KnowledgeGraphVisualizer):
    """
    ASCII-based knowledge graph visualizer
    Creates text-based representations using ASCII symbols and styling
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
        lines.append(f"[GRAPH] {self.graph.graph_id} ({self.graph.graph_type})")
        lines.append("=" * (len(self.graph.graph_id) + len(self.graph.graph_type) + 9))

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
                successors = list(self.graph.successors(node_id) if isinstance(self.graph.graph,
                                                                               nx.DiGraph) else self.graph.neighbors(
                    node_id))

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
            lines.append("[REFS] External References:")
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

        lines.append("[NETWORK] CONNECTED KNOWLEDGE NETWORK")
        lines.append("=" * 40)

        # Show root graph
        lines.append(f"ROOT: {self.graph.graph_id}")
        lines.append(self.render_graph(**kwargs))

        # Show connected graphs
        from ..knowledge_graph import KnowledgeGraphRegistry
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
