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

from ..style.knowledge_graph_style import KnowledgeGraphStyle, StylePresets, ColorScheme

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
        from ..style.knowledge_graph_style import StylePresets
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


