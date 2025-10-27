#!/usr/bin/env python3
"""
Knowledge Graph Style System - Centralized styling for all visualization types
Provides consistent colors, sizes, and visual properties across different visualizers
"""

from typing import Dict, Any, Tuple, List
from enum import Enum

class NodeShape(str, Enum):
    """Node shape options"""
    CIRCLE = "circle"
    SQUARE = "square"
    DIAMOND = "diamond"
    TRIANGLE = "triangle"
    HEXAGON = "hexagon"

class EdgeStyle(str, Enum):
    """Edge style options"""
    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"
    DASH_DOT = "dashdot"

class ColorScheme(str, Enum):
    """Available color schemes"""
    DEFAULT = "default"
    DARK = "dark"
    PASTEL = "pastel"
    HIGH_CONTRAST = "high_contrast"
    MONOCHROME = "monochrome"

class KnowledgeGraphStyle:
    """
    Centralized style system for knowledge graph visualizations
    Provides consistent styling across ASCII, matplotlib, plotly, and future visualizers
    """
    
    def __init__(self, color_scheme: ColorScheme = ColorScheme.DEFAULT):
        """
        Initialize style system with a color scheme
        
        Args:
            color_scheme: The color scheme to use
        """
        self.color_scheme = color_scheme
        self._initialize_color_schemes()
        self._initialize_node_styles()
        self._initialize_edge_styles()
        self._initialize_layout_styles()
    
    def _initialize_color_schemes(self):
        """Initialize all available color schemes"""
        self.color_schemes = {
            ColorScheme.DEFAULT: {
                'mission': '#FF6B6B',      # Red
                'coordination': '#4ECDC4',  # Teal
                'task': '#45B7D1',         # Blue
                'status': '#96CEB4',       # Green
                'requirement': '#FECA57',   # Yellow
                'resource': '#E17055',     # Orange
                'output': '#A29BFE',       # Purple
                'default': '#95A5A6',      # Gray
                'background': '#FFFFFF',   # White
                'text': '#2C3E50',         # Dark blue-gray
                'grid': '#ECF0F1',         # Light gray
                'external_ref': '#E74C3C'  # Red for cross-graph references
            },
            ColorScheme.DARK: {
                'mission': '#FF5555',      # Bright red
                'coordination': '#50FA7B',  # Bright green
                'task': '#8BE9FD',         # Cyan
                'status': '#FFB86C',       # Orange
                'requirement': '#F1FA8C',   # Yellow
                'resource': '#FF79C6',     # Pink
                'output': '#BD93F9',       # Purple
                'default': '#6272A4',      # Blue-gray
                'background': '#282A36',   # Dark gray
                'text': '#F8F8F2',         # Light gray
                'grid': '#44475A',         # Medium gray
                'external_ref': '#FF5555'  # Bright red
            },
            ColorScheme.PASTEL: {
                'mission': '#FFB3BA',      # Light red
                'coordination': '#BAFFC9',  # Light green
                'task': '#BAE1FF',         # Light blue
                'status': '#FFFFBA',       # Light yellow
                'requirement': '#FFDFBA',   # Light orange
                'resource': '#E0BBE4',     # Light purple
                'output': '#C7CEEA',       # Light lavender
                'default': '#D3D3D3',      # Light gray
                'background': '#FAFAFA',   # Very light gray
                'text': '#333333',         # Dark gray
                'grid': '#F0F0F0',         # Light gray
                'external_ref': '#FFB3BA'  # Light red
            },
            ColorScheme.HIGH_CONTRAST: {
                'mission': '#FF0000',      # Pure red
                'coordination': '#00FF00',  # Pure green
                'task': '#0000FF',         # Pure blue
                'status': '#FFFF00',       # Pure yellow
                'requirement': '#FF8000',   # Orange
                'resource': '#8000FF',     # Purple
                'output': '#00FFFF',       # Cyan
                'default': '#808080',      # Gray
                'background': '#FFFFFF',   # White
                'text': '#000000',         # Black
                'grid': '#C0C0C0',         # Silver
                'external_ref': '#FF0000'  # Pure red
            },
            ColorScheme.MONOCHROME: {
                'mission': '#000000',      # Black
                'coordination': '#404040',  # Dark gray
                'task': '#606060',         # Medium gray
                'status': '#808080',       # Gray
                'requirement': '#A0A0A0',   # Light gray
                'resource': '#C0C0C0',     # Silver
                'output': '#E0E0E0',       # Very light gray
                'default': '#909090',      # Medium gray
                'background': '#FFFFFF',   # White
                'text': '#000000',         # Black
                'grid': '#D0D0D0',         # Light gray
                'external_ref': '#000000'  # Black
            }
        }
    
    def _initialize_node_styles(self):
        """Initialize node styling properties"""
        self.node_styles = {
            'base_size': 300,
            'size_multiplier': 100,  # Size increase per connection
            'min_size': 50,
            'max_size': 1000,
            'alpha': 0.8,
            'border_width': 1,
            'border_color': '#000000',
            'shape_map': {
                'mission': NodeShape.DIAMOND,
                'coordination': NodeShape.HEXAGON,
                'task': NodeShape.CIRCLE,
                'status': NodeShape.SQUARE,
                'requirement': NodeShape.TRIANGLE,
                'resource': NodeShape.CIRCLE,
                'output': NodeShape.SQUARE,
                'default': NodeShape.CIRCLE
            }
        }
    
    def _initialize_edge_styles(self):
        """Initialize edge styling properties"""
        self.edge_styles = {
            'width': 1.5,
            'alpha': 0.6,
            'arrow_size': 20,
            'style': EdgeStyle.SOLID,
            'external_ref_style': EdgeStyle.DASHED,
            'external_ref_width': 2.0,
            'external_ref_alpha': 0.7
        }
    
    def _initialize_layout_styles(self):
        """Initialize layout and spacing properties"""
        self.layout_styles = {
            'node_spacing': 1.0,
            'layer_spacing': 1.0,
            'grid_alpha': 0.2,
            'grid_width': 0.5,
            'font_size': 8,
            'title_font_size': 14,
            'legend_font_size': 10,
            'margin': 0.1
        }
    
    def get_node_color(self, node_type: str) -> str:
        """Get color for a node type"""
        colors = self.color_schemes[self.color_scheme]
        return colors.get(node_type, colors['default'])
    
    def get_node_size(self, degree: int, base_size: int = None) -> int:
        """Calculate node size based on degree (number of connections)"""
        if base_size is None:
            base_size = self.node_styles['base_size']
        
        size = base_size + (degree * self.node_styles['size_multiplier'])
        return max(self.node_styles['min_size'], 
                  min(self.node_styles['max_size'], size))
    
    def get_node_shape(self, node_type: str) -> NodeShape:
        """Get shape for a node type"""
        return self.node_styles['shape_map'].get(node_type, NodeShape.CIRCLE)
    
    def get_edge_color(self, is_external_ref: bool = False) -> str:
        """Get color for edges"""
        colors = self.color_schemes[self.color_scheme]
        if is_external_ref:
            return colors['external_ref']
        return colors['text']
    
    def get_edge_style(self, is_external_ref: bool = False) -> Dict[str, Any]:
        """Get complete edge style properties"""
        style = {
            'color': self.get_edge_color(is_external_ref),
            'width': self.edge_styles['external_ref_width'] if is_external_ref else self.edge_styles['width'],
            'alpha': self.edge_styles['external_ref_alpha'] if is_external_ref else self.edge_styles['alpha'],
            'style': self.edge_styles['external_ref_style'] if is_external_ref else self.edge_styles['style'],
            'arrow_size': self.edge_styles['arrow_size']
        }
        return style
    
    def get_background_color(self) -> str:
        """Get background color"""
        return self.color_schemes[self.color_scheme]['background']
    
    def get_text_color(self) -> str:
        """Get text color"""
        return self.color_schemes[self.color_scheme]['text']
    
    def get_grid_color(self) -> str:
        """Get grid color"""
        return self.color_schemes[self.color_scheme]['grid']
    
    def get_font_sizes(self) -> Dict[str, int]:
        """Get font sizes for different text elements"""
        return self.layout_styles.get('font_sizes', {
            'node_label': self.layout_styles['font_size'],
            'edge_label': self.layout_styles['font_size'] - 2,
            'title': self.layout_styles['title_font_size'],
            'legend': self.layout_styles['legend_font_size']
        })
    
    def get_standard_node_style(self) -> Dict[str, Any]:
        """Get standard node styling parameters for consistent appearance"""
        return {
            's': 300,                    # Node size
            'alpha': 0.9,               # Transparency
            'edgecolors': 'lightgray',  # Outline color
            'linewidth': 0.5            # Outline width (hairline)
        }
    
    def get_3d_layer_positions(self) -> Dict[str, float]:
        """Get Z-layer positions for different node types in 3D visualization"""
        return {
            'mission': 3.0,
            'coordination': 2.5,
            'requirement': 2.0,
            'task': 1.5,
            'resource': 1.0,
            'status': 0.5,
            'output': 0.0,
            'default': 1.0
        }
    
    def get_ascii_symbols(self) -> Dict[str, str]:
        """Get ASCII symbols for different node types (for ASCII visualizer)"""
        return {
            'mission': '◆',      # Diamond
            'coordination': '⬢',  # Hexagon
            'task': '●',         # Circle
            'status': '■',       # Square
            'requirement': '▲',   # Triangle
            'resource': '○',     # Circle outline
            'output': '□',       # Square outline
            'default': '●',      # Circle
            'edge': '─',         # Horizontal line
            'arrow': '→',        # Arrow
            'external_ref': '⤏', # Curved arrow
            'vertical': '│',     # Vertical line
            'corner': '└',       # Corner
            'branch': '├'        # Branch
        }
    
    def create_legend_data(self) -> List[Dict[str, Any]]:
        """Create legend data for visualizations"""
        legend_data = []
        colors = self.color_schemes[self.color_scheme]
        
        # Add node type entries
        for node_type in ['mission', 'coordination', 'task', 'status', 'requirement', 'resource', 'output']:
            legend_data.append({
                'type': 'node',
                'label': node_type.title(),
                'color': colors[node_type],
                'shape': self.get_node_shape(node_type)
            })
        
        # Add edge type entries
        legend_data.append({
            'type': 'edge',
            'label': 'Internal Connection',
            'color': colors['text'],
            'style': EdgeStyle.SOLID
        })
        
        legend_data.append({
            'type': 'edge',
            'label': 'External Reference',
            'color': colors['external_ref'],
            'style': EdgeStyle.DASHED
        })
        
        return legend_data
    
    def set_color_scheme(self, color_scheme: ColorScheme):
        """Change the color scheme"""
        self.color_scheme = color_scheme
    
    def customize_node_style(self, **kwargs):
        """Customize node styling properties"""
        self.node_styles.update(kwargs)
    
    def customize_edge_style(self, **kwargs):
        """Customize edge styling properties"""
        self.edge_styles.update(kwargs)
    
    def customize_layout_style(self, **kwargs):
        """Customize layout styling properties"""
        self.layout_styles.update(kwargs)

# Predefined style presets
class StylePresets:
    """Predefined style configurations for common use cases"""
    
    @staticmethod
    def professional() -> KnowledgeGraphStyle:
        """Professional presentation style"""
        style = KnowledgeGraphStyle(ColorScheme.DEFAULT)
        style.customize_node_style(
            base_size=400,
            alpha=0.9,
            border_width=2
        )
        style.customize_edge_style(
            width=2.0,
            alpha=0.7
        )
        return style
    
    @staticmethod
    def dark_mode() -> KnowledgeGraphStyle:
        """Dark mode style"""
        return KnowledgeGraphStyle(ColorScheme.DARK)
    
    @staticmethod
    def high_contrast() -> KnowledgeGraphStyle:
        """High contrast for accessibility"""
        style = KnowledgeGraphStyle(ColorScheme.HIGH_CONTRAST)
        style.customize_edge_style(width=3.0)
        return style
    
    @staticmethod
    def minimal() -> KnowledgeGraphStyle:
        """Minimal, clean style"""
        style = KnowledgeGraphStyle(ColorScheme.MONOCHROME)
        style.customize_node_style(
            base_size=200,
            alpha=0.6
        )
        style.customize_edge_style(
            width=1.0,
            alpha=0.4
        )
        return style
    
    @staticmethod
    def presentation() -> KnowledgeGraphStyle:
        """Large, clear style for presentations"""
        style = KnowledgeGraphStyle(ColorScheme.PASTEL)
        style.customize_node_style(
            base_size=500,
            size_multiplier=150
        )
        style.customize_layout_style(
            font_size=12,
            title_font_size=18
        )
        return style
