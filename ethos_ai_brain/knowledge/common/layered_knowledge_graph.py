#!/usr/bin/env python3
"""
Layered Knowledge Graph - A knowledge graph composed of multiple interconnected layers
Each layer is a separate KnowledgeGraph with Z-indexing for 3D positioning and cross-layer relationships
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from .knowledge_graph import KnowledgeGraph, KnowledgeGraphRegistry
from .knowledge_system import KnowledgeSystem, KnowledgeResult

logger = logging.getLogger(__name__)


class LayeredKnowledgeGraph(KnowledgeSystem):
    """
    A layered knowledge graph - a single knowledge unit composed of multiple interconnected layers
    Each layer is a KnowledgeGraph positioned at different Z-levels with cross-layer relationships
    Represents complex knowledge structures like LLM prompts, documents, conversations, etc.
    """

    def __init__(self, network_id: str = None, z_separation: float = 2.0):
        """
        Initialize the layered knowledge graph

        Args:
            network_id: Unique identifier for this layered knowledge graph
            z_separation: Default Z-axis separation between layers
        """
        # Initialize base class
        super().__init__()
        
        self._network_id = network_id or f"layered_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.z_separation = z_separation

        # Layer management - each "graph" is actually a layer in this layered knowledge graph
        self.graphs: Dict[str, KnowledgeGraph] = {}  # layer_id -> KnowledgeGraph (layer)
        self.z_indices: Dict[str, float] = {}  # layer_id -> Z position
        self.graph_order: List[str] = []  # Ordered list of layer IDs (by Z-index)

        # Color scheme management for consistent visualization across layers
        self.graph_colors: Dict[str, str] = {}  # layer_id -> color
        self.default_color_palette = [
            '#FF6B6B',  # Red
            '#45B7D1',  # Blue
            '#96CEB4',  # Green
            '#FECA57',  # Yellow
            '#FF9FF3',  # Pink
            '#A29BFE',  # Purple
            '#FD79A8',  # Rose
            '#FDCB6E',  # Orange
            '#6C5CE7',  # Violet
            '#74B9FF'  # Light Blue
        ]

        # Physics settings optimized for 100x100 coordinate system
        self.physics_enabled = True
        self.physics_iterations = 50
        self.physics_params = {
            'k_spring': 0.5,  # Spring constant for connected nodes
            'k_repulsion': 3.0,  # Repulsion constant for all nodes
            'damping': 0.9,  # Velocity damping
            'dt': 0.2,  # Time step
            'ideal_length': 12.0  # Ideal edge length for 100x100 space
        }

        # Registry integration
        self.registry = KnowledgeGraphRegistry()

        # Update base metadata with layered graph-specific info
        self.update_metadata({
            "network_id": self._network_id,
            "z_separation": z_separation,
            "layer_count": 0
        })

        logger.info(f"LayeredKnowledgeGraph created: {self._network_id}")

    # ========================================
    # UnifiedKnowledgeInterface Properties
    # ========================================
    
    @property
    def knowledge_id(self) -> str:
        """Unique identifier for this layered knowledge graph"""
        return self._network_id
    
    @property
    def knowledge_type(self) -> str:
        """Type of knowledge structure"""
        return "layered_knowledge_graph"
    
    @property
    def network_id(self) -> str:
        """Legacy property for backward compatibility"""
        return self._network_id

    # ========================================
    # Layer Management
    # ========================================

    def add_layer(self, layer: KnowledgeGraph, z_index: Optional[float] = None,
                  auto_position: bool = True) -> None:
        """
        Add a layer to this layered knowledge graph

        Args:
            layer: KnowledgeGraph instance to add as a layer
            z_index: Specific Z position (auto-calculated if None)
            auto_position: Whether to automatically apply physics positioning
        """
        layer_id = layer.graph_id

        if layer_id in self.graphs:
            logger.warning(f"Layer {layer_id} already exists, replacing...")

        # Calculate Z index if not provided
        if z_index is None:
            z_index = self._calculate_auto_z_index(layer)

        # Add to collections
        self.graphs[layer_id] = layer
        self.z_indices[layer_id] = z_index

        if layer_id not in self.graph_order:
            # Insert in Z-order (highest Z first)
            inserted = False
            for i, existing_id in enumerate(self.graph_order):
                if z_index > self.z_indices[existing_id]:
                    self.graph_order.insert(i, layer_id)
                    inserted = True
                    break
            if not inserted:
                self.graph_order.append(layer_id)

        # Assign color if not already assigned
        if layer_id not in self.graph_colors:
            color_index = len(self.graph_colors) % len(self.default_color_palette)
            self.graph_colors[layer_id] = self.default_color_palette[color_index]

        # Apply positioning if requested
        if auto_position:
            self._apply_graph_positioning(layer_id)

        # Update layer count in metadata
        self.set_metadata("layer_count", len(self.graphs))

        logger.info(f"Added layer {layer_id} at Z={z_index}")

    def remove_graph(self, graph_id: str) -> Optional[KnowledgeGraph]:
        """
        Remove a knowledge graph from the brain network

        Args:
            graph_id: ID of the graph to remove

        Returns:
            The removed KnowledgeGraph instance, or None if not found
        """
        if graph_id not in self.graphs:
            logger.warning(f"Graph {graph_id} not found for removal")
            return None

        # Remove from collections
        graph = self.graphs.pop(graph_id)
        self.z_indices.pop(graph_id, None)
        if graph_id in self.graph_order:
            self.graph_order.remove(graph_id)

        logger.info(f"Removed graph {graph_id}")
        return graph

    def change_graph_z_index(self, graph_id: str, new_z_index: float,
                             auto_reposition: bool = True) -> bool:
        """
        Change the Z-index of a specific graph

        Args:
            graph_id: ID of the graph to modify
            new_z_index: New Z position
            auto_reposition: Whether to automatically reposition nodes

        Returns:
            True if successful, False if graph not found
        """
        if graph_id not in self.graphs:
            logger.warning(f"Graph {graph_id} not found for Z-index change")
            return False

        old_z = self.z_indices[graph_id]
        self.z_indices[graph_id] = new_z_index

        # Update order
        self.graph_order.remove(graph_id)
        inserted = False
        for i, existing_id in enumerate(self.graph_order):
            if new_z_index > self.z_indices[existing_id]:
                self.graph_order.insert(i, graph_id)
                inserted = True
                break
        if not inserted:
            self.graph_order.append(graph_id)

        # Reposition nodes if requested
        if auto_reposition:
            self._apply_graph_positioning(graph_id)

        logger.info(f"Changed graph {graph_id} Z-index: {old_z} -> {new_z_index}")
        return True

    def set_z_separation(self, separation: float, auto_redistribute: bool = True) -> None:
        """
        Set the Z-axis separation between graph layers

        Args:
            separation: New separation distance
            auto_redistribute: Whether to automatically redistribute existing graphs
        """
        old_separation = self.z_separation
        self.z_separation = separation

        if auto_redistribute and self.graphs:
            self._redistribute_z_indices()

        logger.info(f"Changed Z separation: {old_separation} -> {separation}")

    def get_graph(self, graph_id: str) -> Optional[KnowledgeGraph]:
        """Get a specific graph by ID"""
        return self.graphs.get(graph_id)

    def list_graphs(self) -> List[Tuple[str, float, str]]:
        """
        List all graphs with their Z-indices and types

        Returns:
            List of (graph_id, z_index, graph_type) tuples, ordered by Z-index
        """
        return [(graph_id, self.z_indices[graph_id], self.graphs[graph_id].graph_type.value)
                for graph_id in self.graph_order]

    def get_graph_color(self, graph_id: str) -> Optional[str]:
        """Get the assigned color for a specific graph"""
        return self.graph_colors.get(graph_id)

    def set_graph_color(self, graph_id: str, color: str) -> bool:
        """
        Set the color for a specific graph

        Args:
            graph_id: ID of the graph
            color: Hex color string (e.g., '#FF6B6B')

        Returns:
            True if successful, False if graph not found
        """
        if graph_id not in self.graphs:
            logger.warning(f"Graph {graph_id} not found for color assignment")
            return False

        self.graph_colors[graph_id] = color
        logger.info(f"Set color for graph {graph_id}: {color}")
        return True

    def get_all_graph_colors(self) -> Dict[str, str]:
        """Get all graph colors as a dictionary"""
        return self.graph_colors.copy()

    # ========================================
    # Query and Analysis Methods
    # ========================================

    def find_critical_path_nodes(self) -> List[str]:
        """Find all nodes marked as critical or active across all graphs"""
        critical_nodes = []
        for graph_id, graph in self.graphs.items():
            for node_id in graph.nodes():
                attrs = graph.get_node_attributes(node_id)
                if attrs.get('priority') == 'critical' or attrs.get('status') == 'active':
                    critical_nodes.append(f"{graph_id}:{node_id}")
        return critical_nodes

    def detect_bottlenecks(self, min_connections: int = 3) -> List[Tuple[str, int]]:
        """Detect potential bottlenecks based on node connectivity"""
        bottlenecks = []
        for graph_id, graph in self.graphs.items():
            for node_id in graph.nodes():
                degree = graph.degree(node_id)
                ext_refs = len(graph.external_references.get(node_id, {}))
                total_connections = degree + ext_refs
                if total_connections >= min_connections:
                    bottlenecks.append((f"{graph_id}:{node_id}", total_connections))

        return sorted(bottlenecks, key=lambda x: x[1], reverse=True)

    def analyze_cross_graph_dependencies(self) -> Dict[str, List[str]]:
        """Analyze dependencies between graphs"""
        dependencies = {}
        for graph_id, graph in self.graphs.items():
            deps = []
            for local_node, refs in graph.external_references.items():
                for ref_key, ref_data in refs.items():
                    target_graph = ref_data['target_graph']
                    relationship = ref_data.get('relationship', 'unknown')
                    deps.append(f"{target_graph} ({relationship})")
            if deps:
                dependencies[graph_id] = list(set(deps))  # Remove duplicates
        return dependencies

    def analyze_resource_utilization(self) -> Dict[str, Dict[str, int]]:
        """Analyze resource utilization across all graphs"""
        resource_stats = {}
        for graph_id, graph in self.graphs.items():
            stats = {
                'total_nodes': len(graph.nodes()),
                'total_edges': len(graph.edges()),
                'external_refs': sum(len(refs) for refs in graph.external_references.values()),
                'active_tasks': 0,
                'completed_tasks': 0
            }

            for node_id in graph.nodes():
                attrs = graph.get_node_attributes(node_id)
                if attrs.get('type') == 'task':
                    if attrs.get('status') == 'active':
                        stats['active_tasks'] += 1
                    elif attrs.get('status') == 'completed':
                        stats['completed_tasks'] += 1

            resource_stats[graph_id] = stats
        return resource_stats

    def find_coordination_hubs(self) -> List[Dict[str, Any]]:
        """Find coordination hubs across all graphs"""
        coordination_hubs = []
        for graph_id, graph in self.graphs.items():
            for node_id in graph.nodes():
                attrs = graph.get_node_attributes(node_id)
                if (attrs.get('type') == 'coordination' or
                        'coordination' in node_id.lower() or
                        'hub' in node_id.lower()):
                    internal_degree = graph.degree(node_id)
                    external_refs = len(graph.external_references.get(node_id, {}))
                    total_connections = internal_degree + external_refs

                    coordination_hubs.append({
                        'node': f"{graph_id}:{node_id}",
                        'connections': total_connections,
                        'status': attrs.get('status', 'unknown'),
                        'type': attrs.get('type', 'unknown')
                    })

        return sorted(coordination_hubs, key=lambda x: x['connections'], reverse=True)

    def analyze_z_layer_distribution(self) -> Dict[float, List[str]]:
        """Analyze node distribution across Z-layers"""
        z_distribution = {}
        all_positions = self.get_all_nodes_3d()

        for unified_node_id, pos in all_positions.items():
            z_level = round(pos[2], 1)  # Round to nearest 0.1
            if z_level not in z_distribution:
                z_distribution[z_level] = []
            z_distribution[z_level].append(unified_node_id)

        return z_distribution

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run all analysis queries and return comprehensive results"""
        return {
            'critical_path_nodes': self.find_critical_path_nodes(),
            'bottlenecks': self.detect_bottlenecks(),
            'cross_graph_dependencies': self.analyze_cross_graph_dependencies(),
            'resource_utilization': self.analyze_resource_utilization(),
            'coordination_hubs': self.find_coordination_hubs(),
            'z_layer_distribution': self.analyze_z_layer_distribution()
        }

    # ========================================
    # Physics Management
    # ========================================

    def set_physics_params(self, **params) -> None:
        """
        Update physics parameters

        Args:
            **params: Physics parameters (k_spring, k_repulsion, damping, dt, ideal_length)
        """
        for key, value in params.items():
            if key in self.physics_params:
                old_value = self.physics_params[key]
                self.physics_params[key] = value
                logger.info(f"Physics param {key}: {old_value} -> {value}")

    def enable_physics(self, enabled: bool = True) -> None:
        """Enable or disable physics positioning"""
        self.physics_enabled = enabled
        logger.info(f"Physics {'enabled' if enabled else 'disabled'}")

    def apply_physics_to_all(self, iterations: Optional[int] = None) -> None:
        """
        Apply physics positioning to all graphs

        Args:
            iterations: Number of physics iterations (uses default if None)
        """
        if not self.physics_enabled:
            logger.info("Physics disabled, skipping positioning")
            return

        iterations = iterations or self.physics_iterations

        for graph_id in self.graphs:
            self._apply_graph_positioning(graph_id, iterations)

        logger.info(f"Applied physics to {len(self.graphs)} graphs ({iterations} iterations)")

    def apply_physics_to_graph(self, graph_id: str, iterations: Optional[int] = None) -> bool:
        """
        Apply physics positioning to a specific graph

        Args:
            graph_id: ID of the graph to position
            iterations: Number of physics iterations

        Returns:
            True if successful, False if graph not found
        """
        if graph_id not in self.graphs:
            logger.warning(f"Graph {graph_id} not found for physics application")
            return False

        if not self.physics_enabled:
            logger.info("Physics disabled, skipping positioning")
            return False

        iterations = iterations or self.physics_iterations
        self._apply_graph_positioning(graph_id, iterations)

        logger.info(f"Applied physics to graph {graph_id} ({iterations} iterations)")
        return True

    # ========================================
    # Network Operations
    # ========================================

    def get_unified_network_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the unified network

        Returns:
            Dictionary with network topology and statistics
        """
        total_nodes = sum(len(graph.nodes()) for graph in self.graphs.values())
        total_edges = sum(len(graph.edges()) for graph in self.graphs.values())

        # Count cross-graph connections
        cross_connections = []
        for graph in self.graphs.values():
            for local_node, refs in graph.external_references.items():
                for ref_key, ref_data in refs.items():
                    cross_connections.append({
                        'source_graph': graph.graph_id,
                        'source_node': local_node,
                        'target_graph': ref_data['target_graph'],
                        'target_node': ref_data['target_node'],
                        'relationship': ref_data.get('relationship', 'references')
                    })

        return {
            'network_id': self.network_id,
            'total_graphs': len(self.graphs),
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'cross_connections': len(cross_connections),
            'z_separation': self.z_separation,
            'physics_enabled': self.physics_enabled,
            'graphs': [
                {
                    'graph_id': graph_id,
                    'graph_type': self.graphs[graph_id].graph_type.value,
                    'z_index': self.z_indices[graph_id],
                    'node_count': len(self.graphs[graph_id].nodes()),
                    'edge_count': len(self.graphs[graph_id].edges())
                }
                for graph_id in self.graph_order
            ],
            'cross_graph_connections': cross_connections
        }

    def get_all_nodes_3d(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Get 3D positions of all nodes across all graphs

        Returns:
            Dictionary mapping "graph_id:node_id" to (x, y, z) positions
        """
        all_positions = {}

        for graph_id, graph in self.graphs.items():
            for node_id in graph.nodes():
                pos_3d = graph.get_node_3d_position(node_id)
                if pos_3d:
                    unified_node_id = f"{graph_id}:{node_id}"
                    all_positions[unified_node_id] = pos_3d

        return all_positions

    def query_across_graphs(self, **query_params) -> Dict[str, List[Dict[str, Any]]]:
        """
        Query across all graphs in the brain network

        Args:
            **query_params: Query parameters (e.g., type="status", priority="high")

        Returns:
            Dictionary mapping graph_id to list of matching nodes
        """
        results = {}

        for graph_id, graph in self.graphs.items():
            matching_nodes = []
            for node_id in graph.nodes():
                node_attrs = graph.get_node_attributes(node_id)
                if self._matches_query(node_attrs, query_params):
                    matching_nodes.append({
                        "node_id": node_id,
                        "attributes": node_attrs,
                        "z_index": self.z_indices[graph_id]
                    })
            if matching_nodes:
                results[graph_id] = matching_nodes

        return results

    # ========================================
    # Internal Methods
    # ========================================

    def _calculate_auto_z_index(self, graph: KnowledgeGraph) -> float:
        """Calculate automatic Z-index based on graph type and existing graphs"""
        base_z = {
            'mission': 10.0,
            'tactical': 5.0,
            'local_rag': 2.0,
            'cloud_rag': 0.0
        }.get(graph.graph_type.value, 5.0)

        # Check for existing graphs of the same type
        same_type_graphs = [
            gid for gid, g in self.graphs.items()
            if g.graph_type == graph.graph_type
        ]

        if same_type_graphs:
            # Offset by separation for each additional graph of same type
            offset = len(same_type_graphs) * self.z_separation
            return base_z - offset

        return base_z

    def _apply_graph_positioning(self, graph_id: str, iterations: Optional[int] = None) -> None:
        """Apply physics-based positioning to a specific graph"""
        if graph_id not in self.graphs:
            return

        graph = self.graphs[graph_id]
        z_index = self.z_indices[graph_id]
        iterations = iterations or self.physics_iterations

        # Apply spring forces for X,Y positioning
        nodes = list(graph.nodes())
        if nodes:
            positions_2d = self._apply_spring_forces(graph, nodes, iterations)

            # Apply Z-index and 2D positions
            for node_id in nodes:
                if node_id in positions_2d:
                    x, y = positions_2d[node_id]
                    graph.set_node_3d_position(node_id, x, y, z_index)

    def _apply_spring_forces(self, graph: KnowledgeGraph, nodes: List[str],
                             iterations: int) -> Dict[str, Tuple[float, float]]:
        """Apply spring-force physics for better node distribution in 2D"""
        import random
        import math

        if not nodes:
            return {}

        # Initialize random positions within 0-100 bounds with maximum spread
        positions = {}
        for node_id in nodes:
            positions[node_id] = (random.uniform(5, 95), random.uniform(5, 95))

        # Get physics parameters
        k_spring = self.physics_params['k_spring']
        k_repulsion = self.physics_params['k_repulsion']
        damping = self.physics_params['damping']
        dt = self.physics_params['dt']
        ideal_length = self.physics_params['ideal_length']

        # Initialize velocities
        velocities = {node_id: (0.0, 0.0) for node_id in nodes}

        # Run physics simulation
        for iteration in range(iterations):
            forces = {node_id: [0.0, 0.0] for node_id in nodes}

            # Calculate repulsion forces between all pairs
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes):
                    if i >= j:  # Avoid duplicate calculations
                        continue

                    pos1 = positions[node1]
                    pos2 = positions[node2]

                    # Distance vector
                    dx = pos1[0] - pos2[0]
                    dy = pos1[1] - pos2[1]
                    distance = math.sqrt(dx * dx + dy * dy)

                    if distance < 0.1:  # Avoid division by zero
                        distance = 0.1
                        dx = random.uniform(-0.1, 0.1)
                        dy = random.uniform(-0.1, 0.1)

                    # Repulsion force (inverse square law)
                    force_magnitude = k_repulsion / (distance * distance)
                    force_x = (dx / distance) * force_magnitude
                    force_y = (dy / distance) * force_magnitude

                    # Apply equal and opposite forces
                    forces[node1][0] += force_x
                    forces[node1][1] += force_y
                    forces[node2][0] -= force_x
                    forces[node2][1] -= force_y

            # Calculate attraction forces for connected nodes
            for source, target in graph.edges():
                if source in positions and target in positions:
                    pos1 = positions[source]
                    pos2 = positions[target]

                    # Distance vector
                    dx = pos2[0] - pos1[0]
                    dy = pos2[1] - pos1[1]
                    distance = math.sqrt(dx * dx + dy * dy)

                    if distance > 0.1:  # Only apply if nodes are apart
                        # Spring force (Hooke's law)
                        force_magnitude = k_spring * (distance - ideal_length)
                        force_x = (dx / distance) * force_magnitude
                        force_y = (dy / distance) * force_magnitude

                        # Apply forces
                        forces[source][0] += force_x
                        forces[source][1] += force_y
                        forces[target][0] -= force_x
                        forces[target][1] -= force_y

            # Update velocities and positions
            for node_id in nodes:
                # Update velocity
                vx, vy = velocities[node_id]
                fx, fy = forces[node_id]

                vx = (vx + fx * dt) * damping
                vy = (vy + fy * dt) * damping

                velocities[node_id] = (vx, vy)

                # Update position
                x, y = positions[node_id]
                x += vx * dt
                y += vy * dt

                # Keep nodes within 0-100 bounds
                x = max(0, min(100, x))
                y = max(0, min(100, y))

                positions[node_id] = (x, y)

        return positions

    def _redistribute_z_indices(self) -> None:
        """Redistribute Z-indices of existing graphs based on new separation"""
        # Group by type
        type_groups = {}
        for graph_id, graph in self.graphs.items():
            graph_type = graph.graph_type.value
            if graph_type not in type_groups:
                type_groups[graph_type] = []
            type_groups[graph_type].append(graph_id)

        # Reassign Z-indices
        base_z_values = {
            'mission': 10.0,
            'tactical': 5.0,
            'local_rag': 2.0,
            'cloud_rag': 0.0
        }

        for graph_type, graph_ids in type_groups.items():
            base_z = base_z_values.get(graph_type, 5.0)
            for i, graph_id in enumerate(graph_ids):
                new_z = base_z - (i * self.z_separation)
                self.z_indices[graph_id] = new_z

        # Update order
        self.graph_order.sort(key=lambda gid: self.z_indices[gid], reverse=True)

    def _matches_query(self, attributes: Dict[str, Any], query_params: Dict[str, Any]) -> bool:
        """Check if node attributes match query parameters"""
        for key, value in query_params.items():
            if key not in attributes or attributes[key] != value:
                return False
        return True

    def __str__(self) -> str:
        """String representation of the layered knowledge graph"""
        return f"LayeredKnowledgeGraph(id={self.network_id}, layers={len(self.graphs)}, z_sep={self.z_separation})"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__()

    # ========================================
    # UnifiedKnowledgeInterface Implementation
    # ========================================

    def query(self, query: str, context: Dict = None, **filters) -> List[KnowledgeResult]:
        """Query across all layers of this layered knowledge graph"""
        try:
            results = []
            
            # Query each layer
            for layer_name, layer_graph in self.graphs.items():
                # Get all nodes in this layer
                for node_id in layer_graph.nodes():
                    node_attrs = layer_graph.get_node_attributes(node_id)
                    
                    # Simple text matching for now
                    node_text = f"{node_id} {str(node_attrs)}"
                    if query.lower() in node_text.lower():
                        result = KnowledgeResult(
                            content=f"Layer {layer_name}, Node: {node_id}",
                            knowledge_id=f"{self.network_id}:{layer_name}:{node_id}",
                            knowledge_type="layered_graph_node",
                            confidence=0.9,
                            metadata=node_attrs,
                            context_path=[self.network_id, layer_name, node_id]
                        )
                        results.append(result)
            
            return results
            
        except Exception as e:
            logger.warning(f"LayeredKnowledgeGraph query failed: {e}")
            return []

    def add_knowledge(self, content: Any, metadata: Dict = None) -> str:
        """Add knowledge to the first available layer"""
        try:
            # For now, add to first available layer
            if self.graphs:
                first_layer_name = list(self.graphs.keys())[0]
                first_layer = self.graphs[first_layer_name]
                
                if isinstance(content, str):
                    node_id = content
                else:
                    node_id = str(content)
                
                success = first_layer.add_node(node_id, **(metadata or {}))
                return f"{self.network_id}:{first_layer_name}:{node_id}" if success else ""
            
            return ""
            
        except Exception as e:
            logger.warning(f"Failed to add knowledge to layered graph: {e}")
            return ""

    def get_related(self, knowledge_id: str, relationship_type: str = "semantic") -> List[KnowledgeResult]:
        """Find related knowledge across layers"""
        try:
            # Parse knowledge_id: network_id:layer_name:node_id
            parts = knowledge_id.split(":")
            if len(parts) >= 3:
                layer_name = parts[1]
                node_id = parts[2]
                
                results = []
                
                # Get related nodes in same layer
                if layer_name in self.graphs:
                    layer_graph = self.graphs[layer_name]
                    if hasattr(layer_graph, 'graph') and hasattr(layer_graph.graph, 'neighbors'):
                        neighbors = list(layer_graph.graph.neighbors(node_id))
                        
                        for neighbor_id in neighbors:
                            neighbor_attrs = layer_graph.get_node_attributes(neighbor_id)
                            
                            result = KnowledgeResult(
                                content=f"Related in {layer_name}: {neighbor_id}",
                                knowledge_id=f"{self.network_id}:{layer_name}:{neighbor_id}",
                                knowledge_type="layered_graph_node",
                                confidence=0.8,
                                metadata=neighbor_attrs,
                                context_path=[self.network_id, layer_name, neighbor_id]
                            )
                            results.append(result)
                
                return results
            
            return []
            
        except Exception as e:
            logger.warning(f"Failed to get related knowledge: {e}")
            return []

    def get_capabilities(self) -> Dict[str, bool]:
        """Return capabilities of layered knowledge graph"""
        return {
            "semantic_search": True,
            "relationship_traversal": True,
            "cross_layer_queries": True,
            "temporal_queries": False,
            "multi_dimensional_clustering": False,
            "3d_positioning": True,
            "physics_simulation": True,
            "cross_graph_analysis": True
        }
    
    def get_components(self) -> List[str]:
        """Get components (layers for layered graphs)"""
        return list(self.graphs.keys())
    
    def get_relationships(self) -> List[Tuple[str, str, Dict]]:
        """Get relationships (cross-layer connections)"""
        relationships = []
        # Add cross-layer connections based on external references
        for layer_id, layer_graph in self.graphs.items():
            for node_id, refs in layer_graph.external_references.items():
                for ref_key, ref_data in refs.items():
                    target_layer = ref_data.get('target_graph', '')
                    target_node = ref_data.get('target_node', '')
                    if target_layer in self.graphs:
                        relationships.append((
                            f"{layer_id}:{node_id}",
                            f"{target_layer}:{target_node}",
                            dict(ref_data)
                        ))
        return relationships
