#!/usr/bin/env python3
"""
Brain Knowledge Graph - High-level orchestrator for managing multiple knowledge graphs
Handles Z-indexing, physics coordination, and unified network operations
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from .knowledge_graph import KnowledgeGraph, KnowledgeGraphRegistry

logger = logging.getLogger(__name__)


class KnowledgeGraphManager:
    """
    High-level orchestrator for managing multiple knowledge graphs
    Provides unified network management with Z-indexing and physics coordination
    """

    def __init__(self, brain_id: str = None, z_separation: float = 2.0):
        """
        Initialize the brain knowledge graph manager

        Args:
            brain_id: Unique identifier for this brain instance
            z_separation: Default Z-axis separation between graph layers
        """
        self.brain_id = brain_id or f"brain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.z_separation = z_separation

        # Graph management
        self.graphs: Dict[str, KnowledgeGraph] = {}  # graph_id -> KnowledgeGraph
        self.z_indices: Dict[str, float] = {}  # graph_id -> Z position
        self.graph_order: List[str] = []  # Ordered list of graph IDs

        # Color scheme management for consistent visualization
        self.graph_colors: Dict[str, str] = {}  # graph_id -> color
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

        logger.info(f"KnowledgeGraphManager created: {self.brain_id}")

    # ========================================
    # Graph Management
    # ========================================

    def add_graph(self, graph: KnowledgeGraph, z_index: Optional[float] = None,
                  auto_position: bool = True) -> None:
        """
        Add a knowledge graph to the brain network

        Args:
            graph: KnowledgeGraph instance to add
            z_index: Specific Z position (auto-calculated if None)
            auto_position: Whether to automatically apply physics positioning
        """
        graph_id = graph.graph_id

        if graph_id in self.graphs:
            logger.warning(f"Graph {graph_id} already exists, replacing...")

        # Calculate Z index if not provided
        if z_index is None:
            z_index = self._calculate_auto_z_index(graph)

        # Add to collections
        self.graphs[graph_id] = graph
        self.z_indices[graph_id] = z_index

        if graph_id not in self.graph_order:
            # Insert in Z-order (highest Z first)
            inserted = False
            for i, existing_id in enumerate(self.graph_order):
                if z_index > self.z_indices[existing_id]:
                    self.graph_order.insert(i, graph_id)
                    inserted = True
                    break
            if not inserted:
                self.graph_order.append(graph_id)

        # Assign color if not already assigned
        if graph_id not in self.graph_colors:
            color_index = len(self.graph_colors) % len(self.default_color_palette)
            self.graph_colors[graph_id] = self.default_color_palette[color_index]

        # Apply positioning if requested
        if auto_position:
            self._apply_graph_positioning(graph_id)

        logger.info(f"Added graph {graph_id} at Z={z_index}")

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
            'brain_id': self.brain_id,
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
        """String representation of the brain knowledge graph"""
        return f"BrainKnowledgeGraph(id={self.brain_id}, graphs={len(self.graphs)}, z_sep={self.z_separation})"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__()
