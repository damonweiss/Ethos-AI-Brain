"""
Knowledge Graph System - Core graph data structures and operations
Clean version without visualization logic (moved to separate files)
"""

import logging
import json
import networkx as nx
from networkx.readwrite import json_graph
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum
logger = logging.getLogger(__name__)

class GraphType(str, Enum):
    """Types of knowledge graphs"""
    INTENT = "intent"
    EXECUTION = "execution"
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
    """
    Base class for all knowledge graphs using NetworkX
    Provides common functionality through composition and cross-graph operations
    """
    
    def __init__(self, graph_type: GraphType, graph_id: str = None, 
                 directed: bool = True, auto_register: bool = True):
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
    # Core Graph Operations (NetworkX delegation)
    # ========================================
    
    def add_node(self, node_id: str, **attributes) -> None:
        """Add a node with attributes"""
        self.graph.add_node(node_id, **attributes)
        self._update_modified_time()
        logger.debug(f"Added node: {node_id} with attributes: {attributes}")
    
    def add_edge(self, source: str, target: str, **attributes) -> None:
        """Add an edge with attributes"""
        self.graph.add_edge(source, target, **attributes)
        self._update_modified_time()
        logger.debug(f"Added edge: {source} -> {target} with attributes: {attributes}")
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges"""
        if self.has_node(node_id):
            self.graph.remove_node(node_id)
            # Remove from external references
            if node_id in self.external_references:
                del self.external_references[node_id]
            # Remove from 3D positions
            if node_id in self.node_3d_positions:
                del self.node_3d_positions[node_id]
            self._update_modified_time()
            logger.debug(f"Removed node: {node_id}")
    
    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge"""
        if self.has_edge(source, target):
            self.graph.remove_edge(source, target)
            self._update_modified_time()
            logger.debug(f"Removed edge: {source} -> {target}")
    
    def has_node(self, node_id: str) -> bool:
        """Check if node exists"""
        return self.graph.has_node(node_id)
    
    def has_edge(self, source: str, target: str) -> bool:
        """Check if edge exists"""
        return self.graph.has_edge(source, target)
    
    def nodes(self) -> List[str]:
        """Get all node IDs"""
        return list(self.graph.nodes())
    
    def edges(self) -> List[Tuple[str, str]]:
        """Get all edges"""
        return list(self.graph.edges())
    
    def neighbors(self, node_id: str) -> List[str]:
        """Get neighbors of a node"""
        return list(self.graph.neighbors(node_id))
    
    def successors(self, node_id: str) -> List[str]:
        """Get successors of a node (for directed graphs)"""
        if isinstance(self.graph, nx.DiGraph):
            return list(self.graph.successors(node_id))
        return self.neighbors(node_id)
    
    def predecessors(self, node_id: str) -> List[str]:
        if isinstance(self.graph, nx.DiGraph):
            return list(self.graph.predecessors(node_id))
        return self.neighbors(node_id)
    
    def degree(self, node_id: str) -> int:
        """Get the degree (number of connections) of a node"""
        return len(self.graph.adj.get(node_id, {}))
    
    def __len__(self) -> int:
        """Return the number of nodes in the graph"""
        return len(self.graph.nodes)
    
    def __iter__(self):
        """Make the graph iterable over its nodes (for NetworkX compatibility)"""
        return iter(self.graph.nodes)
    
    def get_node_attributes(self, node_id: str) -> Dict[str, Any]:
        """Get all attributes of a node"""
        return dict(self.graph.nodes[node_id]) if self.has_node(node_id) else {}
    
    def get_edge_attributes(self, source: str, target: str) -> Dict[str, Any]:
        """Get all attributes of an edge"""
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
    # Cross-Graph Operations
    # ========================================
    
    def add_external_reference(self, local_node: str, target_graph: str, 
                              target_node: str, **metadata) -> None:
        """
        Add a reference to a node in another graph
        
        Args:
            local_node: Node in this graph
            target_graph: ID of target graph
            target_node: Node in target graph
            **metadata: Additional metadata (relationship, etc.)
        """
        if local_node not in self.external_references:
            self.external_references[local_node] = {}
        
        ref_key = f"{target_graph}:{target_node}"
        self.external_references[local_node][ref_key] = {
            "target_graph": target_graph,
            "target_node": target_node,
            **metadata
        }
        self._update_modified_time()
        logger.debug(f"Added external reference: {local_node} -> {target_graph}:{target_node}")
    
    def remove_external_reference(self, local_node: str, target_graph: str, target_node: str) -> None:
        """Remove an external reference"""
        if local_node in self.external_references:
            ref_key = f"{target_graph}:{target_node}"
            if ref_key in self.external_references[local_node]:
                del self.external_references[local_node][ref_key]
                if not self.external_references[local_node]:
                    del self.external_references[local_node]
                self._update_modified_time()
                logger.debug(f"Removed external reference: {local_node} -> {target_graph}:{target_node}")
    
    def resolve_external_reference(self, local_node: str, target_graph: str, target_node: str) -> Optional[Dict[str, Any]]:
        """
        Resolve an external reference to get the actual node data
        
        Returns:
            Dictionary with node data from target graph, or None if not found
        """
        target_graph_obj = self.registry.get(target_graph)
        if target_graph_obj and target_graph_obj.has_node(target_node):
            return {
                "graph_id": target_graph,
                "node_id": target_node,
                "attributes": target_graph_obj.get_node_attributes(target_node)
            }
        return None
    
    def hive_mind_query(self, **query_params) -> Dict[str, List[Dict[str, Any]]]:
        """
        Query across all connected graphs (hive-mind functionality)
        
        Args:
            **query_params: Query parameters (e.g., type="status", priority="high")
            
        Returns:
            Dictionary mapping graph_id to list of matching nodes
        """
        results = {}
        
        # Query this graph
        matching_nodes = []
        for node_id in self.nodes():
            node_attrs = self.get_node_attributes(node_id)
            if self._matches_query(node_attrs, query_params):
                matching_nodes.append({
                    "node_id": node_id,
                    "attributes": node_attrs
                })
        if matching_nodes:
            results[self.graph_id] = matching_nodes
        
        # Query connected graphs
        connected_graphs = set()
        for refs in self.external_references.values():
            for ref_data in refs.values():
                connected_graphs.add(ref_data['target_graph'])
        
        for graph_id in connected_graphs:
            target_graph = self.registry.get(graph_id)
            if target_graph:
                matching_nodes = []
                for node_id in target_graph.nodes():
                    node_attrs = target_graph.get_node_attributes(node_id)
                    if self._matches_query(node_attrs, query_params):
                        matching_nodes.append({
                            "node_id": node_id,
                            "attributes": node_attrs
                        })
                if matching_nodes:
                    results[graph_id] = matching_nodes
        
        return results
    
    def get_connected_knowledge_network(self) -> Dict[str, Any]:
        """
        Get information about the entire connected knowledge network
        
        Returns:
            Dictionary with network topology and cross-graph connections
        """
        # Get all connected graphs
        connected_graphs = {}
        cross_graph_connections = []
        
        for local_node, refs in self.external_references.items():
            for ref_key, ref_data in refs.items():
                target_graph_id = ref_data['target_graph']
                target_node = ref_data['target_node']
                
                # Add to connected graphs
                if target_graph_id not in connected_graphs:
                    target_graph = self.registry.get(target_graph_id)
                    if target_graph:
                        connected_graphs[target_graph_id] = {
                            "graph_type": target_graph.graph_type.value,
                            "node_count": len(target_graph.nodes()),
                            "edge_count": len(target_graph.edges())
                        }
                
                # Add to cross-graph connections
                cross_graph_connections.append({
                    "source_graph": self.graph_id,
                    "source_node": local_node,
                    "target_graph": target_graph_id,
                    "target_node": target_node,
                    "relationship": ref_data.get('relationship', 'references')
                })
        
        return {
            "root_graph": {
                "graph_id": self.graph_id,
                "graph_type": self.graph_type.value,
                "node_count": len(self.nodes()),
                "edge_count": len(self.edges())
            },
            "connected_graphs": connected_graphs,
            "cross_graph_connections": cross_graph_connections
        }
    
    def _matches_query(self, attributes: Dict[str, Any], query_params: Dict[str, Any]) -> bool:
        """Check if node attributes match query parameters"""
        for key, value in query_params.items():
            if key not in attributes or attributes[key] != value:
                return False
        return True
    
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
        import math
        
        # Get base Z level for this specific graph instance
        # Each graph gets its own unique Z level based on graph_id
        if self.graph_type.value == 'mission':
            graph_z_base = 10.0
        elif self.graph_type.value == 'tactical':
            # Different Z levels for different tactical graphs
            if 'sarah' in self.graph_id.lower() or 'data_analyst' in self.graph_id.lower():
                graph_z_base = 6.0  # DataAnalyst at Z=6
            elif 'marcus' in self.graph_id.lower() or 'security' in self.graph_id.lower():
                graph_z_base = 4.0  # SecurityExpert at Z=4
            else:
                graph_z_base = 5.0  # Other tactical graphs at Z=5
        elif self.graph_type.value == 'local_rag':
            graph_z_base = 2.0
        elif self.graph_type.value == 'cloud_rag':
            graph_z_base = 0.0
        else:
            graph_z_base = 5.0
        
        # Group nodes by type within this graph
        type_layers = {}
        for node_id in self.nodes():
            node_type = self.get_node_attributes(node_id).get('type', 'default')
            if node_type not in type_layers:
                type_layers[node_type] = []
            type_layers[node_type].append(node_id)
        
        # NO Z offsets - all nodes in same graph get same Z level
        # This ensures only 3 Z-levels total (one per graph type)
        type_z_offsets = {
            'mission': 0.0,
            'coordination': 0.0,
            'requirement': 0.0,
            'task': 0.0,
            'resource': 0.0,
            'status': 0.0,
            'output': 0.0,
            'default': 0.0
        }
        
        # Position ALL nodes - ensure every node gets a position
        all_nodes = list(self.nodes())
        if not all_nodes:
            return
            
        # If we have type layers, use them; otherwise treat all as default
        if not type_layers:
            type_layers = {'default': all_nodes}
        
        # Use physics-based positioning for better X,Y distribution
        all_nodes = list(self.nodes())
        if not all_nodes:
            return
        
        # Apply spring-force layout in 2D, then assign Z
        positions_2d = self._apply_spring_forces(all_nodes)
        
        # Assign Z level and apply 2D positions
        for node_id in all_nodes:
            if node_id in positions_2d:
                x, y = positions_2d[node_id]
                self.set_node_3d_position(node_id, x, y, graph_z_base)
        
        # Ensure ALL nodes have positions (safety check)
        for node_id in all_nodes:
            if node_id not in self.node_3d_positions:
                # Fallback position for any missed nodes
                x = random.uniform(-3, 3)
                y = random.uniform(-3, 3)
                z = graph_z_base
                self.set_node_3d_position(node_id, x, y, z)
    
    def _apply_spring_forces(self, nodes: List[str], iterations: int = 50) -> Dict[str, Tuple[float, float]]:
        """
        Apply spring-force physics for better node distribution in 2D
        
        Args:
            nodes: List of node IDs to position
            iterations: Number of physics iterations
            
        Returns:
            Dictionary mapping node_id to (x, y) positions
        """
        import random
        import math
        
        if not nodes:
            return {}
        
        # Initialize random positions
        positions = {}
        for node_id in nodes:
            positions[node_id] = (random.uniform(-2, 2), random.uniform(-2, 2))
        
        # Physics parameters
        k_spring = 1.0      # Spring constant for connected nodes (attraction)
        k_repulsion = 2.0   # Repulsion constant for all nodes
        damping = 0.9       # Velocity damping
        dt = 0.1           # Time step
        
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
                    distance = math.sqrt(dx*dx + dy*dy)
                    
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
            for source, target in self.edges():
                if source in positions and target in positions:
                    pos1 = positions[source]
                    pos2 = positions[target]
                    
                    # Distance vector
                    dx = pos2[0] - pos1[0]
                    dy = pos2[1] - pos1[1]
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance > 0.1:  # Only apply if nodes are apart
                        # Spring force (Hooke's law)
                        ideal_length = 1.5  # Ideal edge length
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
                
                # Keep nodes within reasonable bounds
                x = max(-5, min(5, x))
                y = max(-5, min(5, y))
                
                positions[node_id] = (x, y)
        
        return positions
    
    def _assign_hierarchical_3d_positions(self) -> None:
        """Assign 3D positions based on graph hierarchy"""
        import math
        
        # Get base Z level for this specific graph instance (same logic as layered)
        if self.graph_type.value == 'mission':
            graph_z_base = 10.0
        elif self.graph_type.value == 'tactical':
            if 'sarah' in self.graph_id.lower() or 'data_analyst' in self.graph_id.lower():
                graph_z_base = 6.0  # DataAnalyst at Z=6
            elif 'marcus' in self.graph_id.lower() or 'security' in self.graph_id.lower():
                graph_z_base = 4.0  # SecurityExpert at Z=4
            else:
                graph_z_base = 5.0  # Other tactical graphs at Z=5
        elif self.graph_type.value == 'local_rag':
            graph_z_base = 2.0
        elif self.graph_type.value == 'cloud_rag':
            graph_z_base = 0.0
        else:
            graph_z_base = 5.0
        
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
            level_groups = {}
            for node, level in levels.items():
                if level not in level_groups:
                    level_groups[level] = []
                level_groups[level].append(node)
            
            for level, nodes in level_groups.items():
                z_level = graph_z_base + (level * 1.0)  # Add level offset to graph base
                num_nodes = len(nodes)
                
                for i, node_id in enumerate(nodes):
                    if num_nodes == 1:
                        angle = 0
                    else:
                        angle = 2 * math.pi * i / num_nodes
                    radius = 1.5
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    self.set_node_3d_position(node_id, x, y, z_level)
            
            # Ensure ALL nodes have positions (safety check)
            all_nodes = list(self.nodes())
            for node_id in all_nodes:
                if node_id not in self.node_3d_positions:
                    # Fallback position for any missed nodes
                    import random
                    x = random.uniform(-3, 3)
                    y = random.uniform(-3, 3)
                    z = graph_z_base
                    self.set_node_3d_position(node_id, x, y, z)
                    
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
    # Utility Methods
    # ========================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        stats = {
            "node_count": len(self.nodes()),
            "edge_count": len(self.edges()),
            "graph_type": self.graph_type.value,
            "graph_id": self.graph_id
        }
        
        # Add DAG and connectivity info for directed graphs
        if isinstance(self.graph, nx.DiGraph):
            stats["is_dag"] = nx.is_directed_acyclic_graph(self.graph)
            stats["is_connected"] = nx.is_weakly_connected(self.graph) if len(self.nodes()) > 0 else True
        else:
            stats["is_connected"] = nx.is_connected(self.graph) if len(self.nodes()) > 0 else True
        
        return stats
    
    def to_json(self) -> str:
        """Serialize graph to JSON"""
        data = {
            "metadata": self.metadata,
            "nodes": [
                {"id": node_id, "attributes": self.get_node_attributes(node_id)}
                for node_id in self.nodes()
            ],
            "edges": [
                {"source": source, "target": target, "attributes": self.get_edge_attributes(source, target)}
                for source, target in self.edges()
            ],
            "external_references": self.external_references,
            "node_3d_positions": self.node_3d_positions
        }
        return json.dumps(data, indent=2, default=str)
    
    def from_json(self, json_str: str) -> None:
        """Load graph from JSON"""
        data = json.loads(json_str)
        
        # Clear existing graph
        self.graph.clear()
        self.external_references.clear()
        self.node_3d_positions.clear()
        
        # Load metadata
        if "metadata" in data:
            self.metadata.update(data["metadata"])
        
        # Load nodes
        for node_data in data.get("nodes", []):
            self.add_node(node_data["id"], **node_data.get("attributes", {}))
        
        # Load edges
        for edge_data in data.get("edges", []):
            self.add_edge(edge_data["source"], edge_data["target"], **edge_data.get("attributes", {}))
        
        # Load external references
        self.external_references = data.get("external_references", {})
        
        # Load 3D positions
        for node_id, pos in data.get("node_3d_positions", {}).items():
            if isinstance(pos, list) and len(pos) == 3:
                self.node_3d_positions[node_id] = tuple(pos)
    
    # ========================================
    # Generic Graph Analysis Methods
    # ========================================
    
    def calculate_dependency_depth(self) -> int:
        """Calculate the maximum dependency depth in the graph"""
        if not self.nodes():
            return 0
        
        max_depth = 0
        for node_id in self.nodes():
            depth = self.get_node_depth(node_id)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def get_node_depth(self, node_id: str, visited: Set[str] = None) -> int:
        """Get the dependency depth of a specific node"""
        if visited is None:
            visited = set()
        
        if node_id in visited:
            return 0  # Avoid cycles
        
        visited.add(node_id)
        
        predecessors = list(self.predecessors(node_id))
        if not predecessors:
            return 1  # Leaf node
        
        max_predecessor_depth = max(
            self.get_node_depth(pred, visited.copy()) for pred in predecessors
        )
        return max_predecessor_depth + 1
    
    def find_longest_path_from_node(self, start_node: str) -> Tuple[List[str], float]:
        """
        Find the longest path from a given start node
        Generic implementation - subclasses can override for domain-specific weighting
        """
        longest_path = []
        max_weight = 0
        
        def dfs(node, current_path, current_weight, visited_in_path):
            nonlocal longest_path, max_weight
            
            # Avoid cycles - check if node is already in current path
            if node in visited_in_path:
                return
            
            # Add node to current path
            new_path = current_path + [node]
            new_visited = visited_in_path | {node}
            
            # Add node weight to current weight (default weight = 1.0)
            try:
                node_attrs = self.get_node_attributes(node)
                node_weight = self._get_node_weight(node_attrs)
                new_weight = current_weight + node_weight
            except:
                # If we can't get attributes, use default weight
                new_weight = current_weight + 1.0
            
            # Get successors safely
            try:
                successors = list(self.successors(node))
            except:
                successors = []
            
            if not successors:
                # End of path, check if it's the longest
                if new_weight > max_weight:
                    max_weight = new_weight
                    longest_path = new_path.copy()
            else:
                # Continue exploring successors
                for successor in successors:
                    dfs(successor, new_path, new_weight, new_visited)
        
        # Start DFS with empty path and visited set
        dfs(start_node, [], 0, set())
        return longest_path, max_weight
    
    def _get_node_weight(self, node_attrs: Dict[str, Any]) -> float:
        """
        Get weight for a node - can be overridden by subclasses
        Default implementation returns 1.0 for all nodes
        """
        return 1.0
    
    def find_parallel_chains(self) -> List[List[str]]:
        """Find chains of nodes that can be processed in parallel"""
        parallel_chains = []
        visited = set()
        
        for node_id in self.nodes():
            if node_id not in visited:
                # Find nodes at the same "level" (same distance from start)
                level_nodes = self._find_nodes_at_same_level(node_id)
                if len(level_nodes) > 1:
                    parallel_chains.append(level_nodes)
                    visited.update(level_nodes)
        
        return parallel_chains
    
    def find_sequential_chains(self) -> List[List[str]]:
        """Find chains of nodes that must be processed sequentially"""
        sequential_chains = []
        visited = set()
        
        for node_id in self.nodes():
            if node_id not in visited:
                chain = self._build_sequential_chain(node_id)
                if len(chain) > 1:
                    sequential_chains.append(chain)
                    visited.update(chain)
        
        return sequential_chains
    
    def _find_nodes_at_same_level(self, start_node: str) -> List[str]:
        """Find nodes at the same dependency level as the start node"""
        # Simplified implementation - subclasses can provide more sophisticated level detection
        return [start_node]
    
    def _build_sequential_chain(self, start_node: str) -> List[str]:
        """Build a sequential chain starting from the given node"""
        chain = [start_node]
        current = start_node
        
        while True:
            successors = list(self.successors(current))
            if len(successors) == 1:
                # Single successor - continue chain
                next_node = successors[0]
                predecessors = list(self.predecessors(next_node))
                if len(predecessors) == 1:
                    # Next node has single predecessor - true sequential link
                    chain.append(next_node)
                    current = next_node
                else:
                    break  # Next node has multiple predecessors - end chain
            else:
                break  # Multiple or no successors - end chain
        
        return chain
    
    def _update_modified_time(self):
        """Update the last modified timestamp"""
        self.metadata["last_modified"] = datetime.now().isoformat()
    
    # ========================================
    # Digital Twin State Sync Methods
    # ========================================
    
    def export_to_json(self, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Export knowledge graph to JSON format for digital twin sync
        Uses NetworkX's built-in JSON serialization
        
        Args:
            include_metadata: Whether to include graph metadata
            
        Returns:
            Complete JSON representation of the graph
        """
        # Use NetworkX's node-link format for clean JSON serialization
        graph_data = json_graph.node_link_data(self.graph, edges="links")
        
        export_data = {
            'graph_data': graph_data,
            'graph_type': self.graph_type.value,
            'graph_id': self.graph_id,
            'external_references': self.external_references,
            'node_3d_positions': {k: list(v) for k, v in self.node_3d_positions.items()},
            'export_timestamp': datetime.now().isoformat()
        }
        
        if include_metadata:
            export_data['metadata'] = self.metadata.copy()
        
        logger.info(f"Exported graph {self.graph_id} to JSON: {len(graph_data['nodes'])} nodes, {len(graph_data['links'])} edges")
        return export_data
    
    def update_from_json(self, json_data: Dict[str, Any], merge_mode: str = "update") -> Dict[str, Any]:
        """
        Update knowledge graph state from JSON data (digital twin sync)
        
        Args:
            json_data: JSON data containing graph updates
            merge_mode: "update" (merge), "replace" (overwrite), "append" (add only)
            
        Returns:
            Dictionary with update statistics
        """
        update_stats = {
            'nodes_added': 0,
            'nodes_updated': 0,
            'edges_added': 0,
            'edges_updated': 0,
            'metadata_updated': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Handle different merge modes
            if merge_mode == "replace":
                # Clear existing graph
                self.graph.clear()
                self.external_references.clear()
                self.node_3d_positions.clear()
                logger.info("Cleared existing graph for replace mode")
            
            # Update graph structure from NetworkX JSON format
            if 'graph_data' in json_data:
                graph_data = json_data['graph_data']
                
                if merge_mode == "replace":
                    # Replace entire graph
                    self.graph = json_graph.node_link_graph(graph_data, edges="links")
                    update_stats['nodes_added'] = len(self.graph.nodes())
                    update_stats['edges_added'] = len(self.graph.edges())
                else:
                    # Merge nodes and edges
                    temp_graph = json_graph.node_link_graph(graph_data, edges="links")
                    
                    # Add/update nodes
                    for node_id, node_attrs in temp_graph.nodes(data=True):
                        if node_id in self.graph:
                            if merge_mode == "update":
                                # Update existing node attributes
                                self.graph.nodes[node_id].update(node_attrs)
                                update_stats['nodes_updated'] += 1
                            # Skip if append mode and node exists
                        else:
                            # Add new node
                            self.graph.add_node(node_id, **node_attrs)
                            update_stats['nodes_added'] += 1
                    
                    # Add/update edges
                    for source, target, edge_attrs in temp_graph.edges(data=True):
                        if self.graph.has_edge(source, target):
                            if merge_mode == "update":
                                # Update existing edge attributes
                                self.graph.edges[source, target].update(edge_attrs)
                                update_stats['edges_updated'] += 1
                            # Skip if append mode and edge exists
                        else:
                            # Add new edge
                            self.graph.add_edge(source, target, **edge_attrs)
                            update_stats['edges_added'] += 1
            
            # Update external references
            if 'external_references' in json_data:
                if merge_mode == "replace":
                    self.external_references = json_data['external_references'].copy()
                else:
                    self.external_references.update(json_data['external_references'])
            
            # Update 3D positions
            if 'node_3d_positions' in json_data:
                positions = json_data['node_3d_positions']
                if merge_mode == "replace":
                    self.node_3d_positions = {k: tuple(v) for k, v in positions.items()}
                else:
                    for node_id, pos in positions.items():
                        if isinstance(pos, list) and len(pos) == 3:
                            self.node_3d_positions[node_id] = tuple(pos)
            
            # Update metadata
            if 'metadata' in json_data:
                if merge_mode == "replace":
                    self.metadata = json_data['metadata'].copy()
                else:
                    self.metadata.update(json_data['metadata'])
                update_stats['metadata_updated'] = True
            
            # Update modified time
            self._update_modified_time()
            
            logger.info(f"Graph {self.graph_id} updated from JSON: {update_stats}")
            
        except Exception as e:
            error_msg = f"Error updating graph from JSON: {str(e)}"
            update_stats['errors'].append(error_msg)
            logger.error(error_msg)
        
        return update_stats
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of current graph state for digital twin monitoring
        
        Returns:
            Dictionary with graph state summary
        """
        return {
            'graph_id': self.graph_id,
            'graph_type': self.graph_type.value,
            'node_count': len(self.graph.nodes()),
            'edge_count': len(self.graph.edges()),
            'external_references_count': len(self.external_references),
            'positioned_nodes_count': len(self.node_3d_positions),
            'last_modified': self.metadata.get('last_modified'),
            'creation_date': self.metadata.get('creation_date'),
            'node_types': self._get_node_type_distribution(),
            'edge_types': self._get_edge_type_distribution()
        }
    
    def _get_node_type_distribution(self) -> Dict[str, int]:
        """Get distribution of node types in the graph"""
        type_counts = {}
        for node_id in self.graph.nodes():
            attrs = self.get_node_attributes(node_id)
            node_type = attrs.get('type', 'unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts
    
    def _get_edge_type_distribution(self) -> Dict[str, int]:
        """Get distribution of edge types in the graph"""
        type_counts = {}
        for source, target, attrs in self.graph.edges(data=True):
            edge_type = attrs.get('relationship', 'unknown')
            type_counts[edge_type] = type_counts.get(edge_type, 0) + 1
        return type_counts
    
    def __str__(self) -> str:
        """String representation of the graph"""
        return f"KnowledgeGraph(id={self.graph_id}, type={self.graph_type.value}, nodes={len(self.graph.nodes())}, edges={len(self.graph.edges())})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__()

if __name__ == "__main__":
    # Import and run demo from separate file
    from knowledge_graph_demo import demo_knowledge_graph
    demo_knowledge_graph()
