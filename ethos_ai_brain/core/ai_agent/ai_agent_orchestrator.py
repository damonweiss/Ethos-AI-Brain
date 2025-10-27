"""
AI Agent Orchestrator - Multi-Agent Coordination System
Handles complex multi-agent orchestration with NetworkX intelligence
"""

import asyncio
import logging
import networkx as nx
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add the Ethos-ZeroMQ path
sys.path.append(r'C:\Users\DamonWeiss\PycharmProjects\Ethos-ZeroMQ')

from ethos_zeromq import ZeroMQEngine

logger = logging.getLogger(__name__)


class AgentType(Enum):
    MAJOR_GENERAL = "major_general"
    STRATEGIC_ADVISOR = "strategic_advisor"
    MISSION_PLANNER = "mission_planner"
    QUALITY_ASSURANCE = "quality_assurance"
    TECHNICAL_ARCHITECT = "technical_architect"
    DOCUMENTATION_SPECIALIST = "documentation_specialist"
    PLATOON_LEADER = "platoon_leader"
    GRUNT = "grunt"


class AgentStatus(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class ManagedAgent:
    """Managed Agent data structure for orchestration with ZMQ server"""
    id: str
    agent_type: AgentType
    status: AgentStatus
    created_at: datetime
    last_activity: datetime
    capabilities: List[str]
    zmq_server: Any = None  # Actual ZMQ server instance
    zmq_port: Optional[int] = None
    current_mission: Optional[str] = None
    performance_metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {
                'tasks_completed': 0,
                'success_rate': 0.0,
                'average_response_time': 0.0,
                'last_error': None
            }

    def update_status(self, new_status: AgentStatus):
        """Update agent status with timestamp"""
        self.status = new_status
        self.last_activity = datetime.now()
        logger.info(f"Agent {self.id} status updated to {new_status.value}")

    def assign_mission(self, mission_id: str):
        """Assign mission to agent"""
        self.current_mission = mission_id
        self.update_status(AgentStatus.BUSY)
        logger.info(f"Agent {self.id} assigned to mission {mission_id}")

    def complete_mission(self, success: bool = True):
        """Mark mission as complete"""
        if self.current_mission:
            self.performance_metrics['tasks_completed'] += 1
            if success:
                # Update success rate
                total_tasks = self.performance_metrics['tasks_completed']
                current_successes = self.performance_metrics['success_rate'] * (total_tasks - 1)
                self.performance_metrics['success_rate'] = (current_successes + 1) / total_tasks

            logger.info(f"Agent {self.id} completed mission {self.current_mission} (Success: {success})")
            self.current_mission = None
            self.update_status(AgentStatus.IDLE)

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'agent_type': self.agent_type.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'capabilities': self.capabilities,
            'current_mission': self.current_mission,
            'performance_metrics': self.performance_metrics
        }


class AIAgentOrchestrator:
    """Orchestrates complex multi-agent systems with NetworkX intelligence"""

    def __init__(self):
        self.agents: Dict[str, ManagedAgent] = {}
        self.agent_counter = 0
        self.max_agents = 100  # Prevent resource exhaustion

        # ZMQ Engine for real agent communication
        self.zmq_engine = ZeroMQEngine()

        # Port allocation for agents
        self.port_ranges = {
            AgentType.MAJOR_GENERAL: (5550, 5550),
            AgentType.STRATEGIC_ADVISOR: (5551, 5559),
            AgentType.MISSION_PLANNER: (5560, 5569),
            AgentType.QUALITY_ASSURANCE: (5570, 5579),
            AgentType.TECHNICAL_ARCHITECT: (5580, 5589),
            AgentType.DOCUMENTATION_SPECIALIST: (5590, 5599),
            AgentType.PLATOON_LEADER: (5600, 5699),
            AgentType.GRUNT: (6000, 6999)
        }
        self.allocated_ports: set = set()

        # NetworkX Intelligence Graphs
        self.agent_dependency_graph = nx.DiGraph()  # Agent spawn dependencies
        self.capability_graph = nx.Graph()  # Agent capabilities network
        self.communication_graph = nx.DiGraph()  # Agent communication patterns
        self.collaboration_graph = nx.Graph()  # Agent collaboration relationships

        # Agent capabilities by type
        self.agent_capabilities = {
            AgentType.MAJOR_GENERAL: ['mission_coordination', 'strategic_oversight', 'agent_management'],
            AgentType.STRATEGIC_ADVISOR: ['risk_assessment', 'resource_allocation', 'strategic_planning'],
            AgentType.MISSION_PLANNER: ['tactical_planning', 'dependency_analysis', 'resource_optimization'],
            AgentType.QUALITY_ASSURANCE: ['deliverable_review', 'process_improvement', 'quality_metrics'],
            AgentType.TECHNICAL_ARCHITECT: ['architectural_decisions', 'technical_standards', 'system_coherence'],
            AgentType.DOCUMENTATION_SPECIALIST: ['knowledge_management', 'decision_records', 'specifications'],
            AgentType.PLATOON_LEADER: ['tactical_execution', 'grunt_management', 'mission_components'],
            AgentType.GRUNT: ['task_execution', 'data_processing', 'specific_operations']
        }

        # Initialize NetworkX graphs with agent relationships
        self._initialize_agent_graphs()

    def _initialize_agent_graphs(self):
        """Initialize NetworkX graphs with agent type relationships"""
        # Agent dependency relationships (who needs to be spawned first)
        dependencies = [
            (AgentType.MAJOR_GENERAL, AgentType.STRATEGIC_ADVISOR),
            (AgentType.MAJOR_GENERAL, AgentType.MISSION_PLANNER),
            (AgentType.STRATEGIC_ADVISOR, AgentType.QUALITY_ASSURANCE),
            (AgentType.MISSION_PLANNER, AgentType.PLATOON_LEADER),
            (AgentType.PLATOON_LEADER, AgentType.GRUNT),
            (AgentType.TECHNICAL_ARCHITECT, AgentType.DOCUMENTATION_SPECIALIST)
        ]

        for prerequisite, dependent in dependencies:
            self.agent_dependency_graph.add_edge(prerequisite.value, dependent.value)

        # Capability network (which agents share capabilities)
        for agent_type, capabilities in self.agent_capabilities.items():
            self.capability_graph.add_node(agent_type.value, capabilities=capabilities)

            # Connect agents with overlapping capabilities
            for other_type, other_capabilities in self.agent_capabilities.items():
                if agent_type != other_type:
                    overlap = set(capabilities) & set(other_capabilities)
                    if overlap:
                        self.capability_graph.add_edge(
                            agent_type.value,
                            other_type.value,
                            shared_capabilities=list(overlap),
                            overlap_score=len(overlap)
                        )

        # Communication patterns (who typically communicates with whom)
        communication_patterns = [
            (AgentType.MAJOR_GENERAL, AgentType.STRATEGIC_ADVISOR, {'frequency': 'high', 'type': 'consultation'}),
            (AgentType.MAJOR_GENERAL, AgentType.MISSION_PLANNER, {'frequency': 'high', 'type': 'coordination'}),
            (AgentType.STRATEGIC_ADVISOR, AgentType.QUALITY_ASSURANCE, {'frequency': 'medium', 'type': 'review'}),
            (AgentType.MISSION_PLANNER, AgentType.PLATOON_LEADER, {'frequency': 'high', 'type': 'delegation'}),
            (AgentType.PLATOON_LEADER, AgentType.GRUNT, {'frequency': 'very_high', 'type': 'command'}),
            (AgentType.TECHNICAL_ARCHITECT, AgentType.DOCUMENTATION_SPECIALIST,
             {'frequency': 'medium', 'type': 'collaboration'})
        ]

        for source, target, attributes in communication_patterns:
            self.communication_graph.add_edge(source.value, target.value, **attributes)

        logger.info("Agent NetworkX graphs initialized with relationships")

    def spawn_agent(self, agent_type: AgentType, capabilities: Optional[List[str]] = None) -> str:
        """Spawn a new AI agent with actual ZMQ server"""
        if len(self.agents) >= self.max_agents:
            raise RuntimeError(f"Maximum agent limit ({self.max_agents}) reached")

        # Generate unique agent ID
        self.agent_counter += 1
        agent_id = f"{agent_type.value}_{self.agent_counter:04d}"

        # Get default capabilities for agent type
        default_capabilities = self.agent_capabilities.get(agent_type, [])
        agent_capabilities = capabilities or default_capabilities

        # Allocate port for agent
        agent_port = self._allocate_port(agent_type)

        # Create agent
        agent = ManagedAgent(
            id=agent_id,
            agent_type=agent_type,
            status=AgentStatus.INITIALIZING,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            capabilities=agent_capabilities,
            zmq_port=agent_port
        )

        # Create actual ZMQ server for agent
        try:
            agent.zmq_server = self.zmq_engine.create_and_start_server(
                'reqrep',  # Default pattern for agents
                agent_id
            )
            logger.info(f"Created ZMQ server for {agent_id}")
        except Exception as e:
            logger.error(f"Failed to create ZMQ server for {agent_id}: {e}")
            self._deallocate_port(agent_port)
            raise RuntimeError(f"Failed to spawn agent {agent_id}: {e}")

        # Register agent
        self.agents[agent_id] = agent

        # Mark as active
        agent.update_status(AgentStatus.ACTIVE)

        logger.info(f"Spawned {agent_type.value} agent: {agent_id} with ZMQ server on port {agent_port}")
        return agent_id

    def _allocate_port(self, agent_type: AgentType) -> int:
        """Allocate a port for agent type"""
        start_port, end_port = self.port_ranges[agent_type]

        for port in range(start_port, end_port + 1):
            if port not in self.allocated_ports:
                self.allocated_ports.add(port)
                return port

        raise RuntimeError(f"No available ports for agent type {agent_type.value}")

    def _deallocate_port(self, port: int):
        """Deallocate a port"""
        self.allocated_ports.discard(port)

    def terminate_agent(self, agent_id: str) -> bool:
        """Terminate an AI agent and close ZMQ server"""
        if agent_id not in self.agents:
            logger.warning(f"Attempted to terminate non-existent agent: {agent_id}")
            return False

        agent = self.agents[agent_id]
        agent.update_status(AgentStatus.TERMINATED)

        # Close ZMQ server
        if agent.zmq_server:
            try:
                # Your ZMQ library should have a close/stop method
                if hasattr(agent.zmq_server, 'close'):
                    agent.zmq_server.close()
                elif hasattr(agent.zmq_server, 'stop'):
                    agent.zmq_server.stop()
                logger.info(f"Closed ZMQ server for agent {agent_id}")
            except Exception as e:
                logger.error(f"Error closing ZMQ server for {agent_id}: {e}")

        # Deallocate port
        if agent.zmq_port:
            self._deallocate_port(agent.zmq_port)

        # Remove from active agents
        del self.agents[agent_id]

        logger.info(f"Terminated agent: {agent_id}")
        return True

    def get_agent(self, agent_id: str) -> Optional[ManagedAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)

    def get_agents_by_type(self, agent_type: AgentType) -> List[ManagedAgent]:
        """Get all agents of specific type"""
        return [agent for agent in self.agents.values() if agent.agent_type == agent_type]

    def get_available_agents(self, agent_type: Optional[AgentType] = None) -> List[ManagedAgent]:
        """Get all available (idle) agents, optionally filtered by type"""
        available = [agent for agent in self.agents.values() if agent.status == AgentStatus.IDLE]

        if agent_type:
            available = [agent for agent in available if agent.agent_type == agent_type]

        return available

    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        stats = {
            'total_agents': len(self.agents),
            'by_type': {},
            'by_status': {},
            'performance_summary': {
                'total_tasks_completed': 0,
                'average_success_rate': 0.0
            }
        }

        # Count by type and status
        for agent in self.agents.values():
            # By type
            type_name = agent.agent_type.value
            stats['by_type'][type_name] = stats['by_type'].get(type_name, 0) + 1

            # By status
            status_name = agent.status.value
            stats['by_status'][status_name] = stats['by_status'].get(status_name, 0) + 1

            # Performance
            stats['performance_summary']['total_tasks_completed'] += agent.performance_metrics['tasks_completed']

        # Calculate average success rate
        if self.agents:
            total_success_rate = sum(agent.performance_metrics['success_rate'] for agent in self.agents.values())
            stats['performance_summary']['average_success_rate'] = total_success_rate / len(self.agents)

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all agents"""
        health_report = {
            'healthy_agents': 0,
            'unhealthy_agents': 0,
            'agent_details': []
        }

        for agent in self.agents.values():
            is_healthy = agent.status in [AgentStatus.ACTIVE, AgentStatus.IDLE, AgentStatus.BUSY]

            if is_healthy:
                health_report['healthy_agents'] += 1
            else:
                health_report['unhealthy_agents'] += 1

            health_report['agent_details'].append({
                'id': agent.id,
                'type': agent.agent_type.value,
                'status': agent.status.value,
                'healthy': is_healthy,
                'last_activity': agent.last_activity.isoformat()
            })

        return health_report

    # NetworkX Intelligence Methods

    def get_optimal_spawn_order(self, required_agent_types: List[AgentType]) -> List[AgentType]:
        """Use NetworkX to determine optimal agent spawning order"""
        try:
            # Convert to string values for graph operations
            required_types = [agent_type.value for agent_type in required_agent_types]

            # Get subgraph of required agents and their dependencies
            subgraph = self.agent_dependency_graph.subgraph(required_types)

            # Use topological sort to get optimal spawn order
            spawn_order_strings = list(nx.topological_sort(subgraph))

            # Convert back to AgentType enums
            spawn_order = []
            for agent_string in spawn_order_strings:
                for agent_type in AgentType:
                    if agent_type.value == agent_string:
                        spawn_order.append(agent_type)
                        break

            logger.info(f"Optimal spawn order determined: {[a.value for a in spawn_order]}")
            return spawn_order

        except Exception as e:
            logger.error(f"Failed to determine spawn order: {e}")
            return required_agent_types  # Fallback to original order

    def find_agents_by_capability(self, required_capabilities: List[str]) -> List[AgentType]:
        """Use NetworkX to find agents with required capabilities"""
        matching_agents = []

        for node in self.capability_graph.nodes():
            node_capabilities = self.capability_graph.nodes[node].get('capabilities', [])

            # Check if agent has all required capabilities
            if all(cap in node_capabilities for cap in required_capabilities):
                # Convert string back to AgentType
                for agent_type in AgentType:
                    if agent_type.value == node:
                        matching_agents.append(agent_type)
                        break

        logger.info(f"Found {len(matching_agents)} agents with capabilities {required_capabilities}")
        return matching_agents

    def get_collaboration_recommendations(self, agent_type: AgentType) -> List[Dict[str, Any]]:
        """Use NetworkX to recommend collaboration partners"""
        agent_string = agent_type.value
        recommendations = []

        if agent_string in self.capability_graph:
            # Find agents with shared capabilities
            neighbors = list(self.capability_graph.neighbors(agent_string))

            for neighbor in neighbors:
                edge_data = self.capability_graph.get_edge_data(agent_string, neighbor)
                shared_caps = edge_data.get('shared_capabilities', [])
                overlap_score = edge_data.get('overlap_score', 0)

                recommendations.append({
                    'agent_type': neighbor,
                    'shared_capabilities': shared_caps,
                    'collaboration_strength': overlap_score,
                    'recommendation_reason': f"Shares {len(shared_caps)} capabilities"
                })

        # Sort by collaboration strength
        recommendations.sort(key=lambda x: x['collaboration_strength'], reverse=True)

        logger.info(f"Generated {len(recommendations)} collaboration recommendations for {agent_type.value}")
        return recommendations

    def get_communication_path(self, source_agent: AgentType, target_agent: AgentType) -> List[str]:
        """Use NetworkX to find optimal communication path between agents"""
        try:
            source_string = source_agent.value
            target_string = target_agent.value

            if source_string in self.communication_graph and target_string in self.communication_graph:
                # Find shortest path for communication
                path = nx.shortest_path(self.communication_graph, source_string, target_string)
                logger.info(f"Communication path from {source_string} to {target_string}: {' -> '.join(path)}")
                return path
            else:
                logger.warning(f"No communication path found between {source_string} and {target_string}")
                return [source_string, target_string]  # Direct communication fallback

        except nx.NetworkXNoPath:
            logger.warning(f"No communication path exists between {source_agent.value} and {target_agent.value}")
            return [source_agent.value, target_agent.value]  # Direct communication fallback
        except Exception as e:
            logger.error(f"Error finding communication path: {e}")
            return [source_agent.value, target_agent.value]  # Direct communication fallback

    def analyze_agent_network_metrics(self) -> Dict[str, Any]:
        """Use NetworkX to analyze agent network properties"""
        metrics = {
            'dependency_analysis': {},
            'capability_analysis': {},
            'communication_analysis': {}
        }

        # Dependency graph metrics
        if len(self.agent_dependency_graph) > 0:
            metrics['dependency_analysis'] = {
                'total_dependencies': len(self.agent_dependency_graph.edges()),
                'dependency_depth': len(
                    nx.dag_longest_path(self.agent_dependency_graph)) if nx.is_directed_acyclic_graph(
                    self.agent_dependency_graph) else 0,
                'independent_agents': [node for node in self.agent_dependency_graph.nodes() if
                                       self.agent_dependency_graph.in_degree(node) == 0],
                'most_dependent': max(self.agent_dependency_graph.nodes(),
                                      key=lambda x: self.agent_dependency_graph.in_degree(x), default=None)
            }

        # Capability graph metrics
        if len(self.capability_graph) > 0:
            metrics['capability_analysis'] = {
                'total_capability_connections': len(self.capability_graph.edges()),
                'most_versatile_agent': max(self.capability_graph.nodes(),
                                            key=lambda x: self.capability_graph.degree(x), default=None),
                'capability_clusters': len(list(nx.connected_components(self.capability_graph))),
                'average_capability_overlap': sum(
                    data.get('overlap_score', 0) for _, _, data in self.capability_graph.edges(data=True)) / max(
                    len(self.capability_graph.edges()), 1)
            }

        # Communication graph metrics
        if len(self.communication_graph) > 0:
            metrics['communication_analysis'] = {
                'total_communication_paths': len(self.communication_graph.edges()),
                'communication_hub': max(self.communication_graph.nodes(),
                                         key=lambda x: self.communication_graph.degree(x), default=None),
                'average_path_length': nx.average_shortest_path_length(self.communication_graph) if nx.is_connected(
                    self.communication_graph.to_undirected()) else 0,
                'communication_efficiency': nx.global_efficiency(self.communication_graph)
            }

        logger.info("Agent network metrics analyzed using NetworkX")
        return metrics
