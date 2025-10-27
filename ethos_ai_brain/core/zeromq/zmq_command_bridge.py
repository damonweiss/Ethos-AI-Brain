"""
ZMQ Command Bridge for AI Command System
Integrates with Ethos-ZeroMQ library for military command communication
"""

import asyncio
import logging
import networkx as nx
import sys
import os
from typing import Dict, List, Any, Optional
import json

# Add the Ethos-ZeroMQ path
sys.path.append(r'C:\Users\DamonWeiss\PycharmProjects\Ethos-ZeroMQ')

from ethos_zeromq import ZeroMQEngine
from ethos_zeromq.ReqRepServer import ReqRepServer
from ethos_zeromq.PubSubServer import PubSubServer
from ethos_zeromq.RouterDealerServer import RouterDealerServer

logger = logging.getLogger(__name__)


class ZMQCommandBridge:
    """
    Bridge between dtOS and ZMQ for AI command communication
    Implements biological neural patterns for AI coordination
    """

    def __init__(self):
        self.zmq_engine = ZeroMQEngine()
        self.command_servers: Dict[str, Any] = {}
        self.biological_patterns: Dict[str, str] = {}

        # NetworkX Intelligence Graphs
        self.network_topology = nx.Graph()  # Physical network topology
        self.message_routing_graph = nx.DiGraph()  # Message routing paths
        self.pattern_optimization_graph = nx.DiGraph()  # Pattern selection optimization
        self.performance_graph = nx.DiGraph()  # Performance metrics network

        # Initialize command network and graphs
        self._initialize_command_network()
        self._initialize_network_graphs()

        logger.info("ZMQ Command Bridge initialized with Ethos-ZeroMQ")

    def _initialize_command_network(self):
        """Initialize ZMQ command network with biological patterns"""
        try:
            # Major General Command Hub (Central Command)
            self.command_servers['major_general'] = self.zmq_engine.create_and_start_server(
                'reqrep', 'major_general_hub'
            )

            # Panel of Experts Communication (for Sprint B)
            expert_names = [
                'strategic_advisor',
                'mission_planner',
                'quality_assurance',
                'technical_architect',
                'documentation_specialist'
            ]

            for expert_name in expert_names:
                self.command_servers[expert_name] = self.zmq_engine.create_and_start_server(
                    'reqrep', f'{expert_name}_server'
                )

            # Biological Pattern Servers
            self._setup_biological_patterns()

            logger.info("ZMQ command network initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ZMQ command network: {e}")
            # Continue with mock servers for development

    def _setup_biological_patterns(self):
        """Setup biological neural pattern servers (only supported patterns)"""
        try:
            # Only create patterns that your ZMQ library supports
            supported_patterns = [
                ('reflex_arc', 'reqrep', 'reflex_arc_server'),
                ('spreading_activation', 'pubsub', 'spreading_activation_server')
            ]

            for pattern_name, server_type, server_name in supported_patterns:
                try:
                    server = self.zmq_engine.create_and_start_server(server_type, server_name)
                    if server:
                        self.command_servers[pattern_name] = server
                        self.biological_patterns[pattern_name] = server_type
                        logger.info(f"Created {pattern_name} server ({server_type})")
                    else:
                        logger.warning(f"Failed to create {pattern_name} server")
                except Exception as e:
                    logger.warning(f"Could not create {pattern_name} server: {e}")

            # For unsupported patterns, just register the pattern type for future use
            unsupported_patterns = {
                'hierarchical_processing': 'routerdealer',
                'parallel_processing': 'pushpull',
                'integration_centers': 'scattergather'
            }

            for pattern_name, pattern_type in unsupported_patterns.items():
                self.biological_patterns[pattern_name] = pattern_type
                logger.info(f"Registered {pattern_name} pattern ({pattern_type}) - server creation deferred")

            logger.info(f"Biological patterns initialized: {len(self.biological_patterns)} patterns")

        except Exception as e:
            logger.error(f"Failed to setup biological patterns: {e}")

    def _initialize_network_graphs(self):
        """Initialize NetworkX graphs for network intelligence"""
        try:
            # Network topology - physical connections between servers
            topology_connections = [
                ('major_general', 'strategic_advisor', {'latency': 1, 'bandwidth': 1000, 'reliability': 0.99}),
                ('major_general', 'mission_planner', {'latency': 1, 'bandwidth': 1000, 'reliability': 0.99}),
                ('major_general', 'quality_assurance', {'latency': 2, 'bandwidth': 500, 'reliability': 0.95}),
                ('major_general', 'technical_architect', {'latency': 2, 'bandwidth': 500, 'reliability': 0.95}),
                ('major_general', 'documentation_specialist', {'latency': 3, 'bandwidth': 200, 'reliability': 0.90}),
                ('strategic_advisor', 'mission_planner', {'latency': 1, 'bandwidth': 800, 'reliability': 0.98}),
                ('mission_planner', 'quality_assurance', {'latency': 1, 'bandwidth': 600, 'reliability': 0.97}),
                ('technical_architect', 'documentation_specialist',
                 {'latency': 1, 'bandwidth': 400, 'reliability': 0.95})
            ]

            for source, target, attributes in topology_connections:
                self.network_topology.add_edge(source, target, **attributes)

            # Message routing graph - optimal paths for different message types
            routing_paths = [
                ('major_general', 'strategic_advisor',
                 {'message_types': ['consultation', 'risk_assessment'], 'priority': 'high'}),
                ('major_general', 'mission_planner',
                 {'message_types': ['coordination', 'planning'], 'priority': 'high'}),
                ('strategic_advisor', 'quality_assurance',
                 {'message_types': ['review', 'validation'], 'priority': 'medium'}),
                ('mission_planner', 'platoon_leader', {'message_types': ['delegation', 'command'], 'priority': 'high'}),
                ('quality_assurance', 'documentation_specialist',
                 {'message_types': ['documentation', 'records'], 'priority': 'low'})
            ]

            for source, target, attributes in routing_paths:
                self.message_routing_graph.add_edge(source, target, **attributes)

            # Pattern optimization graph - which patterns work best for different scenarios
            pattern_relationships = [
                ('reflex_arc', 'urgent_response', {'effectiveness': 0.95, 'latency': 'very_low'}),
                ('spreading_activation', 'broadcast_notification', {'effectiveness': 0.90, 'latency': 'low'}),
                ('hierarchical_processing', 'expert_consultation', {'effectiveness': 0.85, 'latency': 'medium'}),
                ('parallel_processing', 'concurrent_execution', {'effectiveness': 0.80, 'latency': 'medium'}),
                ('integration_centers', 'consensus_building', {'effectiveness': 0.75, 'latency': 'high'})
            ]

            for pattern, scenario, attributes in pattern_relationships:
                self.pattern_optimization_graph.add_edge(pattern, scenario, **attributes)

            logger.info("NetworkX graphs initialized for ZMQ network intelligence")

        except Exception as e:
            logger.error(f"Failed to initialize network graphs: {e}")

    async def send_command(self, target: str, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to target agent via appropriate ZMQ pattern"""
        try:
            if target not in self.command_servers:
                logger.warning(f"Target {target} not found, using major_general hub")
                target = 'major_general'

            server = self.command_servers[target]

            # Serialize command
            command_json = json.dumps(command)

            # Send via ZMQ using real message routing
            if hasattr(server, 'send_direct_message'):
                response = server.send_direct_message(command)
            elif hasattr(server, 'route_message'):
                response = server.route_message(command)
            else:
                response = {"status": "no_handler", "target": target, "command": command}

            logger.info(f"Command sent to {target}: {command.get('type', 'unknown')}")
            return response

        except Exception as e:
            logger.error(f"Failed to send command to {target}: {e}")
            return {"status": "error", "message": str(e)}

    async def broadcast_status(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """Broadcast status update using spreading activation pattern"""
        try:
            server = self.command_servers.get('spreading_activation')
            if not server:
                logger.warning("Spreading activation server not available")
                return {"status": "unavailable"}

            status_json = json.dumps(status)

            # Broadcast via ZMQ using real message routing
            if hasattr(server, 'send_direct_message'):
                response = server.send_direct_message(status)
            elif hasattr(server, 'route_message'):
                response = server.route_message(status)
            else:
                response = {"status": "no_handler", "recipients": ["all_agents"]}

            logger.info(f"Status broadcast: {status.get('type', 'unknown')}")
            return response

        except Exception as e:
            logger.error(f"Failed to broadcast status: {e}")
            return {"status": "error", "message": str(e)}

    async def route_to_expert(self, expert_type: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route request to specific expert using hierarchical processing"""
        try:
            # Map expert types to server names
            expert_mapping = {
                'strategic': 'strategic_advisor',
                'planning': 'mission_planner',
                'quality': 'quality_assurance',
                'technical': 'technical_architect',
                'documentation': 'documentation_specialist'
            }

            target_expert = expert_mapping.get(expert_type, 'strategic_advisor')
            server = self.command_servers.get(target_expert)

            if not server:
                logger.warning(f"Expert {target_expert} not available")
                return {"status": "expert_unavailable", "expert": target_expert}

            request_json = json.dumps(request)

            # Route to expert using real ZMQ message routing
            if hasattr(server, 'send_direct_message'):
                response = server.send_direct_message(request)
            elif hasattr(server, 'route_message'):
                response = server.route_message(request)
            else:
                response = {
                    "status": "no_handler",
                    "expert": target_expert,
                    "message": "Expert server has no message handling capability"
                }

            logger.info(f"Request routed to expert {target_expert}")
            return response

        except Exception as e:
            logger.error(f"Failed to route to expert {expert_type}: {e}")
            return {"status": "error", "message": str(e)}

    def select_biological_pattern(self, mission_characteristics: Dict[str, Any]) -> str:
        """
        Select optimal biological pattern based on mission characteristics
        This will be enhanced in Sprint E with Neural Network selection
        """
        # Simple rule-based selection for now
        # Sprint E will add sophisticated NN-based pattern selection

        complexity = mission_characteristics.get('complexity', 'normal')
        urgency = mission_characteristics.get('urgency', 'normal')
        collaboration_needed = mission_characteristics.get('collaboration_needed', False)
        parallel_opportunities = mission_characteristics.get('parallel_opportunities', 0)

        # Reflex Arc: Simple, urgent tasks
        if urgency == 'critical' and complexity == 'low':
            return 'reflex_arc'

        # Spreading Activation: Broadcast information
        elif mission_characteristics.get('broadcast_needed', False):
            return 'spreading_activation'

        # Hierarchical Processing: Expert consultation needed
        elif collaboration_needed or complexity == 'high':
            return 'hierarchical_processing'

        # Parallel Processing: Multiple parallel tasks
        elif parallel_opportunities > 1:
            return 'parallel_processing'

        # Integration Centers: Multiple inputs need synthesis
        elif mission_characteristics.get('synthesis_needed', False):
            return 'integration_centers'

        # Default to reflex arc for simple tasks
        else:
            return 'reflex_arc'

    async def execute_biological_pattern(self, pattern: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specific biological pattern"""
        try:
            server = self.command_servers.get(pattern)
            if not server:
                logger.warning(f"Biological pattern {pattern} not available")
                return {"status": "pattern_unavailable", "pattern": pattern}

            data_json = json.dumps(data)

            # Execute using real ZMQ message routing (pattern-agnostic)
            if hasattr(server, 'send_direct_message'):
                response = server.send_direct_message(data)
            elif hasattr(server, 'route_message'):
                response = server.route_message(data)
            else:
                response = {"status": "no_handler", "pattern": pattern}

            logger.info(f"Executed biological pattern: {pattern}")
            return response

        except Exception as e:
            logger.error(f"Failed to execute biological pattern {pattern}: {e}")
            return {"status": "error", "message": str(e)}

    def get_network_status(self) -> Dict[str, Any]:
        """Get status of ZMQ command network"""
        return {
            'zmq_available': True,
            'active_servers': list(self.command_servers.keys()),
            'biological_patterns': self.biological_patterns,
            'network_health': 'operational'
        }

    async def shutdown(self):
        """Graceful shutdown of ZMQ command bridge"""
        logger.info("Shutting down ZMQ command bridge...")

        # Close all servers
        for server_name, server in self.command_servers.items():
            try:
                if hasattr(server, 'close'):
                    await server.close()
                logger.info(f"Closed server: {server_name}")
            except Exception as e:
                logger.error(f"Error closing server {server_name}: {e}")

        self.command_servers.clear()
        logger.info("ZMQ command bridge shutdown complete")

    # NetworkX Intelligence Methods

    def get_optimal_routing_path(self, source: str, target: str, message_type: str = None) -> List[str]:
        """Use NetworkX to find optimal message routing path"""
        try:
            # First try direct routing in message routing graph
            if source in self.message_routing_graph and target in self.message_routing_graph:
                try:
                    path = nx.shortest_path(self.message_routing_graph, source, target)
                    logger.info(f"Direct routing path found: {' -> '.join(path)}")
                    return path
                except nx.NetworkXNoPath:
                    pass

            # Fallback to network topology for physical routing
            if source in self.network_topology and target in self.network_topology:
                try:
                    # Use weighted shortest path based on latency
                    path = nx.shortest_path(self.network_topology, source, target, weight='latency')
                    logger.info(f"Network topology path found: {' -> '.join(path)}")
                    return path
                except nx.NetworkXNoPath:
                    pass

            # Final fallback to direct connection
            logger.warning(f"No optimal path found, using direct connection: {source} -> {target}")
            return [source, target]

        except Exception as e:
            logger.error(f"Error finding routing path: {e}")
            return [source, target]

    def select_optimal_biological_pattern(self, mission_characteristics: Dict[str, Any]) -> str:
        """Use NetworkX to select optimal biological pattern based on mission characteristics"""
        try:
            # Determine scenario type from characteristics
            scenario_type = self._classify_mission_scenario(mission_characteristics)

            # Find patterns connected to this scenario in optimization graph
            optimal_patterns = []
            for pattern in self.pattern_optimization_graph.nodes():
                if self.pattern_optimization_graph.has_edge(pattern, scenario_type):
                    edge_data = self.pattern_optimization_graph.get_edge_data(pattern, scenario_type)
                    effectiveness = edge_data.get('effectiveness', 0.5)
                    latency = edge_data.get('latency', 'medium')

                    # Score based on effectiveness and latency requirements
                    score = effectiveness
                    if mission_characteristics.get('urgency') == 'critical' and latency in ['very_low', 'low']:
                        score += 0.2
                    elif mission_characteristics.get('urgency') == 'low' and latency == 'high':
                        score += 0.1

                    optimal_patterns.append((pattern, score))

            if optimal_patterns:
                # Select pattern with highest score
                best_pattern = max(optimal_patterns, key=lambda x: x[1])[0]
                logger.info(f"NetworkX selected optimal pattern: {best_pattern} for scenario: {scenario_type}")
                return best_pattern
            else:
                # Fallback to rule-based selection
                return self.select_biological_pattern(mission_characteristics)

        except Exception as e:
            logger.error(f"Error in NetworkX pattern selection: {e}")
            return self.select_biological_pattern(mission_characteristics)

    def _classify_mission_scenario(self, characteristics: Dict[str, Any]) -> str:
        """Classify mission characteristics into scenario types"""
        urgency = characteristics.get('urgency', 'normal')
        complexity = characteristics.get('complexity', 'normal')
        collaboration_needed = characteristics.get('collaboration_needed', False)
        broadcast_needed = characteristics.get('broadcast_needed', False)
        parallel_opportunities = characteristics.get('parallel_opportunities', 0)
        synthesis_needed = characteristics.get('synthesis_needed', False)

        if urgency == 'critical':
            return 'urgent_response'
        elif broadcast_needed:
            return 'broadcast_notification'
        elif collaboration_needed or complexity == 'high':
            return 'expert_consultation'
        elif parallel_opportunities > 1:
            return 'concurrent_execution'
        elif synthesis_needed:
            return 'consensus_building'
        else:
            return 'urgent_response'  # Default fallback

    def analyze_network_performance(self) -> Dict[str, Any]:
        """Use NetworkX to analyze network performance metrics"""
        performance_metrics = {
            'topology_analysis': {},
            'routing_analysis': {},
            'pattern_analysis': {}
        }

        # Network topology analysis
        if len(self.network_topology) > 0:
            # Calculate network efficiency and reliability
            total_reliability = sum(
                data.get('reliability', 0.5) for _, _, data in self.network_topology.edges(data=True))
            avg_reliability = total_reliability / max(len(self.network_topology.edges()), 1)

            total_latency = sum(data.get('latency', 10) for _, _, data in self.network_topology.edges(data=True))
            avg_latency = total_latency / max(len(self.network_topology.edges()), 1)

            performance_metrics['topology_analysis'] = {
                'total_connections': len(self.network_topology.edges()),
                'network_diameter': nx.diameter(self.network_topology) if nx.is_connected(self.network_topology) else 0,
                'average_reliability': avg_reliability,
                'average_latency': avg_latency,
                'network_efficiency': nx.global_efficiency(self.network_topology),
                'most_connected_node': max(self.network_topology.nodes(), key=lambda x: self.network_topology.degree(x),
                                           default=None)
            }

        # Message routing analysis
        if len(self.message_routing_graph) > 0:
            performance_metrics['routing_analysis'] = {
                'total_routing_paths': len(self.message_routing_graph.edges()),
                'routing_efficiency': nx.global_efficiency(self.message_routing_graph),
                'central_routing_hub': max(self.message_routing_graph.nodes(),
                                           key=lambda x: self.message_routing_graph.degree(x), default=None),
                'average_path_length': nx.average_shortest_path_length(
                    self.message_routing_graph) if nx.is_strongly_connected(self.message_routing_graph) else 0
            }

        # Pattern optimization analysis
        if len(self.pattern_optimization_graph) > 0:
            # Calculate pattern effectiveness scores
            pattern_scores = {}
            for pattern in self.biological_patterns.keys():
                if pattern in self.pattern_optimization_graph:
                    successors = list(self.pattern_optimization_graph.successors(pattern))
                    if successors:
                        total_effectiveness = sum(
                            self.pattern_optimization_graph.get_edge_data(pattern, successor).get('effectiveness', 0.5)
                            for successor in successors
                        )
                        pattern_scores[pattern] = total_effectiveness / len(successors)

            performance_metrics['pattern_analysis'] = {
                'available_patterns': len(self.biological_patterns),
                'pattern_effectiveness_scores': pattern_scores,
                'most_effective_pattern': max(pattern_scores.items(), key=lambda x: x[1])[
                    0] if pattern_scores else None,
                'pattern_coverage': len(self.pattern_optimization_graph.edges())
            }

        logger.info("Network performance analysis completed using NetworkX")
        return performance_metrics

    def optimize_network_topology(self) -> Dict[str, Any]:
        """Use NetworkX to suggest network topology optimizations"""
        optimizations = {
            'bottlenecks': [],
            'redundancy_suggestions': [],
            'performance_improvements': []
        }

        try:
            if len(self.network_topology) > 0:
                # Identify bottlenecks (nodes with high betweenness centrality)
                betweenness = nx.betweenness_centrality(self.network_topology)
                bottlenecks = [(node, score) for node, score in betweenness.items() if score > 0.3]
                optimizations['bottlenecks'] = bottlenecks

                # Suggest redundancy for critical paths
                for node, score in bottlenecks:
                    neighbors = list(self.network_topology.neighbors(node))
                    if len(neighbors) < 3:  # Low redundancy
                        optimizations['redundancy_suggestions'].append({
                            'node': node,
                            'current_connections': len(neighbors),
                            'suggestion': 'Add backup connections to reduce bottleneck risk'
                        })

                # Performance improvement suggestions
                for source, target, data in self.network_topology.edges(data=True):
                    latency = data.get('latency', 10)
                    reliability = data.get('reliability', 0.5)

                    if latency > 5:
                        optimizations['performance_improvements'].append({
                            'connection': f"{source} -> {target}",
                            'issue': 'High latency',
                            'current_latency': latency,
                            'suggestion': 'Consider upgrading connection or adding intermediate nodes'
                        })

                    if reliability < 0.9:
                        optimizations['performance_improvements'].append({
                            'connection': f"{source} -> {target}",
                            'issue': 'Low reliability',
                            'current_reliability': reliability,
                            'suggestion': 'Add redundant paths or improve connection quality'
                        })

            logger.info("Network topology optimization analysis completed")
            return optimizations

        except Exception as e:
            logger.error(f"Error in network optimization analysis: {e}")
            return optimizations
