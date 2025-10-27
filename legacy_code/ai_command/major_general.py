"""
Major General - Central AI Command Orchestrator
Integrates with existing meta-reasoning engine and NetworkX planning
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import networkx as nx

try:
    from meta_reasoning_engine import (
        MetaReasoningEngine, 
        ReasoningContext, 
        ReasoningStep,
        ConfidenceLevel,
        CollaborationMode
    )
except ImportError as e:
    logging.error(f"Failed to import meta-reasoning engine: {e}")
    # Create mock classes for development
    class MetaReasoningEngine:
        def __init__(self): pass
        async def reason(self, goal, context): return {"analysis": "mock"}
    
    class ReasoningContext:
        def __init__(self, goal, **kwargs): 
            self.goal = goal
            self.constraints = kwargs.get('constraints', {})

from mission_parser import Mission, MissionParser, MissionStatus, MissionPriority
from ai_agent_manager import AIAgentManager, AgentType, AgentStatus

logger = logging.getLogger(__name__)

class MajorGeneral:
    """
    Central AI Command Orchestrator
    Integrates NetworkX planning with military command structure
    """
    
    def __init__(self):
        # Core reasoning and planning
        self.meta_reasoning = MetaReasoningEngine()
        self.mission_parser = MissionParser()
        self.agent_manager = AIAgentManager()
        
        # Mission management
        self.active_missions: Dict[str, Mission] = {}
        self.mission_execution_graphs: Dict[str, nx.DiGraph] = {}
        
        # Command network (will integrate ZMQ in next phase)
        self.command_network = None
        self.panel_of_experts: Dict[str, str] = {}  # Will be populated in Sprint B
        
        # Performance tracking
        self.command_metrics = {
            'missions_completed': 0,
            'missions_failed': 0,
            'average_mission_time': 0.0,
            'agent_utilization': 0.0
        }
        
        # Spawn the Major General agent
        self.agent_id = self.agent_manager.spawn_agent(AgentType.MAJOR_GENERAL)
        
        logger.info("Major General initialized and ready for command")
    
    async def receive_mission(self, objective: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Receive and process new mission objective
        Uses NetworkX for mission planning and dependency analysis
        """
        try:
            # Parse mission using mission parser
            mission = self.mission_parser.parse_mission(objective, context)
            
            # Store mission
            self.active_missions[mission.id] = mission
            
            # Update mission status
            mission.update_status(MissionStatus.ANALYZING)
            
            # Create reasoning context for meta-reasoning engine
            reasoning_context = ReasoningContext(
                goal=objective,
                constraints=context.get('constraints', {}) if context else {},
                user_preferences=context.get('preferences', {}) if context else {}
            )
            
            # Use meta-reasoning engine for initial analysis
            logger.info(f"Major General analyzing mission: {mission.id}")
            analysis_result = await self.meta_reasoning.reason(objective, reasoning_context)
            
            # Create mission execution graph using NetworkX
            execution_graph = await self._create_mission_execution_graph(mission, analysis_result)
            self.mission_execution_graphs[mission.id] = execution_graph
            
            # Update mission with analysis results
            mission.results['initial_analysis'] = analysis_result
            mission.results['execution_graph_nodes'] = list(execution_graph.nodes())
            mission.results['execution_graph_edges'] = list(execution_graph.edges())
            
            # Move to planning phase
            mission.update_status(MissionStatus.PLANNING)
            
            # Log mission reception
            logger.info(f"Mission {mission.id} received and analyzed. Priority: {mission.priority.value}")
            logger.info(f"Execution graph created with {len(execution_graph.nodes())} steps")
            
            # In Sprint B, this will consult Panel of Experts
            # For now, proceed with basic execution planning
            await self._plan_mission_execution(mission)
            
            return mission.id
            
        except Exception as e:
            logger.error(f"Failed to receive mission: {e}")
            if 'mission' in locals():
                mission.update_status(MissionStatus.FAILED)
                mission.results['error'] = str(e)
            raise
    
    async def _create_mission_execution_graph(self, mission: Mission, analysis: Dict[str, Any]) -> nx.DiGraph:
        """
        Create NetworkX execution graph for mission
        Defines dependencies and execution order
        """
        graph = nx.DiGraph()
        
        # Extract mission components from analysis
        # This is a simplified version - Sprint B will enhance with expert input
        mission_components = self._extract_mission_components(mission.objective, analysis)
        
        # Add nodes for each component
        for i, component in enumerate(mission_components):
            step_id = f"step_{i+1}"
            graph.add_node(step_id, 
                          component=component,
                          estimated_time=component.get('estimated_time', 1.0),
                          required_capabilities=component.get('capabilities', []),
                          priority=component.get('priority', 'normal'))
        
        # Add dependencies between steps
        self._add_step_dependencies(graph, mission_components)
        
        # Validate graph (ensure it's a DAG)
        if not nx.is_directed_acyclic_graph(graph):
            logger.warning(f"Mission {mission.id} execution graph contains cycles, attempting to resolve")
            graph = self._resolve_graph_cycles(graph)
        
        return graph
    
    def _extract_mission_components(self, objective: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract mission components from objective and analysis
        This will be enhanced in Sprint B with expert input
        """
        # Basic component extraction based on objective keywords
        components = []
        
        objective_lower = objective.lower()
        
        # Analysis component (always first)
        components.append({
            'name': 'mission_analysis',
            'description': 'Analyze mission requirements and constraints',
            'type': 'analysis',
            'estimated_time': 2.0,
            'capabilities': ['analysis', 'planning'],
            'priority': 'high'
        })
        
        # Add components based on objective keywords
        if any(word in objective_lower for word in ['create', 'build', 'develop', 'implement']):
            components.append({
                'name': 'design_planning',
                'description': 'Create detailed design and implementation plan',
                'type': 'planning',
                'estimated_time': 3.0,
                'capabilities': ['design', 'architecture'],
                'priority': 'high'
            })
            
            components.append({
                'name': 'implementation',
                'description': 'Execute implementation according to plan',
                'type': 'execution',
                'estimated_time': 8.0,
                'capabilities': ['implementation', 'coding'],
                'priority': 'normal'
            })
        
        if any(word in objective_lower for word in ['test', 'validate', 'verify']):
            components.append({
                'name': 'testing_validation',
                'description': 'Test and validate implementation',
                'type': 'validation',
                'estimated_time': 2.0,
                'capabilities': ['testing', 'validation'],
                'priority': 'high'
            })
        
        if any(word in objective_lower for word in ['document', 'report', 'specification']):
            components.append({
                'name': 'documentation',
                'description': 'Create comprehensive documentation',
                'type': 'documentation',
                'estimated_time': 1.5,
                'capabilities': ['documentation', 'writing'],
                'priority': 'normal'
            })
        
        # Quality assurance (always last)
        components.append({
            'name': 'quality_assurance',
            'description': 'Final quality review and approval',
            'type': 'qa',
            'estimated_time': 1.0,
            'capabilities': ['quality_assurance', 'review'],
            'priority': 'high'
        })
        
        return components
    
    def _add_step_dependencies(self, graph: nx.DiGraph, components: List[Dict[str, Any]]):
        """Add dependencies between execution steps"""
        step_ids = list(graph.nodes())
        
        # Basic linear dependencies for now
        # Sprint E will add sophisticated dependency analysis
        for i in range(len(step_ids) - 1):
            graph.add_edge(step_ids[i], step_ids[i + 1])
        
        # Add parallel execution opportunities
        # For example, documentation can happen in parallel with implementation
        for i, component in enumerate(components):
            if component['type'] == 'documentation' and i > 0:
                # Documentation can start after planning
                planning_steps = [j for j, c in enumerate(components) if c['type'] == 'planning']
                if planning_steps:
                    planning_step_id = f"step_{planning_steps[0] + 1}"
                    doc_step_id = f"step_{i + 1}"
                    if graph.has_edge(f"step_{i}", doc_step_id):
                        graph.remove_edge(f"step_{i}", doc_step_id)
                    graph.add_edge(planning_step_id, doc_step_id)
    
    def _resolve_graph_cycles(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Resolve cycles in execution graph"""
        try:
            # Find and break cycles
            cycles = list(nx.simple_cycles(graph))
            for cycle in cycles:
                # Remove the last edge in each cycle
                if len(cycle) > 1:
                    graph.remove_edge(cycle[-1], cycle[0])
                    logger.info(f"Removed cycle edge: {cycle[-1]} -> {cycle[0]}")
            
            return graph
        except Exception as e:
            logger.error(f"Failed to resolve graph cycles: {e}")
            return graph
    
    async def _plan_mission_execution(self, mission: Mission):
        """
        Plan mission execution using NetworkX topological sort
        This will be enhanced in Sprint B with expert consultation
        """
        try:
            execution_graph = self.mission_execution_graphs[mission.id]
            
            # Get execution order using NetworkX topological sort
            execution_order = list(nx.topological_sort(execution_graph))
            
            # Calculate total estimated time
            total_time = sum(
                execution_graph.nodes[step_id].get('estimated_time', 1.0) 
                for step_id in execution_order
            )
            
            # Store execution plan
            mission.results['execution_order'] = execution_order
            mission.results['estimated_total_time'] = total_time
            mission.results['parallel_opportunities'] = self._identify_parallel_opportunities(execution_graph)
            
            # Update mission status
            mission.update_status(MissionStatus.EXECUTING)
            
            logger.info(f"Mission {mission.id} execution planned:")
            logger.info(f"  - Execution order: {execution_order}")
            logger.info(f"  - Estimated time: {total_time} hours")
            logger.info(f"  - Parallel opportunities: {len(mission.results['parallel_opportunities'])}")
            
        except Exception as e:
            logger.error(f"Failed to plan mission execution: {e}")
            mission.update_status(MissionStatus.FAILED)
            mission.results['planning_error'] = str(e)
            raise
    
    def _identify_parallel_opportunities(self, graph: nx.DiGraph) -> List[List[str]]:
        """Identify steps that can be executed in parallel"""
        parallel_groups = []
        
        # Find nodes with no dependencies that can run in parallel
        levels = {}
        for node in nx.topological_sort(graph):
            # Calculate the level (longest path from start)
            predecessors = list(graph.predecessors(node))
            if not predecessors:
                levels[node] = 0
            else:
                levels[node] = max(levels[pred] for pred in predecessors) + 1
        
        # Group nodes by level
        level_groups = {}
        for node, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node)
        
        # Find levels with multiple nodes (parallel opportunities)
        for level, nodes in level_groups.items():
            if len(nodes) > 1:
                parallel_groups.append(nodes)
        
        return parallel_groups
    
    def get_mission_status(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed mission status"""
        if mission_id not in self.active_missions:
            return None
        
        mission = self.active_missions[mission_id]
        execution_graph = self.mission_execution_graphs.get(mission_id)
        
        status = mission.to_dict()
        
        if execution_graph:
            status['execution_details'] = {
                'total_steps': len(execution_graph.nodes()),
                'dependencies': len(execution_graph.edges()),
                'is_dag': nx.is_directed_acyclic_graph(execution_graph),
                'critical_path_length': len(nx.dag_longest_path(execution_graph)) if nx.is_directed_acyclic_graph(execution_graph) else 0
            }
        
        return status
    
    def get_all_missions(self) -> List[Dict[str, Any]]:
        """Get status of all missions"""
        return [self.get_mission_status(mission_id) for mission_id in self.active_missions.keys()]
    
    def get_command_metrics(self) -> Dict[str, Any]:
        """Get Major General performance metrics"""
        agent_stats = self.agent_manager.get_agent_stats()
        
        return {
            **self.command_metrics,
            'active_missions': len(self.active_missions),
            'agent_stats': agent_stats,
            'major_general_status': 'operational',
            'last_activity': datetime.now().isoformat()
        }
    
    async def shutdown(self):
        """Graceful shutdown of Major General"""
        logger.info("Major General initiating shutdown...")
        
        # Update any active missions
        for mission in self.active_missions.values():
            if mission.status in [MissionStatus.ANALYZING, MissionStatus.PLANNING, MissionStatus.EXECUTING]:
                mission.update_status(MissionStatus.CANCELLED)
                mission.results['shutdown_reason'] = 'System shutdown'
        
        # Terminate Major General agent
        self.agent_manager.terminate_agent(self.agent_id)
        
        logger.info("Major General shutdown complete")
