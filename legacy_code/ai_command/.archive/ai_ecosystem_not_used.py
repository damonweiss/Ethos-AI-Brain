"""
AI Ecosystem - Global AI Agent Management System
Manages the master ZMQ engine, spawns AgentZero, and coordinates all AI agents
Integrates and supersedes ai_agent_manager.py with modern orchestration patterns
"""

import asyncio
import logging
import networkx as nx
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# Add the Ethos-ZeroMQ path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'PycharmProjects', 'Ethos-ZeroMQ'))

from ethos_zeromq import ZeroMQEngine
from agent_orchestration_base import AgentOrchestrationBase, OrchestrationPattern
from agent_orchestration_reqrep import ReqRepOrchestrator
from zmq_node_base import ProcessorInterface, create_llm_processor, create_custom_processor, create_mcp_processor

logger = logging.getLogger(__name__)


class EcosystemStatus(Enum):
    """AI Ecosystem status states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class AgentRole(Enum):
    """Agent roles in the ecosystem"""
    AGENT_ZERO = "agent_zero"           # Primary AgentZero instance
    MAJOR_GENERAL = "major_general"     # Strategic command
    ADVISOR = "advisor"                 # Specialized consultation
    EXECUTOR = "executor"               # Task execution
    MONITOR = "monitor"                 # System monitoring
    SPECIALIST = "specialist"           # Domain expertise


@dataclass
class EcosystemAgent:
    """Enhanced agent representation for ecosystem management"""
    id: str
    role: AgentRole
    orchestration_pattern: OrchestrationPattern
    agent_instance: AgentOrchestrationBase
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    capabilities: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=lambda: {
        'tasks_completed': 0,
        'success_rate': 1.0,
        'average_response_time': 0.0,
        'last_error': None
    })
    parent_agent_id: Optional[str] = None
    child_agent_ids: List[str] = field(default_factory=list)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'role': self.role.value,
            'orchestration_pattern': self.orchestration_pattern.value,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'capabilities': self.capabilities,
            'performance_metrics': self.performance_metrics,
            'parent_agent_id': self.parent_agent_id,
            'child_agent_ids': self.child_agent_ids,
            'is_running': self.agent_instance.is_running if self.agent_instance else False
        }


class AIEcosystem:
    """
    Global AI Agent Ecosystem Manager
    
    Responsibilities:
    - Manages singleton ZMQ engine for all agents
    - Spawns and manages AgentZero (primary agent)
    - Coordinates agent orchestration patterns
    - Provides ecosystem-wide monitoring and health checks
    - Manages agent lifecycle and relationships
    """
    
    def __init__(self, ecosystem_id: str = "dtos_ai_ecosystem"):
        self.ecosystem_id = ecosystem_id
        self.status = EcosystemStatus.INITIALIZING
        
        # Singleton ZMQ Engine - shared by all agents
        self.master_zmq_engine = ZeroMQEngine(name="EcosystemMasterEngine")
        logger.info("🌍 Created master ZMQ engine for ecosystem")
        
        # Agent registry
        self.agents: Dict[str, EcosystemAgent] = {}
        self.agent_counter = 0
        
        # AgentZero - the primary intelligent agent
        self.agent_zero: Optional[EcosystemAgent] = None
        
        # NetworkX intelligence graphs (from ai_agent_manager)
        self.agent_dependency_graph = nx.DiGraph()
        self.capability_graph = nx.Graph()
        self.communication_graph = nx.DiGraph()
        self.orchestration_graph = nx.DiGraph()  # New: tracks orchestration patterns
        
        # Ecosystem configuration
        self.max_agents = 1000
        self.health_check_interval = 30  # seconds
        self.auto_recovery = True
        
        # Performance tracking
        self.ecosystem_metrics = {
            'total_tasks_processed': 0,
            'average_response_time': 0.0,
            'uptime_start': datetime.now(),
            'last_health_check': None
        }
        
        logger.info(f"🌍 AI Ecosystem '{ecosystem_id}' initialized")
    
    async def initialize(self):
        """Initialize the AI ecosystem"""
        try:
            self.status = EcosystemStatus.INITIALIZING
            
            # Initialize master ZMQ engine
            logger.info("🔧 Initializing master ZMQ engine...")
            
            # Spawn AgentZero as the primary agent
            await self.spawn_agent_zero()
            
            # Initialize intelligence graphs
            self._initialize_intelligence_graphs()
            
            # Start health monitoring
            asyncio.create_task(self._health_monitor_loop())
            
            self.status = EcosystemStatus.ACTIVE
            logger.info("✅ AI Ecosystem fully initialized and active")
            
        except Exception as e:
            self.status = EcosystemStatus.ERROR
            logger.error(f"❌ Failed to initialize AI ecosystem: {e}")
            raise
    
    async def spawn_agent_zero(self) -> str:
        """
        Spawn AgentZero - the primary intelligent agent
        AgentZero is a REQ-REP orchestrator that can spawn child agents
        """
        if self.agent_zero:
            logger.warning("AgentZero already exists")
            return self.agent_zero.id
        
        try:
            agent_id = "agent_zero_001"
            
            # Create AgentZero as REQ-REP orchestrator with LLM + MCP capabilities
            agent_zero_instance = ReqRepOrchestrator(
                agent_id=agent_id,
                base_port=7000,  # AgentZero gets dedicated port range
                shared_zmq_engine=self.master_zmq_engine  # Use shared engine
            )
            
            # Add core processors to AgentZero
            agent_zero_instance.add_processor(
                create_llm_processor("agent_zero_brain", "gpt-4")
            )
            agent_zero_instance.add_processor(
                create_mcp_processor("agent_zero_tools", "system_manager")
            )
            
            # Start AgentZero
            await agent_zero_instance.start()
            
            # Register in ecosystem
            self.agent_zero = EcosystemAgent(
                id=agent_id,
                role=AgentRole.AGENT_ZERO,
                orchestration_pattern=OrchestrationPattern.SUPERVISOR,
                agent_instance=agent_zero_instance,
                capabilities=[
                    "strategic_reasoning", "agent_spawning", "task_coordination",
                    "llm_processing", "mcp_tool_usage", "recursive_delegation"
                ]
            )
            
            self.agents[agent_id] = self.agent_zero
            self._update_intelligence_graphs(self.agent_zero)
            
            logger.info(f"🤖 AgentZero spawned successfully: {agent_id}")
            return agent_id
            
        except Exception as e:
            logger.error(f"❌ Failed to spawn AgentZero: {e}")
            raise
    
    async def spawn_agent(
        self,
        role: AgentRole,
        orchestration_pattern: OrchestrationPattern,
        processors: List[ProcessorInterface] = None,
        capabilities: List[str] = None,
        parent_agent_id: Optional[str] = None
    ) -> str:
        """
        Spawn a new agent in the ecosystem
        
        Args:
            role: Agent's role in the ecosystem
            orchestration_pattern: ZMQ orchestration pattern to use
            processors: List of processors for the agent
            capabilities: Agent capabilities
            parent_agent_id: ID of parent agent (for hierarchical spawning)
        """
        if len(self.agents) >= self.max_agents:
            raise RuntimeError(f"Maximum agent limit ({self.max_agents}) reached")
        
        try:
            # Generate unique agent ID
            self.agent_counter += 1
            agent_id = f"{role.value}_{self.agent_counter:04d}"
            
            # Calculate base port (avoid conflicts)
            base_port = 8000 + (self.agent_counter * 100)
            
            # Create agent instance based on orchestration pattern
            if orchestration_pattern == OrchestrationPattern.SUPERVISOR:
                agent_instance = ReqRepOrchestrator(
                    agent_id=agent_id,
                    base_port=base_port,
                    shared_zmq_engine=self.master_zmq_engine
                )
            else:
                # For now, default to REQ-REP. Later add other patterns
                agent_instance = ReqRepOrchestrator(
                    agent_id=agent_id,
                    base_port=base_port,
                    shared_zmq_engine=self.master_zmq_engine
                )
            
            # Add processors if provided
            if processors:
                for processor in processors:
                    agent_instance.add_processor(processor)
            
            # Start agent
            await agent_instance.start()
            
            # Create ecosystem agent record
            ecosystem_agent = EcosystemAgent(
                id=agent_id,
                role=role,
                orchestration_pattern=orchestration_pattern,
                agent_instance=agent_instance,
                capabilities=capabilities or [],
                parent_agent_id=parent_agent_id
            )
            
            # Register in ecosystem
            self.agents[agent_id] = ecosystem_agent
            
            # Update parent-child relationships
            if parent_agent_id and parent_agent_id in self.agents:
                parent_agent = self.agents[parent_agent_id]
                parent_agent.child_agent_ids.append(agent_id)
                
                # Add to parent's orchestration if it supports children
                if hasattr(parent_agent.agent_instance, 'add_child_agent'):
                    parent_agent.agent_instance.add_child_agent(agent_instance)
            
            # Update intelligence graphs
            self._update_intelligence_graphs(ecosystem_agent)
            
            logger.info(f"🚀 Spawned {role.value} agent: {agent_id}")
            return agent_id
            
        except Exception as e:
            logger.error(f"❌ Failed to spawn agent {role.value}: {e}")
            raise
    
    async def terminate_agent(self, agent_id: str) -> bool:
        """Terminate an agent and clean up resources"""
        if agent_id not in self.agents:
            logger.warning(f"Attempted to terminate non-existent agent: {agent_id}")
            return False
        
        try:
            ecosystem_agent = self.agents[agent_id]
            
            # Don't allow terminating AgentZero
            if ecosystem_agent.role == AgentRole.AGENT_ZERO:
                logger.warning("Cannot terminate AgentZero - use shutdown_ecosystem() instead")
                return False
            
            # Terminate child agents first
            for child_id in ecosystem_agent.child_agent_ids.copy():
                await self.terminate_agent(child_id)
            
            # Stop the agent instance
            if ecosystem_agent.agent_instance:
                await ecosystem_agent.agent_instance.stop()
            
            # Remove from parent's child list
            if ecosystem_agent.parent_agent_id:
                parent = self.agents.get(ecosystem_agent.parent_agent_id)
                if parent and agent_id in parent.child_agent_ids:
                    parent.child_agent_ids.remove(agent_id)
            
            # Remove from ecosystem
            del self.agents[agent_id]
            
            # Update intelligence graphs
            self._remove_from_intelligence_graphs(agent_id)
            
            logger.info(f"🛑 Terminated agent: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to terminate agent {agent_id}: {e}")
            return False
    
    async def delegate_to_agent_zero(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delegate a task to AgentZero for intelligent processing
        AgentZero will decide whether to handle locally or spawn child agents
        """
        if not self.agent_zero or not self.agent_zero.agent_instance:
            raise RuntimeError("AgentZero not available")
        
        try:
            self.agent_zero.update_activity()
            
            # Delegate to AgentZero's orchestration
            result = await self.agent_zero.agent_instance.execute_task(task)
            
            # Update metrics
            self.ecosystem_metrics['total_tasks_processed'] += 1
            self.agent_zero.performance_metrics['tasks_completed'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Task delegation to AgentZero failed: {e}")
            self.agent_zero.performance_metrics['last_error'] = str(e)
            raise
    
    def get_ecosystem_status(self) -> Dict[str, Any]:
        """Get comprehensive ecosystem status"""
        uptime = datetime.now() - self.ecosystem_metrics['uptime_start']
        
        return {
            'ecosystem_id': self.ecosystem_id,
            'status': self.status.value,
            'uptime_seconds': uptime.total_seconds(),
            'agent_count': len(self.agents),
            'agent_zero_status': self.agent_zero.to_dict() if self.agent_zero else None,
            'agents_by_role': self._get_agents_by_role_stats(),
            'orchestration_patterns': self._get_orchestration_pattern_stats(),
            'performance_metrics': self.ecosystem_metrics,
            'zmq_engine_status': 'active' if self.master_zmq_engine else 'inactive'
        }
    
    def _get_agents_by_role_stats(self) -> Dict[str, int]:
        """Get agent count by role"""
        stats = {}
        for agent in self.agents.values():
            role = agent.role.value
            stats[role] = stats.get(role, 0) + 1
        return stats
    
    def _get_orchestration_pattern_stats(self) -> Dict[str, int]:
        """Get agent count by orchestration pattern"""
        stats = {}
        for agent in self.agents.values():
            pattern = agent.orchestration_pattern.value
            stats[pattern] = stats.get(pattern, 0) + 1
        return stats
    
    def _initialize_intelligence_graphs(self):
        """Initialize NetworkX intelligence graphs"""
        # Agent roles and their typical relationships
        role_relationships = [
            (AgentRole.AGENT_ZERO, AgentRole.MAJOR_GENERAL),
            (AgentRole.MAJOR_GENERAL, AgentRole.ADVISOR),
            (AgentRole.MAJOR_GENERAL, AgentRole.EXECUTOR),
            (AgentRole.ADVISOR, AgentRole.SPECIALIST),
            (AgentRole.EXECUTOR, AgentRole.MONITOR)
        ]
        
        for parent_role, child_role in role_relationships:
            self.agent_dependency_graph.add_edge(parent_role.value, child_role.value)
        
        logger.info("🧠 Intelligence graphs initialized")
    
    def _update_intelligence_graphs(self, ecosystem_agent: EcosystemAgent):
        """Update intelligence graphs with new agent"""
        agent_id = ecosystem_agent.id
        
        # Add to capability graph
        self.capability_graph.add_node(
            agent_id,
            role=ecosystem_agent.role.value,
            capabilities=ecosystem_agent.capabilities,
            orchestration_pattern=ecosystem_agent.orchestration_pattern.value
        )
        
        # Add to orchestration graph
        self.orchestration_graph.add_node(
            agent_id,
            pattern=ecosystem_agent.orchestration_pattern.value,
            role=ecosystem_agent.role.value
        )
        
        # Add parent-child relationships
        if ecosystem_agent.parent_agent_id:
            self.orchestration_graph.add_edge(
                ecosystem_agent.parent_agent_id,
                agent_id,
                relationship="parent_child"
            )
    
    def _remove_from_intelligence_graphs(self, agent_id: str):
        """Remove agent from intelligence graphs"""
        graphs = [self.capability_graph, self.orchestration_graph, self.communication_graph]
        
        for graph in graphs:
            if agent_id in graph:
                graph.remove_node(agent_id)
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while self.status == EcosystemStatus.ACTIVE:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"❌ Health monitor error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _perform_health_check(self):
        """Perform ecosystem health check"""
        self.ecosystem_metrics['last_health_check'] = datetime.now()
        
        unhealthy_agents = []
        
        for agent_id, agent in self.agents.items():
            if agent.agent_instance and not agent.agent_instance.is_running:
                unhealthy_agents.append(agent_id)
        
        if unhealthy_agents:
            logger.warning(f"⚠️ Unhealthy agents detected: {unhealthy_agents}")
            
            if self.auto_recovery:
                for agent_id in unhealthy_agents:
                    logger.info(f"🔄 Attempting to recover agent {agent_id}")
                    # Add recovery logic here
    
    async def shutdown_ecosystem(self):
        """Gracefully shutdown the entire ecosystem"""
        logger.info("🛑 Shutting down AI ecosystem...")
        self.status = EcosystemStatus.SHUTDOWN
        
        # Terminate all agents except AgentZero
        agent_ids = list(self.agents.keys())
        for agent_id in agent_ids:
            if agent_id != self.agent_zero.id if self.agent_zero else None:
                await self.terminate_agent(agent_id)
        
        # Terminate AgentZero last
        if self.agent_zero:
            await self.agent_zero.agent_instance.stop()
        
        # Stop master ZMQ engine
        self.master_zmq_engine.stop_all_servers()
        
        logger.info("✅ AI ecosystem shutdown complete")
    
    def __repr__(self):
        return f"AIEcosystem(id={self.ecosystem_id}, status={self.status.value}, agents={len(self.agents)})"


# Global ecosystem instance (singleton pattern)
_global_ecosystem: Optional[AIEcosystem] = None


def get_ecosystem() -> AIEcosystem:
    """Get the global AI ecosystem instance"""
    global _global_ecosystem
    if _global_ecosystem is None:
        _global_ecosystem = AIEcosystem()
    return _global_ecosystem


async def initialize_global_ecosystem() -> AIEcosystem:
    """Initialize the global AI ecosystem"""
    ecosystem = get_ecosystem()
    await ecosystem.initialize()
    return ecosystem
