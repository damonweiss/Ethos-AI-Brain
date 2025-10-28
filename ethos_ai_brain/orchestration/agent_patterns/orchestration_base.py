"""
Agent Orchestration Base Class
Provides ZMQ + Processor integration with hierarchical parent-child communication
Uses ZMQ_Node containers that can hold LLMs, neural networks, MCP tools, etc.
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

# Import your existing components
import sys
import os

sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'PycharmProjects', 'Ethos-ZeroMQ'))

from ethos_zeromq import *
from ...core.zeromq.zmq_node_base import (
    ProcessorType, ProcessingRequest, ProcessingResponse, ProcessorInterface, ZMQNode
)


class OrchestrationPattern(Enum):
    """Supported orchestration patterns"""
    SUPERVISOR = "supervisor"  # Central coordinator with specialized agents
    SCATTER_GATHER = "scatter_gather"  # Parallel execution with result aggregation
    PIPELINE = "pipeline"  # Sequential processing stages
    EVENT_DRIVEN = "event_driven"  # Pub-sub reactive coordination
    HIERARCHICAL = "hierarchical"  # Multi-level supervision
    NETWORK = "network"  # Peer-to-peer communication


@dataclass
class AgentMessage:
    """Standard message format for agent communication"""
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: str
    content: Dict[str, Any]
    parent_message_id: Optional[str] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            import time
            self.timestamp = time.time()


@dataclass
class AgentCapability:
    """Defines what an agent can do (now wraps processor capabilities)"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    # Removed processor_types - capabilities are now generic
    cost_estimate: float = 0.0


class AgentOrchestrationBase(ABC):
    """
    Base class for all agent orchestration patterns

    Features:
    - ZMQ Node containers with multiple processor types
    - Intelligent routing between processors (LLM, MCP, Neural Networks, etc.)
    - Parent-child REQ-REP communication
    - Hierarchical agent management
    - Message routing and handling
    """

    def __init__(
            self,
            agent_id: str,
            orchestration_pattern: OrchestrationPattern,
            parent_agent: Optional['AgentOrchestrationBase'] = None,
            base_port: int = 5555,
            capabilities: List[AgentCapability] = None,
            processors: List[ProcessorInterface] = None,
            shared_zmq_engine: Any = None
    ):
        self.agent_id = agent_id
        self.orchestration_pattern = orchestration_pattern
        self.parent_agent = parent_agent
        self.base_port = base_port
        self.capabilities = capabilities or []
        self.shared_zmq_engine = shared_zmq_engine

        # Child agents managed by this orchestrator
        self.child_agents: Dict[str, 'AgentOrchestrationBase'] = {}

        # AI Brain for decision making (placeholder - can be set later)
        self.brain = None

        # ZMQ Node container with processors (ONLY ZMQ infrastructure)
        self.node = self._create_node(processors or [])

        # Agent uses node's ZMQ engine (shared or node's own)
        self.zmq_engine = shared_zmq_engine if shared_zmq_engine else self.node.zmq_engine
        self.zmq_servers: Dict[str, Any] = {}
        self.is_running = False

        # Message handlers
        self.message_handlers: Dict[str, Callable] = {}

        # Register default message handlers
        self._register_default_handlers()

        # Note: Orchestration initialization happens in start() method

    @abstractmethod
    async def _initialize_orchestration(self):
        """Subclasses implement their own orchestration setup"""
        pass

    def _create_node(self, processors: List[ProcessorInterface]) -> ZMQNode:
        """Create ZMQ node with processors"""
        return ZMQNode(
            node_id=f"{self.agent_id}_node",
            processors=processors,
            base_port=self.base_port,
            shared_zmq_engine=self.shared_zmq_engine  # Pass shared engine to node
        )

    def add_processor(self, processor: ProcessorInterface):
        """Add a processor to this agent's node"""
        self.node.add_processor(processor)
        print(f"🔧 Added {processor.processor_type.value} processor to agent {self.agent_id}")

    def get_processor(self, processor_id: str) -> Optional[ProcessorInterface]:
        """Get a specific processor from this agent's node"""
        return self.node.get_processor(processor_id)

    async def process_with_node(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process request using this agent's node processors"""
        return await self.node.process_request(request)

    def get_available_processors(self) -> List[ProcessorType]:
        """Get list of available processor types in this agent's node"""
        return [processor.processor_type for processor in self.node.processors.values()]

    def has_processor_type(self, processor_type: ProcessorType) -> bool:
        """Check if this agent has a specific processor type"""
        return any(p.processor_type == processor_type for p in self.node.processors.values())

    def get_idle_processors(self) -> List[ProcessorInterface]:
        """Get list of processors that are not currently busy"""
        return [p for p in self.node.processors.values() if not p.is_busy]

    def _register_default_handlers(self):
        """Register default message handlers"""
        self.message_handlers.update({
            "task_request": self._handle_task_request,
            "status_query": self._handle_status_query,
            "child_registration": self._handle_child_registration,
            "parent_communication": self._handle_parent_communication
        })

    async def start(self):
        """Start the orchestration agent"""
        print(f"🚀 Starting agent {self.agent_id}...")

        # Initialize orchestration (subclass-specific)
        await self._initialize_orchestration()

        # Start the ZMQ node
        await self.node.initialize_node()

        # Start orchestration-specific logic
        await self._start_orchestration()

        self.is_running = True
        print(f"🚀 Agent {self.agent_id} started with {self.orchestration_pattern.value} pattern")

    async def stop(self):
        """Stop the agent orchestration system"""
        self.is_running = False

        # Stop all child agents
        for child_id, child_agent in self.child_agents.items():
            await child_agent.stop()

        # Stop ZMQ node with processors
        await self.node.shutdown_node()

        # Stop ZMQ infrastructure
        self.zmq_engine.stop_all_servers()

        print(f"🛑 Agent {self.agent_id} stopped")

    def add_child_agent(self, child_agent: 'AgentOrchestrationBase'):
        """Add a child agent to this orchestrator"""
        self.child_agents[child_agent.agent_id] = child_agent
        child_agent.parent_agent = self
        print(f"👶 Child agent {child_agent.agent_id} added to {self.agent_id}")

    def remove_child_agent(self, child_id: str):
        """Remove a child agent"""
        if child_id in self.child_agents:
            del self.child_agents[child_id]
            print(f"👋 Child agent {child_id} removed from {self.agent_id}")

    async def send_to_child(self, child_id: str, message: AgentMessage) -> Dict[str, Any]:
        """Send message to specific child agent"""
        if child_id not in self.child_agents:
            raise ValueError(f"Child agent {child_id} not found")

        # Route through appropriate ZMQ pattern
        return await self._route_message_to_child(child_id, message)

    async def broadcast_to_children(self, message: AgentMessage) -> Dict[str, List[Any]]:
        """Broadcast message to all child agents"""
        results = {}
        for child_id in self.child_agents.keys():
            try:
                result = await self.send_to_child(child_id, message)
                results[child_id] = result
            except Exception as e:
                results[child_id] = {"error": str(e)}

        return results

    @abstractmethod
    async def _start_orchestration(self):
        """Start pattern-specific orchestration logic"""
        pass

    @abstractmethod
    async def _route_message_to_child(self, child_id: str, message: AgentMessage) -> Dict[str, Any]:
        """Route message to child using pattern-specific ZMQ"""
        pass

    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using this orchestration pattern"""
        pass

    # Default Message Handlers

    async def _handle_task_request(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle incoming task requests - delegated to subclass"""
        task = message.content.get("task", {})

        # Let subclass handle task routing and execution
        result = await self.execute_task(task)

        return {
            "status": "completed",
            "result": result
        }

    async def _handle_status_query(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle status queries"""
        return {
            "agent_id": self.agent_id,
            "orchestration_pattern": self.orchestration_pattern.value,
            "is_running": self.is_running,
            "child_count": len(self.child_agents),
            "capabilities": [cap.name for cap in self.capabilities]
        }

    async def _handle_capability_query(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle capability queries"""
        return {
            "agent_id": self.agent_id,
            "capabilities": [cap.__dict__ for cap in self.capabilities]
        }

    async def _handle_child_registration(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle child agent registration"""
        child_info = message.content
        child_id = child_info["agent_id"]

        # Store child agent information
        # Note: Actual child agent object would be added via add_child_agent()
        print(f"📝 Child {child_id} registered with {self.agent_id}")

        return {
            "status": "registered",
            "parent_id": self.agent_id,
            "welcome_message": f"Welcome to {self.orchestration_pattern.value} orchestration"
        }

    async def _handle_parent_communication(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle communication from parent agent"""
        return await self._handle_task_request(message)

    async def _handle_agent_task(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle agent-specific tasks (for supervisor pattern)"""
        return await self._handle_task_request(message)

    def get_agent_hierarchy(self) -> Dict[str, Any]:
        """Get the complete agent hierarchy"""
        hierarchy = {
            "agent_id": self.agent_id,
            "orchestration_pattern": self.orchestration_pattern.value,
            "capabilities": [cap.name for cap in self.capabilities],
            "children": {}
        }

        for child_id, child_agent in self.child_agents.items():
            hierarchy["children"][child_id] = child_agent.get_agent_hierarchy()

        return hierarchy

    def __repr__(self):
        return f"AgentOrchestrator(id={self.agent_id}, pattern={self.orchestration_pattern.value}, children={len(self.child_agents)})"



