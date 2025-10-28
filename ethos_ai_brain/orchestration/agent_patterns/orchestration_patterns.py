"""
Specific Orchestration Pattern Implementations
Each class implements a specific agent coordination behavior
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
# Import path fix
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'PycharmProjects', 'Ethos-ZeroMQ'))

from .orchestration_base import (
    AgentOrchestrationBase,
    OrchestrationPattern,
    AgentMessage,
    AgentCapability
)


class SupervisorOrchestrator(AgentOrchestrationBase):
    """
    Supervisor Pattern: Central coordinator manages specialized agents
    Uses ROUTER-DEALER for agent communication
    """

    def __init__(self, agent_id: str, **kwargs):
        super().__init__(
            agent_id=agent_id,
            orchestration_pattern=OrchestrationPattern.SUPERVISOR,
            **kwargs
        )
        self.agent_assignments: Dict[str, str] = {}  # task_type -> agent_id
        self.agent_workloads: Dict[str, int] = {}    # agent_id -> current_tasks

    async def _initialize_orchestration(self):
        """Initialize supervisor orchestration"""
        print(f"🎯 Supervisor {self.agent_id} ready to coordinate {len(self.child_agents)} agents")

        # Analyze child agent capabilities for smart assignment
        await self._analyze_agent_capabilities()

    async def _start_orchestration(self):
        """Start supervisor orchestration"""
        print(f"🎯 Supervisor {self.agent_id} starting orchestration")
        # Additional startup logic can go here

    async def _analyze_agent_capabilities(self):
        """Use LLM to analyze and categorize child agent capabilities"""
        if not self.child_agents:
            return

        agent_info = {}
        for child_id, child_agent in self.child_agents.items():
            agent_info[child_id] = [cap.name for cap in child_agent.capabilities]

        analysis = await self.brain.think(
            f"Analyze agent capabilities for optimal task assignment: {agent_info}",
            context={
                "orchestration_type": "supervisor",
                "total_agents": len(self.child_agents)
            }
        )

        # Update agent assignments based on LLM analysis
        if analysis.get("recommended_assignments"):
            self.agent_assignments.update(analysis["recommended_assignments"])

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task by delegating to appropriate child agent"""
        task_type = task.get("type", "general")

        # Use LLM to determine best agent for this task
        agent_selection = await self.brain.think(
            f"Select best agent for task: {task}",
            context={
                "available_agents": list(self.child_agents.keys()),
                "agent_capabilities": {
                    child_id: [cap.name for cap in child.capabilities]
                    for child_id, child in self.child_agents.items()
                },
                "current_workloads": self.agent_workloads
            }
        )

        selected_agent_id = agent_selection.get("selected_agent")
        if not selected_agent_id or selected_agent_id not in self.child_agents:
            return {"error": "No suitable agent found for task"}

        # Create task message
        task_message = AgentMessage(
            message_id=f"task_{task.get('id', 'unknown')}",
            sender_id=self.agent_id,
            recipient_id=selected_agent_id,
            message_type="task_request",
            content={"task": task}
        )

        # Send to selected agent and track workload
        self.agent_workloads[selected_agent_id] = self.agent_workloads.get(selected_agent_id, 0) + 1

        try:
            result = await self.send_to_child(selected_agent_id, task_message)
            return {
                "status": "completed",
                "assigned_agent": selected_agent_id,
                "result": result
            }
        finally:
            self.agent_workloads[selected_agent_id] -= 1

    async def _route_message_to_child(self, child_id: str, message: AgentMessage) -> Dict[str, Any]:
        """Route message via ROUTER-DEALER"""
        if hasattr(self, 'supervisor_router'):
            # Send via ROUTER socket to specific DEALER
            message_json = json.dumps(message.__dict__)
            return self.supervisor_router.send_to_dealer(child_id, message_json)
        else:
            raise RuntimeError("Supervisor router not initialized")


class ScatterGatherOrchestrator(AgentOrchestrationBase):
    """
    Scatter-Gather Pattern: Parallel execution with result aggregation
    Perfect for expert panels, consensus building, parallel analysis
    """

    def __init__(self, agent_id: str, **kwargs):
        super().__init__(
            agent_id=agent_id,
            orchestration_pattern=OrchestrationPattern.SCATTER_GATHER,
            **kwargs
        )
        self.gather_timeout = 30.0  # seconds

    async def _start_orchestration(self):
        """Start scatter-gather orchestration"""
        print(f"🌐 Scatter-Gather {self.agent_id} ready for parallel coordination")

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task by scattering to all agents and gathering results"""
        if not self.child_agents:
            return {"error": "No child agents available for scatter-gather"}

        # Use LLM to customize task for each agent
        task_customization = await self.brain.think(
            f"Customize task for each agent in scatter-gather: {task}",
            context={
                "agents": {
                    child_id: [cap.name for cap in child.capabilities]
                    for child_id, child in self.child_agents.items()
                },
                "task_type": task.get("type", "analysis")
            }
        )

        # SCATTER: Send (potentially customized) tasks to all agents
        scatter_tasks = []
        for child_id in self.child_agents.keys():
            customized_task = task_customization.get("customized_tasks", {}).get(child_id, task)

            task_message = AgentMessage(
                message_id=f"scatter_{task.get('id', 'unknown')}_{child_id}",
                sender_id=self.agent_id,
                recipient_id=child_id,
                message_type="task_request",
                content={"task": customized_task}
            )

            scatter_tasks.append(self.send_to_child(child_id, task_message))

        # GATHER: Wait for all results (with timeout)
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*scatter_tasks, return_exceptions=True),
                timeout=self.gather_timeout
            )

            # Process and aggregate results
            aggregated_result = await self._aggregate_results(task, results)

            return {
                "status": "completed",
                "pattern": "scatter_gather",
                "individual_results": dict(zip(self.child_agents.keys(), results)),
                "aggregated_result": aggregated_result
            }

        except asyncio.TimeoutError:
            return {"error": f"Scatter-gather timeout after {self.gather_timeout}s"}

    async def _aggregate_results(self, original_task: Dict[str, Any], results: List[Any]) -> Dict[str, Any]:
        """Use LLM to intelligently aggregate results from multiple agents"""
        aggregation_prompt = f"""
        Aggregate results from multiple agents for task: {original_task}
        
        Individual Results: {results}
        
        Provide a synthesized, comprehensive result that combines the best insights from all agents.
        """

        aggregation = await self.brain.think(
            aggregation_prompt,
            context={
                "aggregation_type": "scatter_gather",
                "result_count": len(results),
                "task_type": original_task.get("type", "analysis")
            }
        )

        return aggregation

    async def _route_message_to_child(self, child_id: str, message: AgentMessage) -> Dict[str, Any]:
        """Route message via SCATTER-GATHER pattern"""
        if hasattr(self, 'scatter_gather'):
            message_json = json.dumps(message.__dict__)
            return self.scatter_gather.scatter_to_agent(child_id, message_json)
        else:
            raise RuntimeError("Scatter-gather server not initialized")


class PipelineOrchestrator(AgentOrchestrationBase):
    """
    Pipeline Pattern: Sequential processing stages
    Each agent processes and passes to the next stage
    """

    def __init__(self, agent_id: str, pipeline_stages: List[str] = None, **kwargs):
        super().__init__(
            agent_id=agent_id,
            orchestration_pattern=OrchestrationPattern.PIPELINE,
            **kwargs
        )
        self.pipeline_stages = pipeline_stages or []
        self.stage_agents: Dict[str, str] = {}  # stage_name -> agent_id

    async def _start_orchestration(self):
        """Start pipeline orchestration"""
        print(f"🔄 Pipeline {self.agent_id} ready with {len(self.pipeline_stages)} stages")

        # Map stages to agents
        await self._map_stages_to_agents()

    async def _map_stages_to_agents(self):
        """Use LLM to optimally map pipeline stages to available agents"""
        if not self.child_agents or not self.pipeline_stages:
            return

        mapping_analysis = await self.brain.think(
            f"Map pipeline stages {self.pipeline_stages} to agents based on capabilities",
            context={
                "agents": {
                    child_id: [cap.name for cap in child.capabilities]
                    for child_id, child in self.child_agents.items()
                },
                "stages": self.pipeline_stages
            }
        )

        if mapping_analysis.get("stage_mapping"):
            self.stage_agents.update(mapping_analysis["stage_mapping"])

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task through pipeline stages"""
        if not self.pipeline_stages:
            return {"error": "No pipeline stages defined"}

        current_data = task
        stage_results = {}

        for stage in self.pipeline_stages:
            stage_agent_id = self.stage_agents.get(stage)
            if not stage_agent_id or stage_agent_id not in self.child_agents:
                return {"error": f"No agent assigned to stage: {stage}"}

            # Create stage-specific task
            stage_task = {
                "stage": stage,
                "input_data": current_data,
                "pipeline_context": {
                    "current_stage": stage,
                    "total_stages": len(self.pipeline_stages),
                    "previous_results": stage_results
                }
            }

            stage_message = AgentMessage(
                message_id=f"pipeline_{task.get('id', 'unknown')}_{stage}",
                sender_id=self.agent_id,
                recipient_id=stage_agent_id,
                message_type="task_request",
                content={"task": stage_task}
            )

            # Process stage
            stage_result = await self.send_to_child(stage_agent_id, stage_message)
            stage_results[stage] = stage_result

            # Output of this stage becomes input for next stage
            current_data = stage_result.get("output_data", stage_result)

        return {
            "status": "completed",
            "pattern": "pipeline",
            "final_result": current_data,
            "stage_results": stage_results
        }

    async def _route_message_to_child(self, child_id: str, message: AgentMessage) -> Dict[str, Any]:
        """Route message via PUSH-PULL pipeline"""
        if hasattr(self, 'pipeline_push'):
            message_json = json.dumps(message.__dict__)
            return self.pipeline_push.push_to_stage(child_id, message_json)
        else:
            raise RuntimeError("Pipeline server not initialized")


class EventDrivenOrchestrator(AgentOrchestrationBase):
    """
    Event-Driven Pattern: Pub-Sub reactive coordination
    Agents respond to events and publish their own events
    """

    def __init__(self, agent_id: str, **kwargs):
        super().__init__(
            agent_id=agent_id,
            orchestration_pattern=OrchestrationPattern.EVENT_DRIVEN,
            **kwargs
        )
        self.event_subscriptions: Dict[str, List[str]] = {}  # event_type -> [agent_ids]
        self.event_history: List[Dict[str, Any]] = []

    async def _start_orchestration(self):
        """Start event-driven orchestration"""
        print(f"📡 Event-Driven {self.agent_id} ready for reactive coordination")

        # Set up event subscriptions based on agent capabilities
        await self._setup_event_subscriptions()

    async def _setup_event_subscriptions(self):
        """Use LLM to determine which agents should subscribe to which events"""
        if not self.child_agents:
            return

        subscription_analysis = await self.brain.think(
            "Determine optimal event subscriptions for agents based on their capabilities",
            context={
                "agents": {
                    child_id: [cap.name for cap in child.capabilities]
                    for child_id, child in self.child_agents.items()
                },
                "orchestration_type": "event_driven"
            }
        )

        if subscription_analysis.get("event_subscriptions"):
            self.event_subscriptions.update(subscription_analysis["event_subscriptions"])

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task by publishing events and coordinating responses"""
        event_type = task.get("event_type", "general_task")
        event_data = task.get("event_data", task)

        # Publish event
        await self.publish_event(event_type, event_data)

        # Wait for and collect responses
        responses = await self._collect_event_responses(event_type, timeout=10.0)

        return {
            "status": "completed",
            "pattern": "event_driven",
            "event_type": event_type,
            "responses": responses
        }

    async def publish_event(self, event_type: str, event_data: Dict[str, Any]):
        """Publish event to subscribed agents"""
        event = {
            "event_type": event_type,
            "event_data": event_data,
            "timestamp": asyncio.get_event_loop().time(),
            "publisher": self.agent_id
        }

        # Add to history
        self.event_history.append(event)

        # Send to subscribed agents
        subscribed_agents = self.event_subscriptions.get(event_type, [])
        for agent_id in subscribed_agents:
            if agent_id in self.child_agents:
                event_message = AgentMessage(
                    message_id=f"event_{event_type}_{len(self.event_history)}",
                    sender_id=self.agent_id,
                    recipient_id=agent_id,
                    message_type="event_notification",
                    content={"event": event}
                )

                await self.send_to_child(agent_id, event_message)

        print(f"📢 Published event {event_type} to {len(subscribed_agents)} agents")

    async def _collect_event_responses(self, event_type: str, timeout: float) -> Dict[str, Any]:
        """Collect responses from agents that received the event"""
        # This would typically involve waiting for responses via a separate channel
        # For now, return a placeholder
        return {"message": f"Collected responses for {event_type}"}

    async def _route_message_to_child(self, child_id: str, message: AgentMessage) -> Dict[str, Any]:
        """Route message via PUB-SUB"""
        if hasattr(self, 'event_publisher'):
            message_json = json.dumps(message.__dict__)
            return self.event_publisher.publish_to_subscriber(child_id, message_json)
        else:
            raise RuntimeError("Event publisher not initialized")


class HierarchicalOrchestrator(AgentOrchestrationBase):
    """
    Hierarchical Pattern: Multi-level supervision
    Combines supervisor pattern with recursive hierarchy
    """

    def __init__(self, agent_id: str, max_children_per_supervisor: int = 5, **kwargs):
        super().__init__(
            agent_id=agent_id,
            orchestration_pattern=OrchestrationPattern.HIERARCHICAL,
            **kwargs
        )
        self.max_children_per_supervisor = max_children_per_supervisor
        self.sub_supervisors: Dict[str, SupervisorOrchestrator] = {}

    async def _start_orchestration(self):
        """Start hierarchical orchestration"""
        print(f"🏢 Hierarchical {self.agent_id} ready for multi-level coordination")

        # Organize agents into hierarchical structure if needed
        await self._organize_hierarchy()

    async def _organize_hierarchy(self):
        """Organize child agents into hierarchical structure"""
        if len(self.child_agents) <= self.max_children_per_supervisor:
            return  # No need for sub-supervisors

        # Use LLM to determine optimal hierarchical organization
        organization_analysis = await self.brain.think(
            f"Organize {len(self.child_agents)} agents into hierarchical structure",
            context={
                "max_per_supervisor": self.max_children_per_supervisor,
                "agent_capabilities": {
                    child_id: [cap.name for cap in child.capabilities]
                    for child_id, child in self.child_agents.items()
                }
            }
        )

        # Create sub-supervisors based on LLM recommendations
        if organization_analysis.get("supervisor_groups"):
            for group_name, agent_ids in organization_analysis["supervisor_groups"].items():
                sub_supervisor = SupervisorOrchestrator(
                    agent_id=f"{self.agent_id}_sub_{group_name}",
                    parent_agent=self
                )

                # Move agents to sub-supervisor
                for agent_id in agent_ids:
                    if agent_id in self.child_agents:
                        agent = self.child_agents.pop(agent_id)
                        sub_supervisor.add_child_agent(agent)

                self.sub_supervisors[group_name] = sub_supervisor
                self.add_child_agent(sub_supervisor)

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task through hierarchical structure"""
        # Use LLM to determine which level/supervisor should handle the task
        routing_decision = await self.brain.think(
            f"Route task through hierarchical structure: {task}",
            context={
                "direct_children": len(self.child_agents),
                "sub_supervisors": list(self.sub_supervisors.keys()),
                "task_complexity": task.get("complexity", "medium")
            }
        )

        target_supervisor = routing_decision.get("target_supervisor", "direct")

        if target_supervisor == "direct":
            # Handle directly with immediate children
            return await super().execute_task(task)
        elif target_supervisor in self.sub_supervisors:
            # Route to specific sub-supervisor
            return await self.sub_supervisors[target_supervisor].execute_task(task)
        else:
            return {"error": f"Unknown supervisor target: {target_supervisor}"}

    async def _route_message_to_child(self, child_id: str, message: AgentMessage) -> Dict[str, Any]:
        """Route message through hierarchical structure"""
        # Use supervisor routing for hierarchical communication
        return await super()._route_message_to_child(child_id, message)


# Factory function for creating orchestrators
def create_orchestrator(
    pattern: OrchestrationPattern,
    agent_id: str,
    **kwargs
) -> AgentOrchestrationBase:
    """Factory function to create appropriate orchestrator"""

    orchestrator_classes = {
        OrchestrationPattern.SUPERVISOR: SupervisorOrchestrator,
        OrchestrationPattern.SCATTER_GATHER: ScatterGatherOrchestrator,
        OrchestrationPattern.PIPELINE: PipelineOrchestrator,
        OrchestrationPattern.EVENT_DRIVEN: EventDrivenOrchestrator,
        OrchestrationPattern.HIERARCHICAL: HierarchicalOrchestrator,
    }

    orchestrator_class = orchestrator_classes.get(pattern)
    if not orchestrator_class:
        raise ValueError(f"Unsupported orchestration pattern: {pattern}")

    return orchestrator_class(agent_id=agent_id, **kwargs)
