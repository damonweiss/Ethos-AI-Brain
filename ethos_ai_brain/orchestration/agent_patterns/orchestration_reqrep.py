"""
REQ-REP Agent Orchestration Pattern
AgentZero implementation with child agent spawning via REQ-REP communication
"""

import asyncio
import uuid
from typing import Dict, Any, List
from orchestration_base import AgentOrchestrationBase, OrchestrationPattern, AgentMessage
from ..zmq_integration.zmq_node_base import ProcessorInterface, ProcessingRequest, ProcessorType


class ReqRepOrchestrator(AgentOrchestrationBase):
    """
    REQ-REP Orchestration Pattern

    Purpose:
    - Parent agent spawns child agents in separate ZMQ nodes
    - Communication via REQ-REP (parent sends requests, children respond)
    - Parent coordinates task distribution and result collection
    - Each child agent is a full Agent (LLM + MCP + sub-agent spawning capability)

    Use Cases:
    - Task delegation to specialized agents
    - Sequential processing with coordination
    - Master-worker pattern with intelligent workers
    """

    def __init__(self, agent_id: str, **kwargs):
        super().__init__(
            agent_id=agent_id,
            orchestration_pattern=OrchestrationPattern.SUPERVISOR,  # REQ-REP is supervisor-style
            **kwargs
        )

        # REQ-REP specific infrastructure
        self.child_req_clients = {}  # child_id -> REQ client for sending requests
        self.rep_server = None  # REP server for receiving child responses

    async def _initialize_orchestration(self):
        """Initialize REQ-REP orchestration infrastructure"""
        print(f"🔧 Initializing REQ-REP orchestration for {self.agent_id}")

        # Create REP server for child communication
        try:
            self.rep_server = self.zmq_engine.create_and_start_server(
                "reqrep", f"{self.agent_id}_coordinator"
            )

            if self.rep_server:
                # Register handlers for child communication
                self.rep_server.register_handler("child_task", self._handle_child_task_response)
                self.rep_server.register_handler("child_status", self._handle_child_status)
                print(f"✅ REP server created for {self.agent_id}")
            else:
                print(f"❌ Failed to create REP server for {self.agent_id}")

        except Exception as e:
            print(f"❌ REQ-REP initialization failed: {e}")

    async def _start_orchestration(self):
        """Start REQ-REP orchestration logic"""
        print(f"🚀 REQ-REP orchestrator {self.agent_id} ready")
        print(f"   Child agents: {len(self.child_agents)}")
        print(f"   REP server: {'✅' if self.rep_server else '❌'}")

    async def spawn_child_agent(self, child_id: str, role: str, processors: List[ProcessorInterface] = None) -> bool:
        """
        Spawn a new child agent in its own ZMQ node

        Args:
            child_id: Unique identifier for the child agent
            role: Role/specialization of the child agent
            processors: Processors to include in child's node

        Returns:
            bool: True if child spawned successfully
        """
        try:
            print(f"🐣 Spawning child agent {child_id} with role '{role}'")

            # Create child agent with its own ZMQ node
            # Note: In real implementation, this would create a separate process/container
            # For now, we'll create it in the same process but with separate ZMQ infrastructure

            from orchestration_base import AgentOrchestrationBase

            # Calculate unique port for child (ensure no conflicts)
            child_port = self.base_port + (len(self.child_agents) + 1) * 200 + 1000

            # Create child agent (this would be in separate process in production)
            child_agent = ReqRepChildAgent(
                agent_id=child_id,
                role=role,
                parent_agent=self,
                base_port=child_port,
                processors=processors or [],
                shared_zmq_engine=self.zmq_engine  # Share parent's ZMQ engine
            )

            # Start child agent
            await child_agent.start()

            # Add to our registry
            self.add_child_agent(child_agent)

            # Create REQ client for communicating with this child
            # Note: In production, this would connect to child's actual network address
            child_req_client = self._create_req_client_for_child(child_id, child_port)
            self.child_req_clients[child_id] = child_req_client

            print(f"✅ Child agent {child_id} spawned successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to spawn child agent {child_id}: {e}")
            return False

    def _create_req_client_for_child(self, child_id: str, child_port: int):
        """Create REQ client for communicating with specific child"""
        # In production, this would create actual ZMQ REQ client
        # For now, return a mock client
        return {
            "child_id": child_id,
            "endpoint": f"tcp://localhost:{child_port + 1}",
            "connected": True
        }

    async def delegate_task_to_child(self, child_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delegate a task to a specific child agent via REQ-REP

        Args:
            child_id: ID of child agent to send task to
            task: Task data to send

        Returns:
            Dict: Response from child agent
        """
        if child_id not in self.child_agents:
            return {"error": f"Child agent {child_id} not found"}

        if child_id not in self.child_req_clients:
            return {"error": f"No REQ client for child {child_id}"}

        try:
            print(f"📤 Delegating task to child {child_id}: {task.get('type', 'unknown')}")

            # Create task message
            task_message = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=child_id,
                message_type="task_request",
                content={"task": task}
            )

            # Send via REQ-REP (in production, this would use actual ZMQ)
            # For now, directly call child's task handler
            child_agent = self.child_agents[child_id]
            response = await child_agent._handle_task_request(task_message)

            print(f"📥 Received response from child {child_id}: {response.get('status', 'unknown')}")
            return response

        except Exception as e:
            print(f"❌ Task delegation to {child_id} failed: {e}")
            return {"error": str(e)}

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute task using REQ-REP orchestration

        Strategy:
        1. Analyze task to determine if delegation is needed
        2. Route to appropriate child agent(s) or handle locally
        3. Coordinate responses and return result
        """
        task_type = task.get("type", "general")

        print(f"🎯 REQ-REP orchestrator executing task: {task_type}")

        # Strategy 1: If we have processors locally, try local execution first
        if self.node.processors:
            try:
                # Use local processors for analysis
                analysis_request = ProcessingRequest(
                    request_id=str(uuid.uuid4()),
                    processor_type=ProcessorType.LLM,  # Try LLM first
                    operation="task_analysis",
                    input_data={
                        "task": task,
                        "context": {
                            "orchestrator": "reqrep",
                            "available_children": list(self.child_agents.keys())
                        }
                    }
                )

                analysis_response = await self.process_with_node(analysis_request)

                if analysis_response.status == "success":
                    analysis = analysis_response.output_data

                    # Decide based on analysis
                    if analysis.get("requires_delegation", False):
                        return await self._delegate_based_on_analysis(task, analysis)
                    else:
                        return await self._handle_locally(task, analysis)

            except Exception as e:
                print(f"⚠️ Local analysis failed, trying delegation: {e}")

        # Strategy 2: Delegate to first available child
        if self.child_agents:
            first_child = list(self.child_agents.keys())[0]
            return await self.delegate_task_to_child(first_child, task)

        # Strategy 3: Handle with basic logic
        return {
            "status": "completed",
            "result": f"REQ-REP orchestrator {self.agent_id} handled task: {task_type}",
            "method": "basic_handling"
        }

    async def _delegate_based_on_analysis(self, task: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate task based on LLM analysis"""
        recommended_child = analysis.get("recommended_child")

        if recommended_child and recommended_child in self.child_agents:
            return await self.delegate_task_to_child(recommended_child, task)
        else:
            # Delegate to first available child
            if self.child_agents:
                first_child = list(self.child_agents.keys())[0]
                return await self.delegate_task_to_child(first_child, task)

        return {"error": "No suitable child agent found"}

    async def _handle_locally(self, task: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task locally using our processors"""
        return {
            "status": "completed",
            "result": f"Handled locally by {self.agent_id}",
            "analysis": analysis,
            "method": "local_processing"
        }

    async def _route_message_to_child(self, child_id: str, message: AgentMessage) -> Dict[str, Any]:
        """Route message to child via REQ-REP"""
        return await self.delegate_task_to_child(child_id, message.content)

    async def _handle_child_task_response(self, message_data):
        """Handle responses from child agents"""
        # This would process responses in production REQ-REP setup
        return {"status": "acknowledged"}

    async def _handle_child_status(self, message_data):
        """Handle status updates from child agents"""
        return {"status": "status_received"}


class ReqRepChildAgent(AgentOrchestrationBase):
    """
    Child agent in REQ-REP pattern

    Purpose:
    - Receives tasks from parent via REQ-REP
    - Processes tasks using its processors (LLM, MCP, Custom Logic)
    - Can spawn its own child agents (recursive AgentZero architecture)
    - Sends responses back to parent
    """

    def __init__(self, agent_id: str, role: str, **kwargs):
        super().__init__(
            agent_id=agent_id,
            orchestration_pattern=OrchestrationPattern.NETWORK,  # Child uses network pattern
            **kwargs
        )
        self.role = role
        self.req_client = None  # For sending requests to parent
        self.rep_server = None  # For receiving requests from parent

    async def _initialize_orchestration(self):
        """Initialize child agent REQ-REP communication"""
        print(f"🔧 Initializing child agent {self.agent_id} (role: {self.role})")

        # Create REP server for receiving tasks from parent
        try:
            self.rep_server = self.zmq_engine.create_and_start_server(
                "reqrep", f"{self.agent_id}_worker"
            )

            if self.rep_server:
                self.rep_server.register_handler("task_request", self._handle_parent_task)
                print(f"✅ Child {self.agent_id} REP server ready")

        except Exception as e:
            print(f"❌ Child {self.agent_id} initialization failed: {e}")

    async def _start_orchestration(self):
        """Start child agent orchestration"""
        print(f"🚀 Child agent {self.agent_id} ({self.role}) ready")
        print(f"   Processors: {len(self.node.processors)}")

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task as child agent"""
        task_type = task.get("type", "general")

        print(f"🎯 Child {self.agent_id} ({self.role}) executing: {task_type}")

        # Use our processors to handle the task
        if self.node.processors:
            # Try to process with available processors
            for processor in self.node.processors.values():
                if not processor.is_busy:
                    try:
                        # For custom logic processors, pass the task data directly
                        if processor.processor_type.value == "custom_logic":
                            request = ProcessingRequest(
                                request_id=str(uuid.uuid4()),
                                processor_type=processor.processor_type,
                                operation="execute",
                                input_data={
                                    "args": [task.get("data", {})],  # Pass task data as args
                                    "kwargs": {}
                                }
                            )
                        else:
                            request = ProcessingRequest(
                                request_id=str(uuid.uuid4()),
                                processor_type=processor.processor_type,
                                operation="process_task",
                                input_data={"task": task, "agent_role": self.role}
                            )

                        response = await processor.process(request)

                        if response.status == "success":
                            # Extract the actual result from the processor
                            result_data = response.output_data.get("result", response.output_data)

                            return {
                                "status": "completed",
                                "result": result_data,
                                "agent_id": self.agent_id,
                                "agent_role": self.role,
                                "processor_used": processor.processor_type.value
                            }
                    except Exception as e:
                        print(f"⚠️ Processor {processor.processor_type.value} failed: {e}")
                        continue

        # Fallback: basic handling
        return {
            "status": "completed",
            "result": f"Child agent {self.agent_id} ({self.role}) handled task: {task_type}",
            "method": "basic_child_handling"
        }

    async def _route_message_to_child(self, child_id: str, message: AgentMessage) -> Dict[str, Any]:
        """Child agents can also have their own children (recursive AgentZero)"""
        if child_id in self.child_agents:
            return await self.child_agents[child_id]._handle_task_request(message)
        return {"error": f"Child {child_id} not found"}

    async def _handle_parent_task(self, message_data):
        """Handle task from parent agent"""
        # In production, this would parse ZMQ message
        # For now, assume it's already a proper task
        return await self.execute_task(message_data)
