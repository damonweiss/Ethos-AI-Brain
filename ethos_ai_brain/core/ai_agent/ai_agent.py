"""
AI Agent - Individual Agent Implementation
AgentZero and other agents are instances of this class
"""

from typing import Dict, Any, Optional, List
from .ai_brain import AIBrain
from ...knowledge import KnowledgeGraph, GraphType


class AIAgent:
    """Individual AI Agent - stub implementation"""

    def __init__(
            self,
            agent_id: str,
            role: str,
            zmq_engine: Any = None,
            base_port: int = 7000,
            capabilities: list = None
    ):
        self.agent_id = agent_id
        self.role = role
        self.zmq_engine = zmq_engine
        self.base_port = base_port
        self.capabilities = capabilities or []
        self.is_running = False

        # Initialize AI Brain
        self.brain = AIBrain(
            brain_name=f"{agent_id}_brain",
            zmq_engine=zmq_engine
        )

    async def start(self):
        """Start the agent"""
        self.is_running = True

    async def stop(self):
        """Stop the agent"""
        self.is_running = False

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task - stub implementation"""
        return {
            "status": "completed",
            "result": f"Agent {self.agent_id} processed task: {task.get('type', 'unknown')}"
        }

    def spawn_agent_pattern(self, pattern_class, pattern_id: str, **kwargs) -> Any:
        """
        Spawn an orchestration pattern for this agent

        Args:
            pattern_class: The orchestration pattern class (AgentOrchestrationBase subclass)
            pattern_id: Unique ID for the pattern instance
            **kwargs: Additional arguments for pattern initialization

        Returns:
            Instance of the orchestration pattern
        """
        # Create pattern instance with ZMQ engine
        pattern_instance = pattern_class(
            agent_id=pattern_id,
            zmq_engine=self.zmq_engine,
            base_port=self.base_port + 100,  # Offset pattern ports
            **kwargs
        )

        return pattern_instance

    # Public API (External Interface)
    def analyze_situation(self) -> Dict[str, Any]:
        """Comprehensive analysis of current knowledge state"""
        # Use knowledge interface directly - works for LayeredKG, basic KG will need different approach
        if hasattr(self.brain.knowledge, 'run_comprehensive_analysis'):
            return self.brain.knowledge.run_comprehensive_analysis()
        return {"status": "analysis_not_available", "knowledge_type": self.brain.knowledge.knowledge_type}

    def find_bottlenecks(self) -> List[tuple]:
        """Find potential bottlenecks in knowledge/processes"""
        if hasattr(self.brain.knowledge, 'detect_bottlenecks'):
            return self.brain.knowledge.detect_bottlenecks()
        return []

    def get_knowledge_status(self) -> Dict[str, Any]:
        """Get current knowledge and network status"""
        if hasattr(self.brain.knowledge, 'get_unified_network_info'):
            return self.brain.knowledge.get_unified_network_info()
        return {
            "knowledge_id": self.brain.knowledge.knowledge_id,
            "knowledge_type": self.brain.knowledge.knowledge_type,
            "capabilities": self.brain.knowledge.get_capabilities()
        }

    def find_critical_tasks(self) -> List[str]:
        """Find critical or active tasks across knowledge domains"""
        if hasattr(self.brain.knowledge, 'find_critical_path_nodes'):
            return self.brain.knowledge.find_critical_path_nodes()
        return []

    def query_knowledge(self, query: str = "", **criteria) -> Dict[str, Any]:
        """Query across all knowledge domains"""
        results = self.brain.knowledge.query(query, **criteria)
        return {"results": [r.to_dict() for r in results], "count": len(results)}

    # async def think_about(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    #     """Ask the agent to think about a goal using meta-reasoning"""
    #     return await self.brain.think(goal, context)

    async def analyze_user_request(self, user_prompt: str) -> tuple:
        """Analyze user request using adaptive intent analysis"""
        # TODO: Implement when adaptive_intent_runner is available
        # result_type, result_data = await self.brain.adaptive_intent_runner.analyze_intent(user_prompt)
        
        # Placeholder implementation
        return "simple_analysis", {
            "prompt": user_prompt,
            "status": "placeholder_response",
            "message": "Adaptive intent runner not yet implemented"
        }

        # Future implementation when reasoning engines are available:
        # if result_type == "complex_analysis":
        #     if isinstance(result_data, dict) and 'graph' in result_data:
        #         intent_graph = result_data['graph']
        #         
        #         # Convert to KnowledgeGraph and add to brain
        #         kg = KnowledgeGraph(
        #             graph_id=f"intent_analysis_{len(self.brain.knowledge.get_components())}",
        #             graph_type=GraphType.INTENT
        #         )
        #         
        #         # Copy nodes and edges from intent graph
        #         for node_id in intent_graph.nodes():
        #             node_attrs = intent_graph.get_node_attributes(node_id)
        #             kg.add_node(node_id, **node_attrs)
        #         
        #         for edge in intent_graph.edges():
        #             source, target = edge
        #             edge_attrs = intent_graph.get_edge_attributes(source, target)
        #             kg.add_edge(source, target, **edge_attrs)
        #         
        #         # Add to brain's knowledge system (if it supports layers)
        #         if hasattr(self.brain.knowledge, 'add_layer'):
        #             self.brain.knowledge.add_layer(kg)
        #         else:
        #             # For basic KG, would need different approach
        #             pass
