"""
AI Agent - Individual Agent Implementation
AgentZero and other agents are instances of this class
"""

from typing import Dict, Any, Optional, List
from ai_brain import AIBrain
from knowledge_graph import KnowledgeGraph, GraphType


class AIAgent:
    """Individual AI Agent - stub implementation"""
    
    def __init__(
        self,
        agent_id: str,
        role: str,
        shared_zmq_engine: Any = None,
        base_port: int = 7000,
        capabilities: list = None
    ):
        self.agent_id = agent_id
        self.role = role
        self.shared_zmq_engine = shared_zmq_engine
        self.base_port = base_port
        self.capabilities = capabilities or []
        self.is_running = False
        
        # Initialize AI Brain
        self.brain = AIBrain(
            brain_id=f"{agent_id}_brain",
            shared_zmq_engine=shared_zmq_engine
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
        # Create pattern instance with shared ZMQ engine
        pattern_instance = pattern_class(
            agent_id=pattern_id,
            shared_zmq_engine=self.shared_zmq_engine,
            base_port=self.base_port + 100,  # Offset pattern ports
            **kwargs
        )
        
        return pattern_instance
    
    # Public API (External Interface)
    def analyze_situation(self) -> Dict[str, Any]:
        """Comprehensive analysis of current knowledge state"""
        return self.brain.knowledge_graph_brain.run_comprehensive_analysis()
    
    def find_bottlenecks(self) -> List[tuple]:
        """Find potential bottlenecks in knowledge/processes"""
        return self.brain.knowledge_graph_brain.detect_bottlenecks()
    
    def get_knowledge_status(self) -> Dict[str, Any]:
        """Get current knowledge and network status"""
        return self.brain.knowledge_graph_brain.get_unified_network_info()
    
    def find_critical_tasks(self) -> List[str]:
        """Find critical or active tasks across knowledge domains"""
        return self.brain.knowledge_graph_brain.find_critical_path_nodes()
    
    def query_knowledge(self, **criteria) -> Dict[str, Any]:
        """Query across all knowledge domains with criteria"""
        return self.brain.knowledge_graph_brain.query_across_graphs(**criteria)
    
    async def think_about(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Ask the agent to think about a goal using meta-reasoning"""
        return await self.brain.think(goal, context)
    
    async def analyze_user_request(self, user_prompt: str) -> tuple:
        """Analyze user request using adaptive intent analysis"""
        result_type, result_data = await self.brain.adaptive_intent_runner.analyze_intent(user_prompt)
        
        # If we got a complex analysis with a knowledge graph, add it to the brain
        if result_type == "complex_analysis":
            if isinstance(result_data, dict) and 'graph' in result_data:
                intent_graph = result_data['graph']
                
                # Convert to KnowledgeGraph and add to brain
                kg = KnowledgeGraph(
                    graph_type=GraphType.INTENT,
                    graph_id=f"intent_analysis_{len(self.brain.knowledge_graph_brain.graphs)}"
                )
                
                # Copy nodes and edges from intent graph
                for node_id in intent_graph.nodes():
                    node_attrs = intent_graph.get_node_attributes(node_id)
                    kg.add_node(node_id, **node_attrs)
                
                for edge in intent_graph.edges():
                    source, target = edge
                    edge_attrs = intent_graph.get_edge_attributes(source, target)
                    kg.add_edge(source, target, **edge_attrs)
                
                # Add to brain's knowledge graph manager
                self.brain.knowledge_graph_brain.add_graph(kg)
                print(f"📊 Added intent analysis graph to brain: {kg.graph_id}")
        
        return result_type, result_data
