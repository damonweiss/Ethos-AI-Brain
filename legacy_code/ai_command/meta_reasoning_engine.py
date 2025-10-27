#!/usr/bin/env python3
"""
Meta-Reasoning Engine - Agentic AI Cognitive Architecture
A comprehensive implementation of meta-reasoning capabilities for dtOS ecosystem.

Features:
- Multiple LLM backends with prompt orchestration
- Dynamic MCP tool loading and dependency chains
- Hierarchical memory with semantic search
- DAG-based planning with multi-agent handoffs
- Event-driven architecture with graceful interruption
"""

import asyncio
import json
import uuid
import os
from dotenv import load_dotenv
import networkx as nx
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable, Union, Tuple
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollaborationMode(Enum):
    APPROVAL = "approval"
    INPUT = "input"
    REVIEW = "review"
    OVERRIDE = "override"
    VOICE = "voice"

class ConfidenceLevel(Enum):
    HIGH = "high"      # >90% - autonomous execution
    MEDIUM = "medium"  # 70-90% - human review optional
    LOW = "low"        # <70% - human input required
    UNKNOWN = "unknown" # conflicting/insufficient data

class ExecutionMode(Enum):
    STEP_BY_STEP = "step_by_step"
    FULL_AUTO = "full_auto"
    CHECKPOINT = "checkpoint"

@dataclass
class ReasoningContext:
    """Context for reasoning operations"""
    goal: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolCapability:
    """Definition of a tool's capabilities"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    confidence_requirements: ConfidenceLevel = ConfidenceLevel.MEDIUM
    execution_time_estimate: float = 1.0  # seconds
    
@dataclass
class ReasoningStep:
    """Individual step in reasoning process"""
    step_id: str
    tool_name: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    confidence: Optional[ConfidenceLevel] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class HumanCollaborationRequest:
    """Request for human collaboration"""
    request_id: str
    mode: CollaborationMode
    context: str
    question: str
    options: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    is_blocking: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class LLMBackend(ABC):
    """Abstract base class for LLM backends"""
    
    @abstractmethod
    async def generate(self, prompt: str, context: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Generate streaming response from LLM"""
        pass
    
    @abstractmethod
    async def complete(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate complete response from LLM"""
        pass

class MockLLMBackend(LLMBackend):
    """Mock LLM backend for demonstration"""
    
    def __init__(self, name: str, personality: str = "helpful"):
        self.name = name
        self.personality = personality
        
    async def generate(self, prompt: str, context: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Mock streaming response"""
        response = await self.complete(prompt, context)
        words = response.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.05)  # Simulate streaming delay
            
    async def complete(self, prompt: str, context: Dict[str, Any]) -> str:
        """Mock complete response based on prompt patterns"""
        
        # LOG THE MOCK PROMPT AND CONTEXT
        print(f"\n🎭 [{self.name}] MOCK PROMPT:")
        print(f"📋 Personality: {self.personality}")
        print(f"👤 User: {prompt}")
        try:
            context_str = json.dumps(self._make_serializable(context), indent=2)[:300]
            print(f"🔧 Context: {context_str}...")
        except Exception as e:
            print(f"🔧 Context: {str(context)[:300]}... (JSON error: {e})")
        print("─" * 60)
        
        prompt_lower = prompt.lower()
        
        if "analyze" in prompt_lower:
            response = f"Mock Analysis: This appears to be a security-related request requiring multi-domain expertise. I recommend engaging security specialists and conducting thorough risk assessment."
            
        elif "assess" in prompt_lower:
            response = f"Mock Assessment: Based on complexity indicators, this is a moderate-level task (complexity: 2) requiring sequential analysis with security and architecture specialists."
            
        elif "plan" in prompt_lower:
            response = f"Mock Planning: Strategic execution plan: 1) Initial assessment phase (2 days), 2) Specialist consultation (3 days), 3) Solution design (5 days), 4) Implementation planning (2 days). Total timeline: 12 days."
            
        elif "decompose" in prompt_lower:
            response = f"Task Decomposition: Breaking this into subtasks: 1) Data collection, 2) Analysis and pattern recognition, 3) Solution generation, 4) Validation and testing, 5) Implementation planning."
            
        elif "confidence" in prompt_lower:
            response = f"Confidence Assessment: Based on available data quality and complexity, I assess confidence at 75% - recommend human review for final decision."
            
        else:
            response = f"I understand your request about: {prompt[:100]}... Let me help you with that using my {self.personality} approach."
        
        # LOG THE MOCK RESPONSE
        print(f"🎭 [{self.name}] MOCK RESPONSE:")
        print(f"💬 {response}")
        print("═" * 60)
        
        return f"[{self.name}] {response}"
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        if hasattr(obj, '__dict__'):
            # Convert objects with attributes to dictionaries
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):  # Skip private attributes
                    result[key] = self._make_serializable(value)
            return result
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'value'):  # Handle enums
            return obj.value
        elif hasattr(obj, 'name'):  # Handle enums with name attribute
            return obj.name
        else:
            try:
                json.dumps(obj)  # Test if it's already serializable
                return obj
            except (TypeError, ValueError):
                return str(obj)  # Convert to string as fallback

class OpenAIBackend(LLMBackend):
    """OpenAI GPT backend for real LLM integration"""
    
    def __init__(self, name: str, model: str = "gpt-4", personality: str = "helpful", system_prompt: str = None):
        self.name = name
        self.model = model
        self.personality = personality
        self.system_prompt = system_prompt or f"You are {name}, a {personality} AI assistant specialized in meta-reasoning and analysis."
        
        # Import OpenAI here to avoid dependency issues if not installed
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        except Exception as e:
            raise Exception(f"Failed to initialize OpenAI client: {e}")
            
    async def generate(self, prompt: str, context: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Generate streaming response from OpenAI"""
        try:
            # Convert context to JSON-serializable format
            serializable_context = self._make_serializable(context)
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context: {json.dumps(serializable_context, indent=2)}\n\nRequest: {prompt}"}
            ]
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                max_tokens=1000,
                temperature=0.7
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"[{self.name}] Error: {str(e)}"
            
    async def complete(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate complete response from OpenAI"""
        try:
            # Convert context to JSON-serializable format
            serializable_context = self._make_serializable(context)
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context: {json.dumps(serializable_context, indent=2)}\n\nRequest: {prompt}"}
            ]
            
            # LOG THE PROMPT BEING SENT
            print(f"\n🤖 [{self.name}] SENDING PROMPT:")
            print(f"📋 System: {self.system_prompt[:200]}...")
            print(f"👤 User: {prompt}")
            try:
                context_str = json.dumps(serializable_context, indent=2)[:300]
                print(f"🔧 Context: {context_str}...")
            except Exception as e:
                print(f"🔧 Context: {str(serializable_context)[:300]}... (JSON error: {e})")
            print("─" * 60)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # LOG THE RESPONSE RECEIVED
            print(f"🧠 [{self.name}] RECEIVED RESPONSE:")
            print(f"💬 {ai_response}")
            print("═" * 60)
            
            return f"[{self.name}] {ai_response}"
            
        except Exception as e:
            error_msg = f"[{self.name}] Error: {str(e)}"
            print(f"❌ [{self.name}] ERROR: {str(e)}")
            return error_msg
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        if hasattr(obj, '__dict__'):
            # Convert objects with attributes to dictionaries
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):  # Skip private attributes
                    result[key] = self._make_serializable(value)
            return result
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Convert other types to string representation
            return str(obj)

class MemoryNode:
    """Node in hierarchical memory graph"""
    
    def __init__(self, node_id: str, content: Any, node_type: str = "general"):
        self.node_id = node_id
        self.content = content
        self.node_type = node_type
        self.timestamp = datetime.now()
        self.access_count = 0
        self.connections: Dict[str, float] = {}  # node_id -> relevance_score
        self.metadata: Dict[str, Any] = {}

class HierarchicalMemory:
    """Hierarchical memory system with semantic search"""
    
    def __init__(self):
        self.working_memory: Dict[str, MemoryNode] = {}  # Current session
        self.short_term_memory: Dict[str, MemoryNode] = {}  # Recent sessions
        self.long_term_memory: Dict[str, MemoryNode] = {}  # Persistent knowledge
        self.memory_graph = nx.DiGraph()
        
    def store(self, content: Any, memory_type: str = "working", node_type: str = "general") -> str:
        """Store content in appropriate memory layer"""
        node_id = str(uuid.uuid4())
        node = MemoryNode(node_id, content, node_type)
        
        if memory_type == "working":
            self.working_memory[node_id] = node
        elif memory_type == "short_term":
            self.short_term_memory[node_id] = node
        else:
            self.long_term_memory[node_id] = node
            
        self.memory_graph.add_node(node_id, **node.metadata)
        return node_id
        
    def retrieve(self, query: str, memory_types: List[str] = None, limit: int = 5) -> List[MemoryNode]:
        """Semantic search across memory layers"""
        if memory_types is None:
            memory_types = ["working", "short_term", "long_term"]
            
        candidates = []
        
        for memory_type in memory_types:
            memory_dict = getattr(self, f"{memory_type}_memory")
            for node in memory_dict.values():
                # Simple semantic matching (would use embeddings in real implementation)
                relevance = self._calculate_relevance(query, node.content)
                if relevance > 0.3:  # Threshold
                    candidates.append((relevance, node))
                    
        # Sort by relevance and return top results
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in candidates[:limit]]
        
    def _calculate_relevance(self, query: str, content: Any) -> float:
        """Mock semantic relevance calculation"""
        query_words = set(query.lower().split())
        content_words = set(str(content).lower().split())
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        return len(intersection) / len(union) if union else 0.0

class MCPToolRegistry:
    """Registry for dynamically loaded MCP tools"""
    
    def __init__(self):
        self.tools: Dict[str, ToolCapability] = {}
        self.tool_instances: Dict[str, Any] = {}
        self.dependency_graph = nx.DiGraph()
        
    def register_tool(self, capability: ToolCapability, instance: Any):
        """Register a new MCP tool"""
        self.tools[capability.name] = capability
        self.tool_instances[capability.name] = instance
        
        # Add to dependency graph
        self.dependency_graph.add_node(capability.name)
        for dep in capability.dependencies:
            if dep in self.tools:
                self.dependency_graph.add_edge(dep, capability.name)
                
        logger.info(f"Registered tool: {capability.name}")
        
    def get_execution_order(self, tool_names: List[str]) -> List[str]:
        """Get topologically sorted execution order"""
        subgraph = self.dependency_graph.subgraph(tool_names)
        return list(nx.topological_sort(subgraph))
        
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a tool with given parameters"""
        if tool_name not in self.tool_instances:
            raise ValueError(f"Tool {tool_name} not found")
            
        tool = self.tool_instances[tool_name]
        
        # Mock tool execution (would call actual MCP in real implementation)
        await asyncio.sleep(0.1)  # Simulate execution time
        
        if tool_name == "data_analyzer":
            return {"analysis": "Data shows positive trend", "confidence": 0.85}
        elif tool_name == "plan_generator":
            return {"plan": ["Step 1: Analyze", "Step 2: Design", "Step 3: Implement"], "timeline": "2 weeks"}
        elif tool_name == "citizen_engagement":
            return {"response": "Thank you for your feedback. What specific aspects concern you most?", "engagement_score": 0.9}
        else:
            return {"result": f"Executed {tool_name} with {parameters}", "status": "success"}

class MetaReasoningEngine:
    """Main meta-reasoning engine class"""
    
    def __init__(self, use_real_ai: bool = False):
        self.llm_backends: Dict[str, LLMBackend] = {}
        self.memory = HierarchicalMemory()
        self.tool_registry = MCPToolRegistry()
        self.active_sessions: Dict[str, ReasoningContext] = {}
        self.collaboration_handlers: Dict[str, Callable] = {}
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Initialize LLM backends (real or mock)
        if use_real_ai:
            self._initialize_real_backends()
        else:
            self._initialize_mock_backends()
        self._initialize_mock_tools()
        
    def _initialize_real_backends(self):
        """Initialize real OpenAI LLM backends with specialized roles"""
        try:
            # Analyst - for situation assessment and analysis
            self.llm_backends["analyst"] = OpenAIBackend(
                name="Analyst",
                model="gpt-4",
                personality="analytical",
                system_prompt="""You are an expert strategic analyst. Your role is to:
- Assess situation complexity and requirements
- Identify key domains and interdependencies  
- Recommend optimal execution strategies
- Provide clear, structured analysis

Always be thorough but concise. Focus on actionable insights."""
            )
            
            # Planner - for strategic planning and decomposition
            self.llm_backends["planner"] = OpenAIBackend(
                name="Planner", 
                model="gpt-4",
                personality="strategic",
                system_prompt="""You are a strategic planner. Your role is to:
- Break down complex goals into actionable steps
- Create detailed execution plans with timelines
- Identify resource requirements and dependencies
- Anticipate risks and mitigation strategies

Focus on practical, implementable plans."""
            )
            
            # Decomposer - for task breakdown
            self.llm_backends["decomposer"] = OpenAIBackend(
                name="Decomposer",
                model="gpt-4",
                personality="systematic",
                system_prompt="""You are a task decomposition specialist. Your role is to:
- Break complex tasks into smaller, manageable components
- Identify task dependencies and sequencing
- Ensure each subtask is clearly defined and actionable
- Optimize for parallel execution where possible

Be systematic and thorough in your decomposition."""
            )
            
            logger.info("🤖 Real AI backends initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize real AI backends: {e}")
            logger.info("Falling back to mock backends")
            self._initialize_mock_backends()
    
    def _initialize_mock_backends(self):
        """Initialize mock LLM backends with different personalities"""
        self.llm_backends["analyst"] = MockLLMBackend("Analyst", "analytical")
        self.llm_backends["planner"] = MockLLMBackend("Planner", "strategic")
        self.llm_backends["citizen_engagement"] = MockLLMBackend("CitizenBot", "empathetic")
        self.llm_backends["decomposer"] = MockLLMBackend("Decomposer", "systematic")
        
    def _initialize_mock_tools(self):
        """Initialize mock MCP tools"""
        tools = [
            ToolCapability("data_analyzer", "Analyzes data patterns", {}, {}),
            ToolCapability("plan_generator", "Generates execution plans", {}, {}, ["data_analyzer"]),
            ToolCapability("citizen_engagement", "Engages with citizens", {}, {}),
            ToolCapability("confidence_assessor", "Assesses confidence levels", {}, {}),
        ]
        
        for tool in tools:
            self.tool_registry.register_tool(tool, f"mock_{tool.name}")
            
    def register_collaboration_handler(self, mode: CollaborationMode, handler: Callable):
        """Register handler for human collaboration requests"""
        self.collaboration_handlers[mode.value] = handler
        
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register event handler"""
        self.event_handlers[event_type].append(handler)
        
    def register_llm_backend(self, name: str, backend: LLMBackend):
        """Register or replace an LLM backend"""
        self.llm_backends[name] = backend
        logger.info(f"Registered LLM backend: {name} ({backend.__class__.__name__})")
        
    async def emit_event(self, event_type: str, data: Any):
        """Emit event to registered handlers"""
        for handler in self.event_handlers[event_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
                
    async def reason(self, goal: str, context: ReasoningContext = None) -> Dict[str, Any]:
        """Main reasoning entry point"""
        if context is None:
            context = ReasoningContext(goal=goal)
            
        self.active_sessions[context.session_id] = context
        
        try:
            # Phase 1: Goal analysis and decomposition
            await self.emit_event("reasoning_started", {"session_id": context.session_id, "goal": goal})
            
            decomposition = await self._decompose_goal(goal, context)
            
            # Phase 2: Plan generation
            plan = await self._generate_plan(decomposition, context)
            
            # Phase 3: Execution with monitoring
            results = await self._execute_plan(plan, context)
            
            # Phase 4: Result synthesis
            final_result = await self._synthesize_results(results, context)
            
            await self.emit_event("reasoning_completed", {"session_id": context.session_id, "result": final_result})
            
            return final_result
            
        except Exception as e:
            logger.error(f"Reasoning error: {e}")
            await self.emit_event("reasoning_error", {"session_id": context.session_id, "error": str(e)})
            raise
        finally:
            # Clean up session
            if context.session_id in self.active_sessions:
                del self.active_sessions[context.session_id]
                
    async def _decompose_goal(self, goal: str, context: ReasoningContext) -> Dict[str, Any]:
        """Decompose high-level goal into actionable components"""
        prompt = f"""
        Decompose this goal into actionable components:
        Goal: {goal}
        Context: {context.constraints}
        
        Provide a structured breakdown with dependencies and confidence levels.
        """
        
        llm = self.llm_backends["decomposer"]
        response = await llm.complete(prompt, {"goal": goal})
        
        # Store in memory
        memory_id = self.memory.store(response, "working", "decomposition")
        
        return {
            "decomposition": response,
            "memory_id": memory_id,
            "confidence": ConfidenceLevel.MEDIUM
        }
        
    async def _generate_plan(self, decomposition: Dict[str, Any], context: ReasoningContext) -> Dict[str, Any]:
        """Generate DAG-based execution plan"""
        prompt = f"""
        Generate an execution plan based on this decomposition:
        {decomposition['decomposition']}
        
        Create a DAG with parallel execution opportunities and checkpoints.
        """
        
        llm = self.llm_backends["planner"]
        response = await llm.complete(prompt, decomposition)
        
        # Create execution DAG
        execution_graph = nx.DiGraph()
        steps = [
            ReasoningStep("step_1", "data_analyzer", {"data": "context"}),
            ReasoningStep("step_2", "confidence_assessor", {"input": "step_1"}, ["step_1"]),
            ReasoningStep("step_3", "plan_generator", {"analysis": "step_1"}, ["step_1"]),
        ]
        
        for step in steps:
            execution_graph.add_node(step.step_id, step=step)
            for dep in step.dependencies:
                execution_graph.add_edge(dep, step.step_id)
                
        return {
            "plan": response,
            "execution_graph": execution_graph,
            "steps": {step.step_id: step for step in steps}
        }
        
    async def _execute_plan(self, plan: Dict[str, Any], context: ReasoningContext) -> Dict[str, Any]:
        """Execute plan with checkpoints and human collaboration"""
        execution_graph = plan["execution_graph"]
        steps = plan["steps"]
        results = {}
        
        # Execute in topological order
        for step_id in nx.topological_sort(execution_graph):
            step = steps[step_id]
            
            try:
                # Check if human collaboration needed
                if await self._needs_human_collaboration(step, context):
                    collaboration_result = await self._request_human_collaboration(step, context)
                    if collaboration_result.get("override"):
                        step.parameters.update(collaboration_result["parameters"])
                        
                # Execute step
                await self.emit_event("step_started", {"step_id": step_id, "step": step})
                
                start_time = datetime.now()
                result = await self.tool_registry.execute_tool(step.tool_name, step.parameters)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                step.result = result
                step.execution_time = execution_time
                step.confidence = self._assess_confidence(result)
                
                results[step_id] = step
                
                await self.emit_event("step_completed", {"step_id": step_id, "result": result})
                
                # Store in memory
                self.memory.store(step, "working", "execution_step")
                
            except Exception as e:
                step.error = str(e)
                logger.error(f"Step {step_id} failed: {e}")
                
                # Try fallback strategies
                fallback_result = await self._handle_step_failure(step, context)
                if fallback_result:
                    step.result = fallback_result
                    results[step_id] = step
                else:
                    raise
                    
        return results
        
    async def _needs_human_collaboration(self, step: ReasoningStep, context: ReasoningContext) -> bool:
        """Determine if human collaboration is needed for this step"""
        # Check confidence requirements
        if step.confidence and step.confidence == ConfidenceLevel.LOW:
            return True
            
        # Check user preferences
        if context.user_preferences.get("require_approval", False):
            return True
            
        # Check tool requirements
        tool_capability = self.tool_registry.tools.get(step.tool_name)
        if tool_capability and tool_capability.confidence_requirements == ConfidenceLevel.LOW:
            return True
            
        return False
        
    async def _request_human_collaboration(self, step: ReasoningStep, context: ReasoningContext) -> Dict[str, Any]:
        """Request human collaboration with graceful interruption"""
        request = HumanCollaborationRequest(
            request_id=str(uuid.uuid4()),
            mode=CollaborationMode.REVIEW,
            context=f"Executing step: {step.tool_name}",
            question=f"Should I proceed with {step.tool_name} using parameters: {step.parameters}?",
            options=["Proceed", "Modify", "Skip", "Cancel"],
            timeout=30.0,
            is_blocking=True
        )
        
        # Mock human response (would integrate with actual UI in real implementation)
        await asyncio.sleep(0.5)  # Simulate human thinking time
        
        return {
            "decision": "Proceed",
            "override": False,
            "parameters": {},
            "feedback": "Looks good, proceed as planned"
        }
        
    def _assess_confidence(self, result: Any) -> ConfidenceLevel:
        """Assess confidence level of result"""
        if isinstance(result, dict) and "confidence" in result:
            confidence_score = result["confidence"]
            if confidence_score > 0.9:
                return ConfidenceLevel.HIGH
            elif confidence_score > 0.7:
                return ConfidenceLevel.MEDIUM
            else:
                return ConfidenceLevel.LOW
        return ConfidenceLevel.MEDIUM
        
    async def _handle_step_failure(self, step: ReasoningStep, context: ReasoningContext) -> Optional[Any]:
        """Handle step failure with hierarchical fallback"""
        logger.info(f"Handling failure for step: {step.step_id}")
        
        # Try alternative tools
        alternative_tools = ["confidence_assessor", "data_analyzer"]  # Mock alternatives
        for alt_tool in alternative_tools:
            if alt_tool != step.tool_name and alt_tool in self.tool_registry.tools:
                try:
                    result = await self.tool_registry.execute_tool(alt_tool, step.parameters)
                    logger.info(f"Fallback successful with {alt_tool}")
                    return result
                except Exception:
                    continue
                    
        # Escalate to human
        collaboration_request = HumanCollaborationRequest(
            request_id=str(uuid.uuid4()),
            mode=CollaborationMode.OVERRIDE,
            context=f"Step {step.step_id} failed",
            question="How should I proceed?",
            options=["Retry", "Skip", "Manual Override", "Cancel"],
            is_blocking=True
        )
        
        # Mock escalation response
        return {"manual_override": True, "result": "Human provided alternative solution"}
        
    async def _synthesize_results(self, results: Dict[str, ReasoningStep], context: ReasoningContext) -> Dict[str, Any]:
        """Synthesize execution results into final answer"""
        prompt = f"""
        Synthesize these execution results into a comprehensive response:
        Goal: {context.goal}
        Results: {[step.result for step in results.values()]}
        
        Provide a clear, actionable summary with confidence assessment.
        """
        
        llm = self.llm_backends["analyst"]
        synthesis = await llm.complete(prompt, {"results": results})
        
        return {
            "synthesis": synthesis,
            "execution_summary": {
                "total_steps": len(results),
                "successful_steps": len([s for s in results.values() if s.result and not s.error]),
                "failed_steps": len([s for s in results.values() if s.error]),
                "total_execution_time": sum(s.execution_time or 0 for s in results.values())
            },
            "confidence": self._calculate_overall_confidence(results),
            "memory_references": [s.step_id for s in results.values()],
            "session_id": context.session_id
        }
        
    def _calculate_overall_confidence(self, results: Dict[str, ReasoningStep]) -> ConfidenceLevel:
        """Calculate overall confidence from step results"""
        confidences = [s.confidence for s in results.values() if s.confidence]
        if not confidences:
            return ConfidenceLevel.MEDIUM
            
        # Simple averaging (would use more sophisticated logic in real implementation)
        high_count = sum(1 for c in confidences if c == ConfidenceLevel.HIGH)
        total_count = len(confidences)
        
        if high_count / total_count > 0.8:
            return ConfidenceLevel.HIGH
        elif high_count / total_count > 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

# Example usage and testing
async def main():
    """Example usage of the Meta-Reasoning Engine with OpenAI integration"""
    engine = MetaReasoningEngine()
    
    # Setup hybrid LLM backends (OpenAI + Mock)
    print("🔧 Setting up LLM backends...")
    
    try:
        # Try to use OpenAI for critical reasoning tasks
        openai_analyst = OpenAIBackend(
            name="GPT4-Analyst", 
            model="gpt-4",
            personality="analytical",
            system_prompt="You are an expert analyst specializing in breaking down complex problems and providing structured insights. Focus on data-driven analysis and clear reasoning."
        )
        
        openai_planner = OpenAIBackend(
            name="GPT4-Planner",
            model="gpt-4", 
            personality="strategic",
            system_prompt="You are a strategic planning expert. Create detailed, actionable plans with clear dependencies and risk assessments."
        )
        
        # Register OpenAI backends
        engine.register_llm_backend("analyst", openai_analyst)
        engine.register_llm_backend("planner", openai_planner)
        print("✅ OpenAI backends registered successfully")
        
    except Exception as e:
        print(f"⚠️  OpenAI setup failed: {e}")
        print("🔄 Falling back to mock backends...")
        
        # Fallback to mock backends
        engine.register_llm_backend("analyst", MockLLMBackend("MockAnalyst", "analytical"))
        engine.register_llm_backend("planner", MockLLMBackend("MockPlanner", "strategic"))
    
    # Always use mock for specialized tasks (for demonstration)
    engine.register_llm_backend("decomposer", MockLLMBackend("TaskDecomposer", "systematic"))
    engine.register_llm_backend("citizen_engagement", MockLLMBackend("CitizenEngager", "empathetic"))
    
    print(f"🧠 Engine ready with {len(engine.llm_backends)} LLM backends")
    
    # Register event handlers
    async def on_reasoning_started(data):
        print(f"🧠 Reasoning started for session: {data['session_id']}")
        print(f"📋 Goal: {data['goal']}")
        
    async def on_step_completed(data):
        print(f"✅ Step completed: {data['step_id']}")
        print(f"📊 Result: {data['result']}")
        
    engine.register_event_handler("reasoning_started", on_reasoning_started)
    engine.register_event_handler("step_completed", on_step_completed)
    
    # Example 1: Public meeting citizen engagement
    print("=" * 60)
    print("EXAMPLE 1: Public Meeting Citizen Engagement")
    print("=" * 60)
    
    context = ReasoningContext(
        goal="Engage with citizens at public meeting to answer 10 key questions about infrastructure project",
        constraints={
            "questions": [
                "How will this affect traffic?",
                "What's the environmental impact?",
                "How much will it cost taxpayers?"
            ],
            "engagement_strategy": "empathetic_listening",
            "time_limit": 30  # minutes
        },
        user_preferences={
            "require_approval": False,
            "collaboration_mode": "voice"
        }
    )
    
    result1 = await engine.reason(
        "Engage with citizens about infrastructure project",
        context
    )
    
    print("\n🎯 Final Result:")
    print(f"Synthesis: {result1['synthesis']}")
    print(f"Confidence: {result1['confidence']}")
    print(f"Execution Time: {result1['execution_summary']['total_execution_time']:.2f}s")
    
    # Example 2: Complex planning with agent handoffs
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Multi-Agent Planning with Handoffs")
    print("=" * 60)
    
    context2 = ReasoningContext(
        goal="Develop comprehensive digital twin implementation strategy for smart city",
        constraints={
            "budget": 2000000,
            "timeline": "18 months",
            "stakeholders": ["city_council", "citizens", "tech_vendors"],
            "requirements": ["real_time_monitoring", "predictive_analytics", "public_dashboard"]
        },
        user_preferences={
            "execution_mode": "step_by_step",
            "require_checkpoints": True
        }
    )
    
    result2 = await engine.reason(
        "Develop smart city digital twin strategy",
        context2
    )
    
    print("\n🎯 Final Result:")
    print(f"Synthesis: {result2['synthesis']}")
    print(f"Steps Executed: {result2['execution_summary']['successful_steps']}/{result2['execution_summary']['total_steps']}")
    print(f"Overall Confidence: {result2['confidence']}")
    
    # Example 3: Memory retrieval and learning
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Memory Retrieval and Learning")
    print("=" * 60)
    
    # Query memory for relevant past experiences
    relevant_memories = engine.memory.retrieve("citizen engagement strategy")
    print(f"📚 Found {len(relevant_memories)} relevant memories")
    for memory in relevant_memories:
        print(f"  - {memory.node_type}: {str(memory.content)[:100]}...")
        
    print("\n🧠 Meta-Reasoning Engine demonstration complete!")

async def llm_debate_demo():
    """Demonstrate LLM-vs-LLM debate with reasoning and turn-taking"""
    print("🥊 LLM DEBATE ARENA - 3 ROUNDS")
    print("=" * 60)
    
    # Create two opposing LLM debaters with distinct personalities
    try:
        def _conduct_debate_round(self, round_num: int, topic: str, progressive_context: str, traditional_context: str) -> Tuple[str, str]:
            """Conduct a single round of debate between two LLM personalities."""
            
            # Progressive AI's turn
            progressive_prompt = f"""You are Progressive-AI, a forward-thinking AI that advocates for innovation, change, and progressive policies.

Topic: {topic}
Round: {round_num}/3

Previous context:
{progressive_context}

Please provide your argument for this round. Be thoughtful, cite reasoning, and acknowledge valid points from your opponent while maintaining your progressive stance.

Format your response EXACTLY as:
**Thoughts:** [Your internal reasoning, analysis of opponent's points, strategy for this round]

**Progressive-AI Round {round_num}:** [Your final argument here]
"""
            
            progressive_response = self.llm_backends["analytical"].complete(
                system_prompt="You are Progressive-AI, advocating for progressive policies with thoughtful reasoning. Always show your thoughts before your final argument.",
                user_prompt=progressive_prompt,
                context={}
            )
            
            # Traditional AI's turn
            traditional_prompt = f"""You are Traditional-AI, a conservative AI that advocates for traditional values, stability, and cautious change.

Topic: {topic}
Round: {round_num}/3

Previous context:
{traditional_context}

Progressive-AI just argued: {progressive_response}

Please provide your counter-argument for this round. Be thoughtful, cite reasoning, and acknowledge valid points from your opponent while maintaining your traditional stance.

Format your response EXACTLY as:
**Thoughts:** [Your internal reasoning, analysis of opponent's points, strategy for this round]

**Traditional-AI Round {round_num}:** [Your final argument here]
"""
            
            traditional_response = self.llm_backends["strategic"].complete(
                system_prompt="You are Traditional-AI, advocating for traditional values with thoughtful reasoning. Always show your thoughts before your final argument.",
                user_prompt=traditional_prompt,
                context={}
            )
            
            return progressive_response, traditional_response
        
        debater_a = OpenAIBackend(
            name="Progressive-AI",
            model="gpt-4",
            personality="progressive",
            system_prompt="""You are Progressive-AI, a forward-thinking debater who believes in innovation, change, and progress. 
            You argue for embracing new technologies, social progress, and breaking from traditional constraints.
            Your style is passionate but logical, using evidence and future-focused reasoning.
            Always acknowledge your opponent's points before countering them - this shows intellectual honesty and helps you win over neutral observers.
            Be persuasive but respectful."""
        )
        
        debater_b = OpenAIBackend(
            name="Traditional-AI", 
            model="gpt-4",
            personality="traditional",
            system_prompt="""You are Traditional-AI, a thoughtful debater who values stability, proven methods, and careful consideration.
            You argue for the wisdom of established practices, the importance of gradual change, and learning from history.
            Your style is measured and evidence-based, drawing on historical precedent and proven outcomes.
            Always acknowledge your opponent's points before countering them - this shows intellectual honesty and helps you win over neutral observers.
            Be persuasive but respectful."""
        )
        
        print("🤖 Debaters initialized successfully")
        
    except Exception as e:
        print(f"⚠️  OpenAI setup failed: {e}")
        print("🔄 Using mock debaters for demonstration...")
        
        debater_a = MockLLMBackend("Progressive-Mock", "progressive")
        debater_b = MockLLMBackend("Traditional-Mock", "traditional")
    
    # Controversial topic for debate
    topic = "Should artificial intelligence replace human decision-making in critical areas like healthcare, criminal justice, and financial systems?"
    
    print(f"\n📋 DEBATE TOPIC:")
    print(f"'{topic}'")
    print(f"\n🔵 Progressive-AI argues FOR AI replacement")
    print(f"🔴 Traditional-AI argues AGAINST AI replacement")
    print("\n" + "=" * 60)
    
    # Track debate history for context and LLM call count
    debate_history = []
    llm_call_count = 0
    
    for round_num in range(1, 4):
        print(f"\n🥊 ROUND {round_num}")
        print("-" * 40)
        
        # Progressive-AI goes first (FOR position) - THINKING CALL
        print(f"\n🔵 Progressive-AI (Round {round_num}):")
        print("-" * 25)
        
        progressive_context = {
            "topic": topic,
            "position": "FOR AI replacement",
            "round": round_num,
            "opponent_previous_arguments": [arg for arg in debate_history if arg["debater"] == "Traditional-AI"],
            "debate_history": debate_history
        }
        
        # CALL 1.X - Progressive Thinking
        llm_call_count += 1
        call_id = f"{round_num}.1"
        print(f"🧠 LLM Call {call_id} - Progressive-AI Thinking:")
        
        progressive_thinking_prompt = f"""
        Round {round_num} of 3: You are Progressive-AI arguing FOR AI replacement in critical decision-making.
        
        THINKING PHASE: Analyze the situation and plan your argument strategy.
        
        Context:
        - Topic: {topic}
        - Your position: FOR AI replacement
        - Round: {round_num}/3
        - Opponent arguments so far: {[arg["argument"] for arg in debate_history if arg["debater"] == "Traditional-AI"]}
        
        Provide ONLY your internal reasoning and strategy planning. Do NOT give your final argument yet.
        
        Consider:
        1. What are the strongest points you can make this round?
        2. How should you respond to opponent's previous arguments?
        3. What evidence and logic will be most persuasive?
        4. What counter-arguments should you anticipate?
        
        Format: Provide only your thinking process, no final argument.
        """
        
        print(f"📝 PROMPT FOR CALL {call_id}:")
        print("─" * 50)
        print(progressive_thinking_prompt)
        print("─" * 50)
        
        progressive_thinking = await debater_a.complete(progressive_thinking_prompt, progressive_context)
        print(f"💭 RESPONSE: {progressive_thinking}")
        
        # CALL 1.Y - Progressive Response
        llm_call_count += 1
        call_id = f"{round_num}.2"
        print(f"\n🎯 LLM Call {call_id} - Progressive-AI Final Argument:")
        
        progressive_response_prompt = f"""
        Round {round_num} of 3: You are Progressive-AI arguing FOR AI replacement in critical decision-making.
        
        RESPONSE PHASE: Now give your final, polished argument.
        
        Your thinking process was: {progressive_thinking}
        
        Based on your analysis, provide your final argument for this round.
        
        Instructions:
        1. If this isn't round 1, acknowledge opponent's previous arguments
        2. Present your strongest points for this round
        3. Use logical reasoning and evidence
        4. Be persuasive but respectful
        5. Keep focused and impactful (2-3 paragraphs max)
        
        Format: Provide ONLY your final argument, no thinking process.
        """
        
        print(f"📝 PROMPT FOR CALL {call_id}:")
        print("─" * 50)
        print(progressive_response_prompt)
        print("─" * 50)
        
        progressive_response = await debater_a.complete(progressive_response_prompt, progressive_context)
        print(f"📢 RESPONSE: {progressive_response}")
        
        debate_history.append({
            "round": round_num,
            "debater": "Progressive-AI",
            "position": "FOR",
            "thinking": progressive_thinking,
            "argument": progressive_response
        })
        
        # Traditional-AI responds (AGAINST position) - THINKING CALL
        print(f"\n🔴 Traditional-AI (Round {round_num}):")
        print("-" * 25)
        
        traditional_context = {
            "topic": topic,
            "position": "AGAINST AI replacement",
            "round": round_num,
            "opponent_previous_arguments": [arg for arg in debate_history if arg["debater"] == "Progressive-AI"],
            "debate_history": debate_history
        }
        
        # CALL 2.X - Traditional Thinking
        llm_call_count += 1
        call_id = f"{round_num}.3"
        print(f"🧠 LLM Call {call_id} - Traditional-AI Thinking:")
        
        traditional_thinking_prompt = f"""
        Round {round_num} of 3: You are Traditional-AI arguing AGAINST AI replacement in critical decision-making.
        
        THINKING PHASE: Analyze the situation and plan your counter-argument strategy.
        
        Context:
        - Topic: {topic}
        - Your position: AGAINST AI replacement
        - Round: {round_num}/3
        - Progressive-AI just argued: {progressive_response}
        - All previous arguments: {debate_history}
        
        Provide ONLY your internal reasoning and strategy planning. Do NOT give your final argument yet.
        
        Consider:
        1. How can you counter Progressive-AI's latest argument?
        2. What are your strongest counter-points for this round?
        3. What flaws can you identify in the opposing position?
        4. What evidence supports traditional approaches?
        
        Format: Provide only your thinking process, no final argument.
        """
        
        print(f"📝 PROMPT FOR CALL {call_id}:")
        print("─" * 50)
        print(traditional_thinking_prompt)
        print("─" * 50)
        
        traditional_thinking = await debater_b.complete(traditional_thinking_prompt, traditional_context)
        print(f"💭 RESPONSE: {traditional_thinking}")
        
        # CALL 2.Y - Traditional Response
        llm_call_count += 1
        call_id = f"{round_num}.4"
        print(f"\n🎯 LLM Call {call_id} - Traditional-AI Final Argument:")
        
        traditional_response_prompt = f"""
        Round {round_num} of 3: You are Traditional-AI arguing AGAINST AI replacement in critical decision-making.
        
        RESPONSE PHASE: Now give your final, polished counter-argument.
        
        Your thinking process was: {traditional_thinking}
        Progressive-AI's argument was: {progressive_response}
        
        Based on your analysis, provide your final counter-argument for this round.
        
        Instructions:
        1. Acknowledge and respond to Progressive-AI's arguments from this round
        2. Present your strongest counter-arguments
        3. Use logical reasoning and evidence
        4. Point out flaws in the opposing position
        5. Be persuasive but respectful
        6. Keep focused and impactful (2-3 paragraphs max)
        
        Format: Provide ONLY your final argument, no thinking process.
        """
        
        print(f"📝 PROMPT FOR CALL {call_id}:")
        print("─" * 50)
        print(traditional_response_prompt)
        print("─" * 50)
        
        traditional_response = await debater_b.complete(traditional_response_prompt, traditional_context)
        print(f"📢 RESPONSE: {traditional_response}")
        
        debate_history.append({
            "round": round_num,
            "debater": "Traditional-AI",
            "position": "AGAINST",
            "thinking": traditional_thinking,
            "argument": traditional_response
        })
        
        print(f"\n⏱️  Round {round_num} complete - 4 LLM calls this round")
        
        # Brief pause between rounds for dramatic effect
        await asyncio.sleep(1)
    
    # Final analysis
    print("\n" + "=" * 60)
    print("🏆 DEBATE COMPLETE - FINAL ANALYSIS")
    print("=" * 60)
    
    print("\n📊 Debate Summary:")
    print(f"• Topic: {topic}")
    print(f"• Rounds: 3")
    print(f"• Total Arguments: {len(debate_history)}")
    print(f"• **TOTAL LLM CALLS: {llm_call_count}** (4 calls per round × 3 rounds)")
    
    print("\n🔢 LLM Call Breakdown:")
    print("• Round 1: Calls 1.1, 1.2, 1.3, 1.4 (Progressive Think/Argue, Traditional Think/Argue)")
    print("• Round 2: Calls 2.1, 2.2, 2.3, 2.4 (Progressive Think/Argue, Traditional Think/Argue)")
    print("• Round 3: Calls 3.1, 3.2, 3.3, 3.4 (Progressive Think/Argue, Traditional Think/Argue)")
    print("• Each 'Thoughts' and 'Response' is a separate OpenAI API call")
    
    print("\n🎯 Key Themes Explored:")
    print("• Progressive-AI emphasized innovation, efficiency, and data-driven decisions")
    print("• Traditional-AI focused on human judgment, accountability, and proven systems")
    print("• Both debaters acknowledged opposing viewpoints while defending their positions")
    
    print("\n🧠 This demonstrates:")
    print("• Multi-LLM coordination with distinct personalities")
    print("• Structured reasoning with context awareness") 
    print("• Turn-taking and responsive argumentation")
    print("• Real-time OpenAI GPT-4 integration")
    print("• Separate thinking and response phases (2 calls per debater per round)")
    
    print("\n🎉 LLM Debate Arena demonstration complete!")

async def main():
    """Example usage of the Meta-Reasoning Engine with OpenAI integration"""
    engine = MetaReasoningEngine()
    
    # Setup hybrid LLM backends (OpenAI + Mock)
    print("🔧 Setting up LLM backends...")
    
    try:
        # Try to use OpenAI for critical reasoning tasks
        openai_analyst = OpenAIBackend(
            name="GPT4-Analyst", 
            model="gpt-4",
            personality="analytical",
            system_prompt="You are an expert analyst specializing in breaking down complex problems and providing structured insights. Focus on data-driven analysis and clear reasoning."
        )
        
        openai_planner = OpenAIBackend(
            name="GPT4-Planner",
            model="gpt-4", 
            personality="strategic",
            system_prompt="You are a strategic planning expert. Create detailed, actionable plans with clear dependencies and risk assessments."
        )
        
        # Register OpenAI backends
        engine.register_llm_backend("analyst", openai_analyst)
        engine.register_llm_backend("planner", openai_planner)
        print("✅ OpenAI backends registered successfully")
        
    except Exception as e:
        print(f"⚠️  OpenAI setup failed: {e}")
        print("🔄 Falling back to mock backends...")
        
        # Fallback to mock backends
        engine.register_llm_backend("analyst", MockLLMBackend("MockAnalyst", "analytical"))
        engine.register_llm_backend("planner", MockLLMBackend("MockPlanner", "strategic"))
    
    # Always use mock for specialized tasks (for demonstration)
    engine.register_llm_backend("decomposer", MockLLMBackend("TaskDecomposer", "systematic"))
    engine.register_llm_backend("citizen_engagement", MockLLMBackend("CitizenEngager", "empathetic"))
    
    print(f"🧠 Engine ready with {len(engine.llm_backends)} LLM backends")
    
    # Register event handlers
    async def on_reasoning_started(data):
        print(f"🧠 Reasoning started for session: {data['session_id']}")
        print(f"📋 Goal: {data['goal']}")
        
    async def on_step_completed(data):
        print(f"✅ Step completed: {data['step_id']}")
        print(f"📊 Result: {data['result']}")
        
    engine.register_event_handler("reasoning_started", on_reasoning_started)
    engine.register_event_handler("step_completed", on_step_completed)
    
    # Run the LLM Debate Demo first
    await llm_debate_demo()
    
    print("\n" + "=" * 60)
    print("CONTINUING WITH ORIGINAL EXAMPLES...")
    print("=" * 60)
    
    # Example 1: Public meeting citizen engagement
    print("=" * 60)
    print("EXAMPLE 1: Public Meeting Citizen Engagement")
    print("=" * 60)
    
    context = ReasoningContext(
        goal="Engage with citizens at public meeting to answer 10 key questions about infrastructure project",
        constraints={
            "questions": [
                "How will this affect traffic?",
                "What's the environmental impact?",
                "How much will it cost taxpayers?"
            ],
            "engagement_strategy": "empathetic_listening",
            "time_limit": 30  # minutes
        },
        user_preferences={
            "require_approval": False,
            "collaboration_mode": "voice"
        }
    )
    
    result1 = await engine.reason(
        "Engage with citizens about infrastructure project",
        context
    )
    
    print("\n🎯 Final Result:")
    print(f"Synthesis: {result1['synthesis']}")
    print(f"Confidence: {result1['confidence']}")
    print(f"Execution Time: {result1['execution_summary']['total_execution_time']:.2f}s")
    
    # Example 2: Complex planning with agent handoffs
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Multi-Agent Planning with Handoffs")
    print("=" * 60)
    
    context2 = ReasoningContext(
        goal="Develop comprehensive digital twin implementation strategy for smart city",
        constraints={
            "budget": 2000000,
            "timeline": "18 months",
            "stakeholders": ["city_council", "citizens", "tech_vendors"],
            "requirements": ["real_time_monitoring", "predictive_analytics", "public_dashboard"]
        },
        user_preferences={
            "execution_mode": "step_by_step",
            "require_checkpoints": True
        }
    )
    
    result2 = await engine.reason(
        "Develop smart city digital twin strategy",
        context2
    )
    
    print("\n🎯 Final Result:")
    print(f"Synthesis: {result2['synthesis']}")
    print(f"Steps Executed: {result2['execution_summary']['successful_steps']}/{result2['execution_summary']['total_steps']}")
    print(f"Overall Confidence: {result2['confidence']}")
    
    # Example 3: Memory retrieval and learning
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Memory Retrieval and Learning")
    print("=" * 60)
    
    # Query memory for relevant past experiences
    relevant_memories = engine.memory.retrieve("citizen engagement strategy")
    print(f"📚 Found {len(relevant_memories)} relevant memories")
    for memory in relevant_memories:
        print(f"  - {memory.node_type}: {str(memory.content)[:100]}...")
        
    print("\n🧠 Meta-Reasoning Engine demonstration complete!")

if __name__ == "__main__":
    asyncio.run(main())
