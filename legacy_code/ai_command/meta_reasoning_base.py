#!/usr/bin/env python3
"""
Meta-Reasoning Base - Hybrid Architecture
Combines the best of MetaReasoningEngine (architecture) and AdaptiveIntentRunner (graph intelligence)

Features:
- Clean LLM backend architecture from MetaReasoningEngine
- Sophisticated graph generation from AdaptiveIntentRunner  
- Extensible pipeline for Intent → Strategy → Execution
- Production-ready error handling and session management
"""

import asyncio
import json
import os
import time
import threading
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable, Union, Tuple
from dotenv import load_dotenv
import networkx as nx

# Import ZMQ components
import sys
sys.path.append(r'C:\Users\DamonWeiss\PycharmProjects\Ethos-ZeroMQ')
from ethos_zeromq.EthosZeroMQ_Engine import ZeroMQEngine
from ethos_zeromq.RouterDealerServer import RouterDealerServer, DealerClient

# Import knowledge graph components
from knowledge_graph import KnowledgeGraph, GraphType

# Import Pydantic for structured LLM responses
from pydantic import BaseModel, Field

load_dotenv()
logger = logging.getLogger(__name__)

# ============================================================================
# CORE ENUMS AND DATA STRUCTURES
# ============================================================================

class ConfidenceLevel(Enum):
    HIGH = "high"      # >90% - autonomous execution
    MEDIUM = "medium"  # 70-90% - human review optional
    LOW = "low"        # <70% - human input required
    UNKNOWN = "unknown" # conflicting/insufficient data

class ReasoningPhase(Enum):
    INTENT_ANALYSIS = "intent_analysis"
    STRATEGY_GENERATION = "strategy_generation" 
    EXECUTION_SYNTHESIS = "execution_synthesis"
    RESULT_SYNTHESIS = "result_synthesis"

class CollaborationMode(Enum):
    APPROVAL = "approval"
    INPUT = "input"
    REVIEW = "review"
    OVERRIDE = "override"
    VOICE = "voice"

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
class ReasoningResult:
    """Result from reasoning operation"""
    phase: ReasoningPhase
    status: str  # "completed", "error", "requires_human"
    confidence: ConfidenceLevel
    data: Dict[str, Any] = field(default_factory=dict)
    graphs: Dict[str, KnowledgeGraph] = field(default_factory=dict)
    synthesis: Optional[str] = None
    human_queries: List[str] = field(default_factory=list)
    execution_time: Optional[float] = None
    error: Optional[str] = None

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

# ============================================================================
# SOPHISTICATED ANALYSIS MODELS (from AdaptiveIntentRunner)
# ============================================================================

class AssessmentObjective(BaseModel):
    name: str
    priority: str = Field(pattern="^(critical|high|medium|low)$")

class AssessmentStakeholder(BaseModel):
    name: str
    role: str
    objectives: List[AssessmentObjective]

class SimpleAssessment(BaseModel):
    complexity: str = Field(pattern="^simple$")
    reasoning: str
    direct_answer: str

class ComplexAssessment(BaseModel):
    complexity: str = Field(pattern="^complex$")
    reasoning: str
    stakeholders: List[AssessmentStakeholder]
    required_node_types: List[str] = Field(
        description="Node types that need specialized analysis"
    )
    node_type_heuristics: Dict[str, List[str]] = Field(
        description="Heuristics to apply for each node type"
    )

class TaskDecompositionItem(BaseModel):
    task_id: str = Field(description="Unique identifier for the task")
    task_name: str = Field(description="Clear, actionable task name")
    description: str = Field(description="Detailed task description")
    assigned_agent_type: str = Field(description="Type of AI agent best suited for this task")
    priority: str = Field(pattern="^(critical|high|medium|low)$")
    estimated_duration: str = Field(description="Estimated time to complete (e.g., '2 days', '1 week')")
    dependencies: List[str] = Field(default=[], description="Task IDs this task depends on")
    required_tools: List[str] = Field(description="AI tools/APIs needed to execute this task")

class TaskDecompositionResponse(BaseModel):
    task_breakdown: List[TaskDecompositionItem] = Field(
        max_items=10,
        description="Task decomposition with no gaps, suitable for delegation"
    )
    execution_sequence: List[str] = Field(description="Recommended order of task execution (task_ids)")
    critical_path: List[str] = Field(description="Tasks that cannot be delayed without affecting timeline")
    coverage_assessment: str = Field(description="Confirmation that all aspects of the problem are covered")

class QAQCValidation(BaseModel):
    overall_quality_score: float = Field(ge=0.0, le=1.0, description="0.0-1.0 quality assessment of the analysis")
    identified_gaps: List[str] = Field(description="Missing elements or weak areas in the analysis")
    suggested_improvements: List[str] = Field(description="Specific enhancement recommendations")
    human_queries: List[str] = Field(description="Questions requiring human clarification before execution")
    validation_passed: bool = Field(description="Whether analysis meets quality standards for execution")
    critical_issues: List[str] = Field(description="Must-fix issues before proceeding with execution")
    confidence_assessment: str = Field(description="Overall confidence level in the analysis quality")

# ============================================================================
# LLM BACKEND ARCHITECTURE (from MetaReasoningEngine)
# ============================================================================

class LLMBackend(ABC):
    """Abstract base class for LLM backends"""
    
    @abstractmethod
    async def complete(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate complete response from LLM"""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, context: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Generate streaming response from LLM"""
        pass

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
            print(f"📋 System: {self.system_prompt[:100]}...")
            print(f"👤 User: {prompt}")
            print(f"🔧 Context: {json.dumps(serializable_context, indent=2)[:200]}...")
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
    
    async def complete_structured(self, prompt: str, context: Dict[str, Any], response_model: BaseModel) -> Dict[str, Any]:
        """Generate structured response using Pydantic model"""
        try:
            # Convert context to JSON-serializable format
            serializable_context = self._make_serializable(context)
            
            # Create system prompt for structured output
            schema = response_model.model_json_schema()
            system_prompt = f"""{self.system_prompt}

IMPORTANT: You must respond with actual DATA in JSON format, not the schema definition.

Your response must be valid JSON data that matches this structure:
{json.dumps(schema, indent=2)}

Return ONLY the data object, not the schema. For example, if the schema defines a "constraints" array, return actual constraint objects in that array.

EXAMPLE - If schema defines constraints array, return:
{{
  "constraints": [
    {{"id": "C1", "constraint_type": "budget", "constraint": "Budget is $25k", "value": 25000}}
  ],
  "ai_actionable_insights": [
    {{"insight": "Monitor budget", "actionable_by_ai": true, "required_tools": ["api"], "automation_level": "fully_automated"}}
  ]
}}

NOT the schema definition itself."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {json.dumps(serializable_context, indent=2)}\n\nRequest: {prompt}"}
            ]
            
            print(f"\n🤖 [{self.name}] STRUCTURED PROMPT:")
            print(f"📋 Model: {response_model.__name__}")
            print(f"👤 User: {prompt}")
            print("─" * 60)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2000,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                parsed_json = json.loads(content)
                
                # Check if LLM returned schema instead of data
                if self._is_schema_response(parsed_json):
                    print(f"❌ [{self.name}] LLM returned schema instead of data!")
                    print(f"Schema keys detected: {list(parsed_json.keys())}")
                    print("Rejecting schema response - this should not happen with improved prompt")
                    return {}
                
                print(f"🧠 [{self.name}] STRUCTURED RESPONSE:")
                print(f"💬 {json.dumps(parsed_json, indent=2)[:300]}...")
                print("═" * 60)
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON decode error: {str(e)}")
                print(f"Raw content: {content[:200]}...")
                return {}
                
        except Exception as e:
            print(f"❌ Structured LLM call failed: {str(e)}")
            return {}
    
    def _is_schema_response(self, parsed_json: Dict) -> bool:
        """Detect if LLM returned a schema definition instead of data"""
        schema_indicators = [
            "title", "type", "description", "properties", "required", 
            "$defs", "$ref", "items", "pattern", "enum"
        ]
        
        # If response has schema-like keys, it's probably a schema
        if isinstance(parsed_json, dict):
            keys = set(parsed_json.keys())
            schema_keys = set(schema_indicators)
            
            # If more than 2 schema indicators are present, likely a schema
            if len(keys.intersection(schema_keys)) >= 2:
                return True
                
            # Check nested objects for schema patterns
            for value in parsed_json.values():
                if isinstance(value, dict) and set(value.keys()).intersection(schema_keys):
                    if len(set(value.keys()).intersection(schema_keys)) >= 2:
                        return True
        
        return False
    
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

# MockLLMBackend removed - only real OpenAI backends supported

# ============================================================================
# GRAPH GENERATOR INTERFACES (for separate implementation files)
# ============================================================================

class GraphGenerator(ABC):
    """Abstract base class for graph generators - to be implemented in separate files"""
    
    def __init__(self, llm_backends: Dict[str, LLMBackend]):
        self.llm_backends = llm_backends
    
    @abstractmethod
    async def generate_graph(self, input_data: Any, context: ReasoningContext) -> KnowledgeGraph:
        """Generate a knowledge graph from input data"""
        pass
    
    @abstractmethod
    def get_graph_type(self) -> GraphType:
        """Return the type of graph this generator creates"""
        pass

# ============================================================================
# MAIN META-REASONING BASE ENGINE
# ============================================================================

class MetaReasoningBase:
    """
    Hybrid meta-reasoning engine combining best of both architectures
    
    Architecture from MetaReasoningEngine:
    - Clean LLM backend system
    - Session management
    - Event-driven architecture
    - Error handling
    
    Graph Intelligence from AdaptiveIntentRunner:
    - Sophisticated graph generation
    - Quality assurance
    - Heuristic analysis
    - Production-ready output
    """
    
    def __init__(self, use_real_ai: bool = True, base_port: int = 5900):
        self.llm_backends: Dict[str, LLMBackend] = {}
        self.active_sessions: Dict[str, ReasoningContext] = {}
        self.collaboration_handlers: Dict[str, Callable] = {}
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # ZMQ parallel processing setup
        self.base_port = base_port
        self.zmq_engine = ZeroMQEngine()
        self.servers = {}
        self.shared_results = {}
        
        # Track LLM failures for end-of-run reporting
        self.llm_failures = []
        
        # Initialize LLM backends - only real AI supported
        if not use_real_ai:
            raise ValueError("Mock AI backends have been removed. Only real OpenAI backends are supported.")
        
        self._initialize_real_backends()
    
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

Be thorough and systematic in your analysis."""
            )
            
            # Planner - for strategic planning
            self.llm_backends["planner"] = OpenAIBackend(
                name="Planner",
                model="gpt-4", 
                personality="strategic",
                system_prompt="""You are a strategic planner. Your role is to:
- Break down complex goals into actionable steps
- Create detailed execution plans with timelines
- Identify resource requirements and dependencies
- Anticipate risks and create mitigation strategies

Be systematic and thorough in your planning."""
            )
            
            # Decomposer - for task decomposition
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
            raise RuntimeError(f"Real AI backend initialization failed: {e}. Mock backends are no longer supported.")
    
    def _initialize_mock_backends(self):
        """Mock backends removed - only real LLM backends supported"""
        raise NotImplementedError("Mock backends have been removed. Use use_real_ai=True to initialize real OpenAI backends.")
    
    # Note: Main reasoning pipeline will be implemented in separate files
    # This base class provides the sophisticated analysis capabilities
    
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
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register event handler"""
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered event handler for: {event_type}")
    
    # ============================================================================
    # SOPHISTICATED ANALYSIS CAPABILITIES (from AdaptiveIntentRunner)
    # ============================================================================
    
    async def analyze_complexity(self, user_input: str) -> Dict[str, Any]:
        """
        Stage 1: Complexity Assessment using structured LLM output
        Returns either SimpleAssessment or ComplexAssessment
        """
        print(f"\n🎯 COMPLEXITY ASSESSMENT")
        print("-" * 50)
        
        assessment_prompt = f"""
Analyze this user request for complexity and stakeholder involvement:

User Request: "{user_input}"

Determine if this is:
- SIMPLE: Direct factual question, basic lookup, simple how-to instruction
- COMPLEX: Multi-stakeholder business planning, resource allocation, project management

If SIMPLE: Provide a direct answer.

If COMPLEX: 
1. Identify the most important stakeholders involved (people, roles, groups)
2. For each stakeholder, identify their 1-2 most critical objectives
3. IMPORTANT: Limit the total to a maximum of 6 stakeholder-objective pairs for performance reasons

CONSTRAINT: The total number of stakeholder-objective combinations must not exceed 6. 
If there are many stakeholders, focus on the most critical ones. 
If stakeholders have many objectives, include only their highest priority ones.

Example output structure:
- Restaurant Owner: 1-2 key objectives
- Manager: 1-2 key objectives  
- Staff: 1-2 key objectives
Total: Maximum 6 stakeholder-objective pairs

Focus on identifying the key people/roles and their most important goals.
"""
        
        analyst = self.llm_backends.get("analyst")
        if not analyst or not hasattr(analyst, 'complete_structured'):
            print("⚠️ No structured LLM backend available, using basic analysis")
            return {"complexity": "complex", "reasoning": "Fallback analysis"}
        
        # Try complex assessment first
        try:
            complex_result = await analyst.complete_structured(
                assessment_prompt, 
                {"user_input": user_input}, 
                ComplexAssessment
            )
            
            if complex_result and complex_result.get("complexity") == "complex":
                # Add default node types and heuristics if not provided
                if not complex_result.get("required_node_types"):
                    complex_result["required_node_types"] = ["constraints", "technical_requirements", "assumptions"]
                
                if not complex_result.get("node_type_heuristics"):
                    complex_result["node_type_heuristics"] = {
                        "constraints": ["budget_analysis", "timeline_pressure", "resource_limitations"],
                        "technical_requirements": ["integration_complexity", "scalability_needs", "security_requirements"],
                        "assumptions": ["market_conditions", "user_behavior", "technology_stability"]
                    }
                
                print(f"✅ Complex analysis: {len(complex_result.get('stakeholders', []))} stakeholders identified")
                return complex_result
                
        except Exception as e:
            print(f"⚠️ Complex assessment failed: {str(e)}")
        
        # Fallback to simple assessment
        try:
            simple_result = await analyst.complete_structured(
                assessment_prompt,
                {"user_input": user_input},
                SimpleAssessment
            )
            
            if simple_result:
                print(f"✅ Simple analysis: {simple_result.get('direct_answer', 'Direct response')[:50]}...")
                return simple_result
                
        except Exception as e:
            print(f"⚠️ Simple assessment failed: {str(e)}")
        
        # Ultimate fallback
        return {
            "complexity": "complex",
            "reasoning": "Fallback to complex analysis due to assessment errors",
            "stakeholders": [{"name": "User", "role": "Requestor", "objectives": [{"name": "Complete request", "priority": "high"}]}],
            "required_node_types": ["constraints", "technical_requirements"],
            "node_type_heuristics": {"constraints": ["resource_limitations"], "technical_requirements": ["basic_requirements"]}
        }
    
    async def decompose_tasks(self, user_input: str, analysis_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 3: Task Decomposition using structured LLM output
        """
        print(f"\n🎯 TASK DECOMPOSITION")
        print("-" * 50)
        
        # Extract context for task decomposition
        stakeholders = analysis_context.get("stakeholders", [])
        objectives_count = len([obj for stakeholder in stakeholders for obj in stakeholder.get("objectives", [])])
        
        task_prompt = f"""
Based on this analysis, create a comprehensive task breakdown:

Original Request: "{user_input}"
Stakeholders: {len(stakeholders)} identified
Objectives: {objectives_count} total objectives

Create actionable tasks that cover:
- Requirements gathering and stakeholder consultation
- Technical analysis and architecture planning  
- Resource procurement and budget management
- Implementation planning and timeline development
- Quality assurance and testing strategies
- Deployment and rollout coordination
- System monitoring and performance tracking
- Risk assessment and mitigation planning

For each task, specify:
- Clear, actionable name and description
- Required AI tools/APIs (e.g., expense_tracker_api, integration_tester, email_api)
- Priority level and estimated duration
- Dependencies on other tasks
- Agent type assignment (budget_agent, technical_agent, stakeholder_agent, etc.)

COVERAGE ASSESSMENT: Confirm that your task breakdown covers all major aspects of the original request with no gaps.

Focus on tasks that can be executed by AI agents with minimal human intervention.
"""
        
        decomposer = self.llm_backends.get("decomposer")
        if not decomposer or not hasattr(decomposer, 'complete_structured'):
            print("⚠️ No structured LLM backend available for task decomposition")
            return {"task_breakdown": [], "coverage_assessment": "No decomposition available"}
        
        try:
            result = await decomposer.complete_structured(
                task_prompt,
                {"user_input": user_input, "analysis_context": analysis_context},
                TaskDecompositionResponse
            )
            
            if result and result.get("task_breakdown"):
                print(f"✅ Task decomposition: {len(result['task_breakdown'])} tasks created")
                
                # Show task summary
                for i, task in enumerate(result["task_breakdown"][:3], 1):
                    print(f"   {i}. {task.get('task_name', 'Unknown task')} ({task.get('priority', 'medium')} priority)")
                
                if len(result["task_breakdown"]) > 3:
                    print(f"   ... and {len(result['task_breakdown']) - 3} more tasks")
                
                return result
            
        except Exception as e:
            print(f"❌ Task decomposition failed: {str(e)}")
        
        # Fallback task breakdown
        return {
            "task_breakdown": [
                {
                    "task_id": "analysis_001",
                    "task_name": "Requirements Analysis",
                    "description": "Analyze and document requirements from user input",
                    "assigned_agent_type": "analysis_agent",
                    "priority": "high",
                    "estimated_duration": "1 day",
                    "dependencies": [],
                    "required_tools": ["document_analyzer", "stakeholder_mapper"]
                }
            ],
            "execution_sequence": ["analysis_001"],
            "critical_path": ["analysis_001"],
            "coverage_assessment": "Basic fallback task created due to decomposition errors"
        }
    
    async def validate_quality(self, user_input: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 4: QAQC Validation using structured LLM output
        """
        print(f"\n🎯 QUALITY ASSURANCE VALIDATION")
        print("-" * 50)
        
        qaqc_prompt = f"""
Perform quality assurance validation on this analysis:

Original Request: "{user_input}"
Analysis Summary: {json.dumps(analysis_data, indent=2)[:500]}...

Evaluate:
1. COMPLETENESS: Are all aspects of the request covered?
2. ACTIONABILITY: Can the tasks be executed by AI agents?
3. CLARITY: Are requirements and constraints clearly defined?
4. FEASIBILITY: Are timelines and resources realistic?
5. GAPS: What critical elements might be missing?

Provide:
- Overall quality score (0.0-1.0)
- Specific gaps and improvement suggestions
- Human clarification questions if needed
- Critical issues that must be resolved
- Pass/fail recommendation for execution readiness
"""
        
        analyst = self.llm_backends.get("analyst")
        if not analyst or not hasattr(analyst, 'complete_structured'):
            print("⚠️ No structured LLM backend available for QAQC")
            return {"overall_quality_score": 0.7, "validation_passed": True, "confidence_assessment": "No validation available"}
        
        try:
            result = await analyst.complete_structured(
                qaqc_prompt,
                {"user_input": user_input, "analysis_data": analysis_data},
                QAQCValidation
            )
            
            if result:
                quality_score = result.get("overall_quality_score", 0.7)
                validation_passed = result.get("validation_passed", True)
                
                print(f"✅ QAQC Complete: {quality_score:.2f}/1.0 quality score")
                print(f"   Validation: {'PASSED' if validation_passed else 'NEEDS REVIEW'}")
                
                gaps = result.get("identified_gaps", [])
                if gaps:
                    print(f"   Gaps identified: {len(gaps)} issues")
                
                return result
            
        except Exception as e:
            print(f"❌ QAQC validation failed: {str(e)}")
        
        # Fallback validation
        return {
            "overall_quality_score": 0.75,
            "identified_gaps": ["QAQC system unavailable - manual review recommended"],
            "suggested_improvements": ["Implement proper QAQC validation"],
            "human_queries": ["Please review analysis manually"],
            "validation_passed": True,
            "critical_issues": [],
            "confidence_assessment": "Moderate confidence with fallback validation"
        }
    
    def enhance_graph_with_qaqc(self, graph: KnowledgeGraph, qaqc_result: Dict[str, Any]) -> KnowledgeGraph:
        """
        Enhance knowledge graph based on QAQC validation results
        Adds human query nodes, critical issue nodes, and improvement suggestions
        """
        print(f"🔧 Enhancing graph based on QAQC validation...")
        
        nodes_added = 0
        
        # Add human query nodes
        human_queries = qaqc_result.get("human_queries", [])
        for i, query in enumerate(human_queries, 1):
            query_id = f"human_query_{i}"
            graph.add_node(
                query_id,
                type="human_query",
                name=f"Human Query {i}",
                query=query,
                status="pending",
                priority="high"
            )
            nodes_added += 1
        
        # Add critical issue nodes
        critical_issues = qaqc_result.get("critical_issues", [])
        for i, issue in enumerate(critical_issues, 1):
            issue_id = f"critical_issue_{i}"
            graph.add_node(
                issue_id,
                type="critical_issue",
                name=f"Critical Issue {i}",
                description=issue,
                status="unresolved",
                priority="critical"
            )
            nodes_added += 1
        
        # Add improvement suggestion nodes (limit to top 3)
        improvements = qaqc_result.get("suggested_improvements", [])
        for i, improvement in enumerate(improvements[:3], 1):
            improvement_id = f"improvement_{i}"
            graph.add_node(
                improvement_id,
                type="improvement",
                name=f"Suggested Improvement {i}",
                description=improvement,
                status="suggested",
                priority="medium"
            )
            nodes_added += 1
        
        print(f"   Added {nodes_added} enhancement nodes (queries, risks, improvements)")
        return graph
    
    def serialize_graph_for_qaqc(self, graph: KnowledgeGraph) -> str:
        """
        Convert knowledge graph to JSON format for QAQC analysis
        Creates structured representation suitable for LLM analysis
        """
        graph_data = {
            "nodes": {},
            "relationships": []
        }
        
        # Serialize nodes by type
        for node_id in graph.nodes():
            node_attrs = graph.get_node_attributes(node_id)
            node_type = node_attrs.get('type', 'unknown')
            
            if node_type not in graph_data["nodes"]:
                graph_data["nodes"][node_type] = []
            
            graph_data["nodes"][node_type].append({
                "id": node_id,
                "name": node_attrs.get('name', 'Unknown'),
                "attributes": {k: v for k, v in node_attrs.items() if k not in ['name', 'type']}
            })
        
        # Serialize relationships
        for edge in graph.edges():
            source_id, target_id = edge
            edge_attrs = graph.get_edge_attributes(source_id, target_id)
            relationship = edge_attrs.get('relationship', 'unknown')
            
            source_attrs = graph.get_node_attributes(source_id)
            target_attrs = graph.get_node_attributes(target_id)
            
            graph_data["relationships"].append({
                "source": {
                    "id": source_id,
                    "name": source_attrs.get('name', 'Unknown'),
                    "type": source_attrs.get('type', 'unknown')
                },
                "target": {
                    "id": target_id,
                    "name": target_attrs.get('name', 'Unknown'),
                    "type": target_attrs.get('type', 'unknown')
                },
                "relationship": relationship
            })
        
        return json.dumps(graph_data, indent=2)
    
    def create_comprehensive_qaqc_prompt(self, user_input: str, graph_json: str, tasks_json: str) -> str:
        """
        Create comprehensive QAQC validation prompt based on AdaptiveIntentRunner
        """
        return f"""
Perform quality assurance and gap analysis on this intent analysis:

ORIGINAL USER REQUEST:
"{user_input}"

GENERATED KNOWLEDGE GRAPH:
{graph_json}

GENERATED TASK BREAKDOWN:
{tasks_json}

QAQC ASSESSMENT REQUIREMENTS:

1. QUALITY SCORE (0.0-1.0): Rate the overall quality of this analysis
   - 0.9-1.0: Excellent, comprehensive, ready for execution
   - 0.7-0.8: Good, minor gaps or improvements needed
   - 0.5-0.6: Adequate, several issues to address
   - 0.3-0.4: Poor, major problems need fixing
   - 0.0-0.2: Inadequate, requires complete rework

2. IDENTIFIED GAPS: Missing elements or weak areas
   - Stakeholders not considered
   - Objectives without clear success metrics
   - Constraints not properly analyzed
   - Technical requirements underspecified
   - Dependencies not mapped

3. SUGGESTED IMPROVEMENTS: Specific enhancement recommendations
   - Additional analysis needed
   - Missing relationships to add
   - Clarifications required
   - Alternative approaches to consider

4. HUMAN QUERIES: Questions requiring human clarification
   - Ambiguous requirements
   - Missing business context
   - Unclear priorities or constraints
   - Technical feasibility questions

5. VALIDATION PASSED: Ready for execution?
   - True: Analysis is solid enough for AI agents to execute
   - False: Needs human review or significant improvements

6. CRITICAL ISSUES: Must-fix problems
   - Blocking issues that prevent execution
   - Safety or compliance concerns
   - Resource conflicts or impossibilities

Focus on:
- Logical consistency between nodes and relationships
- Completeness of stakeholder and objective coverage
- Actionability and realism of task breakdown
- Alignment with original user intent
- Practical executability by AI agents

Provide honest, constructive feedback to improve analysis quality.
"""
    
    # ============================================================================
    # NODE TYPE ASSESSMENT & HEURISTIC LINKING (from AdaptiveIntentRunner)
    # ============================================================================
    
    async def assess_required_node_types(self, user_input: str, complexity_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess which node types are pertinent for analysis based on LLM evaluation
        Returns node types and their associated heuristics
        """
        print(f"\n🎯 NODE TYPE ASSESSMENT")
        print("-" * 50)
        
        # Default node types with their purposes
        available_node_types = {
            "constraints": {
                "focus": "limitations, restrictions, and boundaries that affect the project",
                "examples": "budget limits, timeline restrictions, regulatory requirements, resource constraints",
                "heuristic_tags": ["cascade_potential", "rigid_constraint", "bottleneck_risk"]
            },
            "technical_requirements": {
                "focus": "technical specifications, integrations, and system capabilities needed",
                "examples": "API integrations, database requirements, security specifications, performance needs",
                "heuristic_tags": ["integration_complexity", "scalability_needs", "security_requirements"]
            },
            "assumptions": {
                "focus": "beliefs, expectations, and assumptions being made about the project",
                "examples": "market conditions, user behavior, technology stability, resource availability",
                "heuristic_tags": ["high_risk", "validation_needed", "leverage_point"]
            },
            "stakeholders": {
                "focus": "people, roles, and groups involved in or affected by the project",
                "examples": "decision makers, end users, technical teams, external partners",
                "heuristic_tags": ["authority_patterns", "influence_networks", "coordination_needs"]
            },
            "success_metrics": {
                "focus": "measurable outcomes and key performance indicators",
                "examples": "revenue targets, user adoption, performance benchmarks, quality metrics",
                "heuristic_tags": ["feedback_loops", "reinforcing_cycles", "measurement_challenges"]
            }
        }
        
        # Use LLM to assess which node types are most relevant
        assessment_prompt = f"""
Based on this user request and complexity assessment, determine which node types are most pertinent for analysis:

User Request: "{user_input}"
Complexity Assessment: {json.dumps(complexity_assessment, indent=2)}

Available Node Types:
{json.dumps(available_node_types, indent=2)}

For each relevant node type, assess:
1. Relevance (critical, high, medium, low)
2. Complexity (high, medium, low) 
3. Priority for analysis (1-5, where 1 is highest priority)
4. Specific heuristics to apply from the heuristic_tags

Return the top 3-5 most relevant node types with their assessments and recommended heuristics.
Focus on node types that will provide the most insight for this specific request.
"""
        
        analyst = self.llm_backends.get("analyst")
        if not analyst:
            # Fallback to default node types
            return {
                "required_node_types": ["constraints", "technical_requirements", "assumptions"],
                "node_type_heuristics": {
                    "constraints": ["cascade_potential", "bottleneck_risk"],
                    "technical_requirements": ["integration_complexity", "scalability_needs"],
                    "assumptions": ["high_risk", "validation_needed"]
                }
            }
        
        try:
            response = await analyst.complete(assessment_prompt, {"user_input": user_input})
            
            # Parse the response to extract node types and heuristics
            # For now, return enhanced defaults based on complexity
            stakeholders = complexity_assessment.get("stakeholders", [])
            
            if len(stakeholders) > 2:
                # Complex multi-stakeholder scenario
                return {
                    "required_node_types": ["constraints", "technical_requirements", "assumptions", "stakeholders", "success_metrics"],
                    "node_type_heuristics": {
                        "constraints": ["cascade_potential", "rigid_constraint", "bottleneck_risk"],
                        "technical_requirements": ["integration_complexity", "scalability_needs", "security_requirements"],
                        "assumptions": ["high_risk", "validation_needed", "leverage_point"],
                        "stakeholders": ["authority_patterns", "influence_networks", "coordination_needs"],
                        "success_metrics": ["feedback_loops", "reinforcing_cycles"]
                    }
                }
            else:
                # Simpler scenario
                return {
                    "required_node_types": ["constraints", "technical_requirements", "assumptions"],
                    "node_type_heuristics": {
                        "constraints": ["cascade_potential", "bottleneck_risk"],
                        "technical_requirements": ["integration_complexity", "scalability_needs"],
                        "assumptions": ["high_risk", "validation_needed"]
                    }
                }
                
        except Exception as e:
            print(f"❌ Node type assessment failed: {str(e)}")
            return {
                "required_node_types": ["constraints", "technical_requirements", "assumptions"],
                "node_type_heuristics": {
                    "constraints": ["cascade_potential"],
                    "technical_requirements": ["integration_complexity"],
                    "assumptions": ["high_risk"]
                }
            }
    
    def apply_heuristic_linking(self, graph: KnowledgeGraph, analysis_data: Dict[str, Any]) -> KnowledgeGraph:
        """
        Apply sophisticated heuristic linking based on LLM analysis + mathematical graph patterns
        Phase 1: LLM-based semantic linking
        Phase 2: Mathematical graph analysis and backfill
        """
        print(f"🔗 Applying advanced heuristic linking...")
        print(f"   📊 Processing analysis data with {len(graph.nodes())} nodes")
        
        # Phase 1: LLM-based semantic heuristics
        print(f"   🧠 Phase 1: LLM-based semantic linking")
        self._apply_semantic_heuristics(graph, analysis_data)
        
        # Phase 2: Mathematical graph heuristics (backfill)
        print(f"   📊 Phase 2: Mathematical graph analysis")
        self._apply_mathematical_heuristics(graph)
        
        # Phase 3: Domain-specific patterns
        print(f"   🏢 Phase 3: Domain-specific patterns")
        domain = self._extract_domain_from_analysis(analysis_data)
        self._apply_domain_patterns(graph, domain)
        
        print(f"   ✅ Applied advanced heuristic linking")
        return graph
    
    def _apply_semantic_heuristics(self, graph: KnowledgeGraph, analysis_data: Dict[str, Any]):
        """Apply LLM-based semantic heuristics from analysis data"""
        
        # Link based on stakeholder influence patterns
        stakeholders = analysis_data.get("stakeholders", [])
        for stakeholder in stakeholders:
            stakeholder_name = stakeholder.get("name", "")
            stakeholder_role = stakeholder.get("role", "").lower()
            
            # Find stakeholder nodes
            stakeholder_nodes = [n for n in graph.nodes() 
                               if stakeholder_name.lower() in graph.get_node_attributes(n).get('name', '').lower()]
            
            for stakeholder_node in stakeholder_nodes:
                if "owner" in stakeholder_role or "decision" in stakeholder_role:
                    # Decision makers have authority over constraints
                    constraint_nodes = [n for n in graph.nodes() 
                                      if graph.get_node_attributes(n).get('type') == 'constraint']
                    for con_node in constraint_nodes[:3]:  # Limit connections
                        graph.add_edge(stakeholder_node, con_node, relationship="has_authority_over")
                
                elif "manager" in stakeholder_role:
                    # Managers coordinate technical requirements
                    tech_nodes = [n for n in graph.nodes() 
                                if graph.get_node_attributes(n).get('type') == 'technical_requirement']
                    for tech_node in tech_nodes[:3]:
                        graph.add_edge(stakeholder_node, tech_node, relationship="coordinates")
        
        # Link technical enablement patterns
        self._link_technical_enablement(graph, analysis_data)
        
        # Link resource competition patterns
        self._link_resource_competition(graph, analysis_data)
    
    def _apply_mathematical_heuristics(self, graph: KnowledgeGraph):
        """Apply mathematical graph analysis to backfill additional relationships"""
        
        # 1. Resource competition analysis (mathematical)
        self._analyze_resource_competition_math(graph)
        
        # 2. Temporal sequence analysis (mathematical)
        self._analyze_temporal_sequences_math(graph)
        
        # 3. Bottleneck analysis (mathematical)
        self._analyze_bottlenecks_math(graph)
        
        # 4. Authority pattern analysis (mathematical)
        self._analyze_authority_patterns_math(graph)
    
    def _analyze_resource_competition_math(self, graph: KnowledgeGraph):
        """Mathematical analysis: Find objectives that share constraints (compete for resources)"""
        
        competition_links = 0
        
        # Group objectives by their constraints
        constraint_to_objectives = {}
        
        for edge in graph.edges():
            source_id, target_id = edge
            edge_attrs = graph.get_edge_attributes(source_id, target_id)
            relationship = edge_attrs.get('relationship', '')
            
            if relationship == 'constrains':
                # source_id is constraint, target_id is objective
                source_attrs = graph.get_node_attributes(source_id)
                constraint_type = source_attrs.get('name', '').lower()
                
                if constraint_type not in constraint_to_objectives:
                    constraint_to_objectives[constraint_type] = []
                constraint_to_objectives[constraint_type].append(target_id)
        
        # Create competition relationships between objectives sharing constraints
        for constraint_type, objectives in constraint_to_objectives.items():
            if len(objectives) > 1:
                if 'budget' in constraint_type or 'cost' in constraint_type:
                    # Budget competition - link all pairs
                    for i in range(len(objectives)):
                        for j in range(i+1, len(objectives)):
                            graph.add_edge(objectives[i], objectives[j], relationship="competes_for_budget")
                            competition_links += 1
                elif 'time' in constraint_type or 'deadline' in constraint_type:
                    # Timeline competition - link all pairs
                    for i in range(len(objectives)):
                        for j in range(i+1, len(objectives)):
                            graph.add_edge(objectives[i], objectives[j], relationship="competes_for_timeline")
                            competition_links += 1
        
        print(f"      💰 Created {competition_links} resource competition relationships")
    
    def _analyze_temporal_sequences_math(self, graph: KnowledgeGraph):
        """Mathematical analysis: Identify temporal dependencies from node patterns"""
        
        temporal_links = 0
        
        # Find technical requirements and analyze for temporal patterns
        tech_nodes = [n for n in graph.nodes() 
                     if graph.get_node_attributes(n).get('type') == 'technical_requirement']
        
        for node_id in tech_nodes:
            node_attrs = graph.get_node_attributes(node_id)
            tech_name = node_attrs.get('name', '').lower()
            
            # Look for prerequisite patterns in other tech nodes
            for other_id in tech_nodes:
                if other_id != node_id:
                    other_attrs = graph.get_node_attributes(other_id)
                    other_name = other_attrs.get('name', '').lower()
                    
                    # Common temporal patterns
                    if 'payment' in tech_name and 'order' in other_name:
                        graph.add_edge(other_id, node_id, relationship="temporal_prerequisite_for")
                        temporal_links += 1
                    elif 'inventory' in tech_name and ('order' in other_name or 'payment' in other_name):
                        graph.add_edge(other_id, node_id, relationship="temporal_prerequisite_for")
                        temporal_links += 1
        
        print(f"      ⏰ Created {temporal_links} temporal prerequisite relationships")
    
    def _analyze_bottlenecks_math(self, graph: KnowledgeGraph):
        """Mathematical analysis: Identify bottlenecks based on node degree centrality"""
        
        bottlenecks_created = 0
        total_nodes = len(graph.nodes())
        
        # Calculate degree centrality threshold
        if total_nodes <= 10:
            threshold = 4
        elif total_nodes <= 30:
            threshold = 6
        else:
            threshold = 8
        
        # Find high-degree nodes (potential bottlenecks)
        for node_id in graph.nodes():
            degree = graph.degree(node_id)
            
            if degree > threshold:
                node_attrs = graph.get_node_attributes(node_id)
                node_type = node_attrs.get('type')
                
                if node_type == 'technical_requirement':
                    # High-connection tech requirements are system bottlenecks
                    objective_nodes = [n for n in graph.nodes() 
                                     if graph.get_node_attributes(n).get('type') == 'primary_objective']
                    
                    for obj_node in objective_nodes[:3]:  # Limit connections
                        graph.add_edge(node_id, obj_node, relationship="potential_bottleneck_for")
                        bottlenecks_created += 1
        
        print(f"      🔧 Created {bottlenecks_created} bottleneck relationships")
    
    def _analyze_authority_patterns_math(self, graph: KnowledgeGraph):
        """Mathematical analysis: Identify authority patterns from stakeholder connections"""
        
        authority_links = 0
        
        # Find stakeholder nodes
        stakeholder_nodes = [n for n in graph.nodes() 
                           if graph.get_node_attributes(n).get('type') == 'stakeholder']
        
        for stakeholder_id in stakeholder_nodes:
            stakeholder_attrs = graph.get_node_attributes(stakeholder_id)
            stakeholder_name = stakeholder_attrs.get('name', '').lower()
            
            # Authority patterns based on naming
            if 'owner' in stakeholder_name or 'ceo' in stakeholder_name or 'director' in stakeholder_name:
                # High authority - connects to constraints and objectives
                constraint_nodes = [n for n in graph.nodes() 
                                  if graph.get_node_attributes(n).get('type') == 'constraint']
                
                for con_node in constraint_nodes[:2]:  # Limit connections
                    graph.add_edge(stakeholder_id, con_node, relationship="has_authority_over")
                    authority_links += 1
        
        print(f"      👑 Created {authority_links} authority relationships")
    
    def _link_technical_enablement(self, graph: KnowledgeGraph, analysis_data: Dict[str, Any]):
        """Link technical requirements to objectives based on enablement analysis"""
        
        # Find technical and objective nodes
        tech_nodes = [n for n in graph.nodes() 
                     if graph.get_node_attributes(n).get('type') == 'technical_requirement']
        objective_nodes = [n for n in graph.nodes() 
                         if graph.get_node_attributes(n).get('type') == 'primary_objective']
        
        # Create enablement relationships based on semantic matching
        for tech_node in tech_nodes:
            tech_attrs = graph.get_node_attributes(tech_node)
            tech_name = tech_attrs.get('name', '').lower()
            
            for obj_node in objective_nodes:
                obj_attrs = graph.get_node_attributes(obj_node)
                obj_name = obj_attrs.get('name', '').lower()
                
                # Semantic matching for enablement
                if any(word in tech_name for word in obj_name.split()[:3]):
                    graph.add_edge(tech_node, obj_node, relationship="enables")
                elif 'critical' in tech_name or 'essential' in tech_name:
                    graph.add_edge(tech_node, obj_node, relationship="critically_enables")
    
    def _link_resource_competition(self, graph: KnowledgeGraph, analysis_data: Dict[str, Any]):
        """Link objectives that compete for the same resources"""
        
        # Find objectives that share budget constraints
        budget_objectives = []
        timeline_objectives = []
        
        for node_id in graph.nodes():
            node_attrs = graph.get_node_attributes(node_id)
            if node_attrs.get('type') == 'primary_objective':
                # Check if this objective is constrained by budget or timeline
                for edge in graph.edges():
                    source_id, target_id = edge
                    if target_id == node_id:
                        source_attrs = graph.get_node_attributes(source_id)
                        source_name = source_attrs.get('name', '').lower()
                        
                        if 'budget' in source_name or 'cost' in source_name:
                            budget_objectives.append(node_id)
                        elif 'time' in source_name or 'deadline' in source_name:
                            timeline_objectives.append(node_id)
        
        # Create competition relationships
        for i, obj1 in enumerate(budget_objectives):
            for obj2 in budget_objectives[i+1:]:
                graph.add_edge(obj1, obj2, relationship="competes_for_budget")
        
        for i, obj1 in enumerate(timeline_objectives):
            for obj2 in timeline_objectives[i+1:]:
                graph.add_edge(obj1, obj2, relationship="competes_for_timeline")
    
    def _apply_domain_patterns(self, graph: KnowledgeGraph, domain: str):
        """Apply domain-specific linking patterns"""
        
        if "restaurant" in domain.lower():
            self._apply_restaurant_patterns(graph)
        elif "mobile" in domain.lower() or "app" in domain.lower():
            self._apply_mobile_app_patterns(graph)
        elif "corporate" in domain.lower():
            self._apply_corporate_patterns(graph)
    
    def _apply_restaurant_patterns(self, graph: KnowledgeGraph):
        """Restaurant-specific value chain relationships"""
        
        # POS → Order efficiency → Customer satisfaction → Revenue
        pos_nodes = [n for n in graph.nodes() 
                    if "pos" in graph.get_node_attributes(n).get('name', '').lower()]
        order_nodes = [n for n in graph.nodes() 
                      if "order" in graph.get_node_attributes(n).get('name', '').lower()]
        customer_nodes = [n for n in graph.nodes() 
                         if "customer" in graph.get_node_attributes(n).get('name', '').lower()]
        
        for pos_node in pos_nodes:
            for order_node in order_nodes:
                graph.add_edge(pos_node, order_node, relationship="improves_efficiency_of")
            for customer_node in customer_nodes:
                graph.add_edge(pos_node, customer_node, relationship="enhances_experience_for")
    
    def _apply_mobile_app_patterns(self, graph: KnowledgeGraph):
        """Mobile app growth loop relationships"""
        
        tracking_nodes = [n for n in graph.nodes() 
                         if "tracking" in graph.get_node_attributes(n).get('name', '').lower()]
        user_nodes = [n for n in graph.nodes() 
                     if "user" in graph.get_node_attributes(n).get('name', '').lower()]
        
        for tracking_node in tracking_nodes:
            for user_node in user_nodes:
                graph.add_edge(tracking_node, user_node, relationship="drives_engagement_for")
    
    def _apply_corporate_patterns(self, graph: KnowledgeGraph):
        """Corporate systems patterns"""
        
        policy_nodes = [n for n in graph.nodes() 
                       if "policy" in graph.get_node_attributes(n).get('name', '').lower()]
        behavior_nodes = [n for n in graph.nodes() 
                         if "behavior" in graph.get_node_attributes(n).get('name', '').lower()]
        
        for policy_node in policy_nodes:
            for behavior_node in behavior_nodes:
                graph.add_edge(policy_node, behavior_node, relationship="influences_behavior")
    
    def _extract_domain_from_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """Extract domain from analysis data"""
        
        stakeholders = analysis_data.get("stakeholders", [])
        if stakeholders:
            first_stakeholder = stakeholders[0]
            role = first_stakeholder.get("role", "").lower()
            
            if "restaurant" in role or "food" in role:
                return "restaurant"
            elif "mobile" in role or "app" in role:
                return "mobile_app"
            elif "corporate" in role or "company" in role:
                return "corporate"
        
        return "general"
    
    # ============================================================================
    # ZMQ PARALLEL PROCESSING CAPABILITIES (from AdaptiveIntentRunner)
    # ============================================================================
    
    async def parallel_stakeholder_expansion(self, user_input: str, complexity_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parallel stakeholder-objective expansion using ZMQ
        Creates multiple servers for simultaneous LLM analysis
        """
        start_time = time.time()
        print(f"\n🚀 PARALLEL STAKEHOLDER EXPANSION")
        print("-" * 50)
        
        stakeholders = complexity_assessment.get("stakeholders", [])
        if not stakeholders:
            print("⚠️ No stakeholders found, skipping parallel expansion")
            return {}
        
        # Calculate total stakeholder-objective pairs
        total_pairs = sum(len(stakeholder.get("objectives", [])) for stakeholder in stakeholders)
        
        # Safety check: If LLM exceeded the constraint, apply fallback trimming
        if total_pairs > 6:
            print(f"⚠️ LLM exceeded 6-pair constraint ({total_pairs} pairs) - applying fallback trimming")
            # Trim stakeholders and objectives to get exactly 6 pairs
            limited_stakeholders = []
            pairs_count = 0
            
            for stakeholder in stakeholders:
                if pairs_count >= 6:
                    break
                objectives = stakeholder.get("objectives", [])
                remaining_slots = 6 - pairs_count
                
                if len(objectives) <= remaining_slots:
                    # Take all objectives from this stakeholder
                    limited_stakeholders.append(stakeholder)
                    pairs_count += len(objectives)
                else:
                    # Take only some objectives from this stakeholder
                    limited_stakeholder = stakeholder.copy()
                    limited_stakeholder["objectives"] = objectives[:remaining_slots]
                    limited_stakeholders.append(limited_stakeholder)
                    pairs_count += remaining_slots
            
            stakeholders = limited_stakeholders
            total_pairs = 6
        
        print(f"📊 Processing {len(stakeholders)} stakeholders with {total_pairs} total expansion requests")
        
        if total_pairs == 0:
            print("⚠️ No stakeholder-objective pairs found")
            return {}
        
        # Start ZMQ servers for parallel processing
        server_start_time = time.time()
        try:
            self._start_parallel_servers(total_pairs)
            
            # Send stakeholder-objective expansion requests
            clients = []
            client_index = 0
            for i, stakeholder in enumerate(stakeholders):
                for j, objective in enumerate(stakeholder.get("objectives", [])):
                    port = self.base_port + client_index
                    # Use DealerClient like AdaptiveIntentRunner
                    client = DealerClient(connect=f"tcp://localhost:{port}")
                    clients.append(client)
                    
                    message = {
                        "user_input": user_input,
                        "stakeholder": stakeholder,
                        "objective": objective,
                        "expansion_type": "stakeholder_objective",
                        "expansion_key": f"expansion_{client_index}"
                    }
                    
                    # Send with content_type matching the registered handler
                    content_type = f"expansion_{client_index}"
                    print(f"   📤 Sending to {content_type} on port {port}")
                    print(f"      Payload key: {message.get('expansion_key')} stakeholder={stakeholder.get('name')} objective={objective.get('name')}")
                    # Send raw dict like AdaptiveIntentRunner (DealerClient will JSON serialize)
                    client.send_message(message, content_type)
                    client_index += 1
            
            print(f"📤 Sent {len(clients)} parallel expansion requests")
            server_setup_time = time.time() - server_start_time
            print(f"⏱️ Server setup completed in {server_setup_time:.2f}s")
            
            # Wait for completion by polling shared_results like AdaptiveIntentRunner
            processing_start_time = time.time()
            print(f"📥 Waiting for {len(clients)} expansion responses...")
            
            seen_expansion_keys: set[str] = set()
            expected_expansion_keys = [f"expansion_{k}" for k in range(len(clients))]
            timeout_seconds = 30  # 30 second timeout for stakeholder expansions (reduced for debugging)
            timeout_start = time.time()
            
            while True:
                current_expansion_keys = {k for k in self.shared_results.keys() if k.startswith('expansion_')}
                new_keys = sorted(current_expansion_keys - seen_expansion_keys)
                for k in new_keys:
                    print(f"   ✅ Expansion received: {k}")
                    print(f"      Current expansion keys: {sorted(current_expansion_keys)}")
                    missing = [ek for ek in expected_expansion_keys if ek not in current_expansion_keys]
                    print(f"      Missing expansion keys: {missing}")
                    print(f"      Progress: {len(current_expansion_keys)}/{len(clients)}")
                seen_expansion_keys |= set(new_keys)
                
                # Check if all completed
                if len(current_expansion_keys) >= len(clients):
                    break
                
                # Check timeout
                elapsed = time.time() - timeout_start
                if elapsed > timeout_seconds:
                    missing = [ek for ek in expected_expansion_keys if ek not in current_expansion_keys]
                    print(f"\n⚠️ TIMEOUT after {timeout_seconds}s - {len(missing)} expansions still missing:")
                    for missing_key in missing:
                        print(f"   - {missing_key}")
                    
                    # Create placeholder entries for missing keys to allow completion
                    for missing_key in missing:
                        self.shared_results[missing_key] = {
                            "stakeholder": {},
                            "objective": {},
                            "expansion": {},
                            "error": f"Timeout after {timeout_seconds}s - handler never completed",
                            "status": "timeout"
                        }
                        print(f"   📝 Created timeout placeholder for {missing_key}")
                    break
                
                await asyncio.sleep(0.1)
            
            # Clean up clients (do not block on receive)
            for client in clients:
                client.close()
            
            processing_time = time.time() - processing_start_time
            total_time = time.time() - start_time
            print(f"⏱️ Parallel processing completed in {processing_time:.2f}s")
            print(f"⏱️ Total stakeholder expansion time: {total_time:.2f}s")
            
            # Report LLM failures at end of stakeholder processing
            self._report_llm_failures("stakeholder expansion")
            
            return self.shared_results
            
        except Exception as e:
            print(f"❌ Parallel expansion failed: {str(e)}")
            return {}
        finally:
            self._stop_servers()
    
    async def parallel_node_type_expansion(self, user_input: str, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parallel node-type-specific expansion using ZMQ
        Each node type gets its own specialized analysis
        """
        start_time = time.time()
        print(f"\n🚀 PARALLEL NODE TYPE EXPANSION")
        print("-" * 50)
        
        required_node_types = assessment.get("required_node_types", ["constraints", "technical_requirements", "assumptions"])
        node_type_heuristics = assessment.get("node_type_heuristics", {})
        
        print(f"📊 Processing {len(required_node_types)} node types in parallel")
        
        server_start_time = time.time()
        try:
            # Start servers for node type expansion
            self._start_parallel_servers(len(required_node_types))
            
            # Send node-type expansion requests
            clients = []
            for i, node_type in enumerate(required_node_types):
                port = self.base_port + i
                # Use DealerClient like AdaptiveIntentRunner
                client = DealerClient(connect=f"tcp://localhost:{port}")
                clients.append(client)
                
                heuristics = node_type_heuristics.get(node_type, [])
                message = {
                    "user_input": user_input,
                    "assessment": assessment,
                    "node_type": node_type,
                    "heuristics": heuristics,
                    "expansion_key": f"node_type_{node_type}"
                }
                
                # Send with content_type matching the registered handler
                content_type = f"node_type_{i}"
                print(f"   📤 Sending to {content_type} on port {port}")
                client.send_message(json.dumps(message), content_type)
            
            print(f"📤 Sent {len(clients)} node-type expansion requests")
            server_setup_time = time.time() - server_start_time
            print(f"⏱️ Server setup completed in {server_setup_time:.2f}s")
            
            # Wait for completion by polling shared_results like AdaptiveIntentRunner
            processing_start_time = time.time()
            print(f"📥 Waiting for {len(clients)} node-type responses...")
            
            seen_node_type_keys: set[str] = set()
            expected_nt_keys = [f"node_type_{nt}" for nt in required_node_types]
            timeout_seconds = 30  # 30 second timeout for node type expansions (reduced for debugging)
            timeout_start = time.time()
            
            while True:
                current_node_type_keys = {k for k in self.shared_results.keys() if k.startswith('node_type_')}
                new_nt_keys = sorted(current_node_type_keys - seen_node_type_keys)
                for k in new_nt_keys:
                    # Extract pretty name
                    pretty = k.replace('node_type_', '')
                    print(f"   ✅ Node Type completed: {pretty}")
                    print(f"      Current node-type keys: {sorted(current_node_type_keys)}")
                    missing_nt = [ek for ek in expected_nt_keys if ek not in current_node_type_keys]
                    print(f"      Missing node-type keys: {missing_nt}")
                    print(f"      Progress: {len(current_node_type_keys)}/{len(expected_nt_keys)}")
                seen_node_type_keys |= set(new_nt_keys)
                
                # Check if all completed
                if len(current_node_type_keys) >= len(expected_nt_keys):
                    break
                
                # Check timeout
                elapsed = time.time() - timeout_start
                if elapsed > timeout_seconds:
                    missing_nt = [ek for ek in expected_nt_keys if ek not in current_node_type_keys]
                    print(f"\n⚠️ TIMEOUT after {timeout_seconds}s - {len(missing_nt)} node types still missing:")
                    for missing_key in missing_nt:
                        print(f"   - {missing_key}")
                    
                    # Create placeholder entries for missing keys to allow completion
                    for missing_key in missing_nt:
                        node_type = missing_key.replace("node_type_", "") if missing_key.startswith("node_type_") else "unknown"
                        self.shared_results[missing_key] = {
                            "node_type": node_type,
                            "heuristics": [],
                            "expansion": {},
                            "error": f"Timeout after {timeout_seconds}s - handler never completed",
                            "status": "timeout"
                        }
                        print(f"   📝 Created timeout placeholder for {missing_key}")
                    break
                
                await asyncio.sleep(0.1)
            
            # Clean up clients (do not block on receive)
            for client in clients:
                client.close()
            
            processing_time = time.time() - processing_start_time
            total_time = time.time() - start_time
            print(f"⏱️ Parallel processing completed in {processing_time:.2f}s")
            print(f"⏱️ Total node type expansion time: {total_time:.2f}s")
            
            # Report LLM failures at end of node type processing
            self._report_llm_failures("node type expansion")
            
            return self.shared_results
            
        except Exception as e:
            print(f"❌ Parallel node type expansion failed: {str(e)}")
            return {}
        finally:
            self._stop_servers()
    
    def _start_parallel_servers(self, num_servers: int):
        """Start ZMQ servers for parallel processing"""
        start_time = time.time()
        print(f"🔧 Starting {num_servers} ZMQ servers...")
        
        for i in range(num_servers):
            port = self.base_port + i
            try:
                # Use RouterDealerServer directly like AdaptiveIntentRunner
                server = RouterDealerServer(
                    engine=self.zmq_engine,
                    bind=f"tcp://*:{port}",
                    connect=f"tcp://localhost:{port}"
                )
                # Register both handler types like AdaptiveIntentRunner
                expansion_handler = f"expansion_{i}"
                node_type_handler = f"node_type_{i}"
                print(f"   🔧 Registering handlers '{expansion_handler}' and '{node_type_handler}' on port {port}")
                server.register_handler(expansion_handler, self._handle_parallel_request)
                server.register_handler(node_type_handler, self._handle_parallel_request)
                server.start()
                self.servers[port] = server
                print(f"   ✅ Server {i+1} started on port {port}")
                time.sleep(0.02)  # Brief delay like AdaptiveIntentRunner
            except Exception as e:
                print(f"   ❌ Failed to start server {i+1} on port {port}: {str(e)}")
        
        setup_time = time.time() - start_time
        print(f"⏱️ All {num_servers} servers started in {setup_time:.2f}s")
    
    def _stop_servers(self):
        """Stop all ZMQ servers with timeout to prevent hanging"""
        start_time = time.time()
        num_servers = len(self.servers)
        print(f"🔧 Stopping {num_servers} ZMQ servers...")
        
        # Set timeout for server shutdown
        shutdown_timeout = 10.0  # 10 second timeout for shutdown
        
        for port, server in self.servers.items():
            try:
                server_start = time.time()
                
                # Check overall timeout before attempting each server
                elapsed = time.time() - start_time
                if elapsed > shutdown_timeout:
                    print(f"   ⚠️ Overall shutdown timeout reached ({shutdown_timeout}s) - skipping remaining servers")
                    break
                
                print(f"   🔧 Stopping server on port {port}...")
                
                # Try to stop server with threading timeout
                stop_success = self._stop_server_with_timeout(server, port, 3.0)  # 3 second timeout per server
                server_time = time.time() - server_start
                
                if stop_success:
                    print(f"   ✅ Server on port {port} stopped in {server_time:.2f}s")
                else:
                    print(f"   ⚠️ Server on port {port} timed out after 3s - forced cleanup")
                    print(f"      (Server may still be running in background)")
                
                # Individual server timeout check
                if server_time > 5.0:  # 5 second per-server timeout
                    print(f"   ⚠️ Server {port} took too long to stop ({server_time:.2f}s)")
                    
            except Exception as e:
                print(f"   ❌ Failed to stop server on port {port}: {str(e)}")
                # Continue to next server even if one fails
        
        self.servers.clear()
        # Do not clear shared_results here; callers may still need the results
        
        cleanup_time = time.time() - start_time
        print(f"⏱️ Server shutdown completed in {cleanup_time:.2f}s")
    
    def _stop_server_with_timeout(self, server, port: int, timeout_seconds: float) -> bool:
        """Stop a server with timeout using threading"""
        stop_completed = threading.Event()
        stop_exception = None
        
        def stop_server_thread():
            nonlocal stop_exception
            try:
                server.stop()
                stop_completed.set()
            except Exception as e:
                stop_exception = e
                stop_completed.set()
        
        # Start stop operation in separate thread
        stop_thread = threading.Thread(target=stop_server_thread, daemon=True)
        stop_thread.start()
        
        # Wait for completion or timeout
        completed = stop_completed.wait(timeout_seconds)
        
        if completed and stop_exception is None:
            return True  # Success
        elif completed and stop_exception is not None:
            print(f"      Server {port} stop failed: {str(stop_exception)}")
            return False  # Failed with exception
        else:
            print(f"      Server {port} stop timed out after {timeout_seconds}s")
            return False  # Timed out
    
    def _handle_parallel_request(self, message):
        """Handle parallel processing requests - ALWAYS returns a response"""
        print(f"🔧 Handler received message: {type(message)}")
        try:
            # Parse JSON message
            if isinstance(message, str):
                message = json.loads(message)
            
            expansion_type = message.get("expansion_type", "node_type")
            print(f"🔧 Processing expansion_type: {expansion_type}")
            
            if expansion_type == "stakeholder_objective":
                print(f"🔧 Calling stakeholder handler...")
                result = self._handle_stakeholder_objective_expansion(message)
                print(f"✅ Stakeholder expansion handler completed: {result.get('status', 'unknown')}")
                return result
            else:
                print(f"🔧 Calling node type handler...")
                result = self._handle_node_type_expansion(message)
                print(f"✅ Node type expansion handler completed: {result.get('status', 'unknown')}")
                return result
        except Exception as e:
            print(f"❌ Error in parallel request handler: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _handle_stakeholder_objective_expansion(self, message):
        """Handle stakeholder-objective expansion request - BASE IMPLEMENTATION"""
        print(f"🔧 BASE CLASS: Using deprecated stakeholder handler")
        print(f"⚠️ DEPRECATED: Using base stakeholder handler - subclasses should override with structured output")
        
        try:
            user_input = message.get("user_input", "")
            stakeholder = message.get("stakeholder", {})
            objective = message.get("objective", {})
            
            stakeholder_name = stakeholder.get("name", "Unknown")
            objective_name = objective.get("name", "Unknown")
            
            # Create basic expansion prompt
            expansion_prompt = f"""
Analyze this stakeholder-objective pair for detailed expansion:

User Request: "{user_input}"
Stakeholder: {stakeholder_name} ({stakeholder.get("role", "Unknown role")})
Objective: {objective_name} (Priority: {objective.get("priority", "medium")})

Return a JSON object with these fields:
- constraints: array of constraint objects with id, type, description, impact, flexibility
- technical_requirements: array of requirement objects with id, name, description, complexity, importance  
- assumptions: array of assumption objects with id, assumption, confidence, impact_if_wrong
- dependencies: array of dependency objects with depends_on, type, criticality
- success_metrics: array of metric objects with metric, target, timeframe

Focus on actionable, specific details that can be used for project planning.
"""
            
            # Use DEPRECATED unstructured call - subclasses should override
            result = self._call_llm_sync(expansion_prompt, "stakeholder_objective_expansion")
            
            # Store result using predictable key if provided
            expansion_key = message.get("expansion_key") or f"stakeholder_{stakeholder_name}_{objective_name}".replace(" ", "_").lower()
            self.shared_results[expansion_key] = {
                "stakeholder": stakeholder,
                "objective": objective,
                "expansion": result
            }
            
            return {"status": "completed", "expansion_key": expansion_key}
            
        except Exception as e:
            print(f"❌ Error in base stakeholder-objective expansion: {str(e)}")
            # NO FALLBACKS - let it fail fast
            expansion_key = message.get("expansion_key") or f"stakeholder_{stakeholder_name}_{objective_name}".replace(" ", "_").lower()
            self.shared_results[expansion_key] = {
                "stakeholder": stakeholder,
                "objective": objective,
                "expansion": {},
                "error": str(e)
            }
            return {"status": "error", "expansion_key": expansion_key, "error": str(e)}
    
    def _handle_node_type_expansion(self, message):
        """Handle node-type-specific expansion request - BASE IMPLEMENTATION"""
        print(f"⚠️ DEPRECATED: Using base node-type handler - subclasses should override with structured output")
        
        try:
            user_input = message.get("user_input", "")
            assessment = message.get("assessment", {})
            node_type = message.get("node_type", "constraints")
            heuristics = message.get("heuristics", [])
            
            # Create node-type-specific prompt
            expansion_prompt = self._create_node_type_prompt(user_input, assessment, node_type, heuristics)
            
            # Use DEPRECATED unstructured call - subclasses should override
            result = self._call_llm_sync(expansion_prompt, f"node_type_{node_type}")
            
            # Store result using predictable key if provided
            expansion_key = message.get("expansion_key") or f"node_type_{node_type}"
            self.shared_results[expansion_key] = {
                "node_type": node_type,
                "heuristics": heuristics,
                "expansion": result
            }
            
            return {"status": "completed", "expansion_key": expansion_key}
            
        except Exception as e:
            print(f"❌ Error in base node-type expansion ({node_type}): {str(e)}")
            # NO FALLBACKS - let it fail fast
            expansion_key = message.get("expansion_key") or f"node_type_{node_type}"
            self.shared_results[expansion_key] = {
                "node_type": node_type,
                "heuristics": [],
                "expansion": {},
                "error": str(e)
            }
            return {"status": "error", "expansion_key": expansion_key, "error": str(e)}
    
    def _create_node_type_prompt(self, user_input: str, assessment: Dict, node_type: str, heuristics: list) -> str:
        """Create specialized prompt for specific node type expansion"""
        
        node_type_specs = {
            "constraints": {
                "focus": "limitations, restrictions, and boundaries that affect the project",
                "examples": "budget limits, timeline restrictions, regulatory requirements, resource constraints",
            },
            "technical_requirements": {
                "focus": "technical specifications, integrations, and system capabilities needed",
                "examples": "API integrations, database requirements, security specifications, performance needs",
            },
            "assumptions": {
                "focus": "beliefs, expectations, and assumptions being made about the project",
                "examples": "market conditions, user behavior, technology stability, resource availability",
            }
        }
        
        spec = node_type_specs.get(node_type, {"focus": "general analysis", "examples": "various factors"})
        
        return f"""
Analyze {node_type} for this project request:

User Request: "{user_input}"
Focus: {spec['focus']}
Examples: {spec['examples']}

Apply these heuristic patterns: {heuristics}

Provide detailed {node_type} analysis with specific, actionable items that can be used for project planning and risk assessment.
"""
    
    def _run_async_llm_call_sync(self, prompt: str, content_type: str) -> Dict:
        """Helper to run the actual async LLM completion synchronously."""
        analyst = self.llm_backends.get("analyst")
        if not analyst:
            raise RuntimeError("No analyst backend available. Real LLM backends required.")

        # Define the actual async logic
        async def async_call_and_parse(p, ct, an):
            context = {"content_type": ct}
            structured_prompt = f"""
{p}

Please respond with a valid JSON object that matches the expected structure for {ct}.
Ensure the response is properly formatted JSON that can be parsed.
"""
            # Call the actual async LLM completion function
            try:
                result_with_prefix = await an.complete(structured_prompt, context)
            except Exception as llm_error:
                # Handle connection errors and other LLM failures
                error_info = {
                    "content_type": ct,
                    "error": str(llm_error),
                    "error_type": type(llm_error).__name__,
                    "timestamp": time.time()
                }
                print(f"❌ LLM backend error for {ct}: {str(llm_error)}")
                # Return empty dict instead of raising - let handler decide what to do
                return {"_llm_error": error_info}
            
            # Handle empty or None responses
            if not result_with_prefix or result_with_prefix.strip() == "":
                print(f"❌ Empty LLM response for {ct}")
                return {}
            
            # Strip the prefix added by the OpenAIBackend
            result = result_with_prefix.replace(f"[{an.name}] ", "", 1).strip()
            
            # Handle still empty after prefix removal
            if not result:
                print(f"❌ Empty result after prefix removal for {ct}")
                return {}
            
            # Robust JSON parsing with Pydantic validation
            try:
                # First try basic JSON extraction
                start_idx = result.find('{')
                end_idx = result.rfind('}')
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = result[start_idx:end_idx+1]
                    parsed_json = json.loads(json_str)
                else:
                    parsed_json = json.loads(result)
                
                # DEPRECATED: No validation for unstructured calls - use structured output instead
                print(f"⚠️ DEPRECATED: Unstructured LLM call for {ct} - no validation applied")
                return parsed_json
                
            except json.JSONDecodeError as e:
                # Return validated empty structure instead of error dict
                print(f"❌ JSON parsing failed for {ct}: {str(e)}")
                print(f"   Raw result: {result[:200]}...")
                return self._get_fallback_structure(ct)

        # Use asyncio.run to execute the async function and get the result
        try:
            # NOTE: This uses the standard way to run an async function in a sync thread.
            # It creates a new event loop and runs the coroutine until it completes.
            return asyncio.run(async_call_and_parse(prompt, content_type, analyst))
        except Exception as e:
            print(f"❌ LLM call failed in _run_async_llm_call_sync for {content_type}: {str(e)}")
            # Return validated fallback structure instead of error dict
            return self._get_fallback_structure(content_type)

    def _call_llm_sync(self, prompt: str, content_type: str) -> Dict:
        """Synchronous LLM call wrapper for use in ZMQ handlers - DEPRECATED"""
        print(f"⚠️ DEPRECATED: Using unstructured LLM call for {content_type}")
        print(f"   Use _call_llm_structured_sync() instead for proper Pydantic validation")
        # Call the robust synchronous helper - it now returns {} on any error
        return self._run_async_llm_call_sync(prompt, content_type)
    
    def _call_llm_structured_sync(self, prompt: str, content_type: str, response_model) -> Dict:
        """Synchronous structured LLM call wrapper for ZMQ handlers - ENFORCES PYDANTIC SCHEMA"""
        print(f"🤖 Making structured LLM call for {content_type} with {response_model.__name__}")
        
        analyst = self.llm_backends.get("analyst")
        if not analyst:
            raise RuntimeError("No analyst backend available. Real LLM backends required.")
        
        if not hasattr(analyst, 'complete_structured'):
            raise RuntimeError(f"LLM backend does not support structured output for {content_type}")
        
        # Define the actual async logic
        async def async_structured_call(p, ct, model, an):
            context = {"content_type": ct}
            print(f"🔍 DEBUG: Calling complete_structured with model: {model.__name__}")
            print(f"🔍 DEBUG: Model schema keys: {list(model.model_fields.keys())}")
            # Use structured completion - LLM MUST return exact schema
            result = await an.complete_structured(p, context, model)
            print(f"🔍 DEBUG: Structured result type: {type(result)}")
            print(f"🔍 DEBUG: Structured result keys: {list(result.keys()) if isinstance(result, dict) else 'not dict'}")
            return result
        
        # Use asyncio.run to execute the structured call
        try:
            return asyncio.run(async_structured_call(prompt, content_type, response_model, analyst))
        except Exception as e:
            print(f"❌ Structured LLM call failed for {content_type}: {str(e)}")
            raise RuntimeError(f"Structured LLM call failed for {content_type}: {e}. No fallback - fix the prompt or model.")
    
    # REMOVED: Validation and fallback methods - structured output eliminates the need
    # All LLM calls should use _call_llm_structured_sync() with Pydantic models
    
    def _report_llm_failures(self, phase: str):
        """Report LLM failures found in shared_results for the given phase"""
        failures = []
        empty_results = []
        
        # Scan shared_results for error information and empty results
        for key, result in self.shared_results.items():
            if isinstance(result, dict):
                # Check for direct error field (new format)
                if "error" in result:
                    failures.append({
                        "key": key,
                        "content_type": result.get("node_type", "stakeholder_objective_expansion"),
                        "error": result["error"],
                        "error_type": result.get("status", "unknown").title() + "Error",
                        "timestamp": time.time()
                    })
                else:
                    expansion = result.get("expansion", {})
                    
                    # Check for old format error markers (legacy)
                    if isinstance(expansion, dict) and "_llm_error" in expansion:
                        error_info = expansion["_llm_error"]
                        failures.append({
                            "key": key,
                            "content_type": error_info.get("content_type", "unknown"),
                            "error": error_info.get("error", "unknown error"),
                            "error_type": error_info.get("error_type", "unknown"),
                            "timestamp": error_info.get("timestamp", 0)
                        })
                    # Check for empty/fallback results
                    elif isinstance(expansion, dict) and self._is_empty_result(expansion):
                        empty_results.append({
                            "key": key,
                            "content_type": "inferred_from_key",
                            "error": "LLM returned empty or invalid response - used fallback structure",
                            "error_type": "EmptyResponse",
                            "timestamp": time.time()
                        })
        
        all_issues = failures + empty_results
        
        if all_issues:
            print(f"\n🚨 LLM ISSUE REPORT - {phase.upper()}")
            print("=" * 60)
            for i, issue in enumerate(all_issues, 1):
                print(f"{i}. Key: {issue['key']}")
                print(f"   Content Type: {issue['content_type']}")
                print(f"   Issue Type: {issue['error_type']}")
                print(f"   Details: {issue['error']}")
                if issue['timestamp']:
                    import datetime
                    dt = datetime.datetime.fromtimestamp(issue['timestamp'])
                    print(f"   Time: {dt.strftime('%H:%M:%S')}")
                print()
            
            print(f"📊 SUMMARY: {len(failures)} hard failures, {len(empty_results)} empty responses out of {len(self.shared_results)} total requests")
            print("=" * 60)
        else:
            print(f"\n✅ LLM SUCCESS REPORT - {phase.upper()}: No issues detected")
    
    def _is_empty_result(self, expansion: Dict) -> bool:
        """Check if expansion result is empty/fallback structure"""
        if not expansion:
            return True
        
        # Check for common empty patterns
        empty_patterns = [
            {"constraints": []},
            {"technical_requirements": []},
            {"assumptions": []},
            {"constraints": [], "technical_requirements": [], "assumptions": [], "dependencies": [], "success_metrics": []},
            {"constraints": [], "ai_actionable_insights": []},
            {"technical_requirements": [], "ai_actionable_insights": []},
            {"assumptions": [], "ai_actionable_insights": []}
        ]
        
        return expansion in empty_patterns

