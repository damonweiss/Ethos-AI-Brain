"""
Reasoning Manager - Modern Meta-Reasoning Implementation
Built on the clean inference engine architecture
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum

from ..inference_engines.llm_engine import LLMEngine
from ..inference_model_manager.inference_model_manager import InferenceModelManager
from ...core.prompt_routing_manager.prompt_routing_manager import PromptRoutingManager


class ConfidenceLevel(Enum):
    HIGH = "high"      # >90% - autonomous execution
    MEDIUM = "medium"  # 70-90% - human review optional
    LOW = "low"        # <70% - human input required
    UNKNOWN = "unknown"


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
class ReasoningStep:
    """Individual step in reasoning process"""
    step_id: str
    description: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    confidence: Optional[ConfidenceLevel] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


class ReasoningManager:
    """
    Modern meta-reasoning engine built on clean inference architecture
    """
    
    def __init__(self):
        # Use the new clean architecture components
        self.prompt_router = PromptRoutingManager()
        self.model_manager = InferenceModelManager(auto_discover=True)
        self.llm_engine = LLMEngine(model="gpt-4o")
        
        # Session management
        self.active_sessions: Dict[str, ReasoningContext] = {}
        self.reasoning_history: Dict[str, List[ReasoningStep]] = {}
    
    async def reason(self, goal: str, context: ReasoningContext = None) -> Dict[str, Any]:
        """
        Main reasoning entry point - modern implementation
        """
        if context is None:
            context = ReasoningContext(goal=goal)
            
        self.active_sessions[context.session_id] = context
        self.reasoning_history[context.session_id] = []
        
        try:
            # Phase 1: Goal analysis and decomposition
            decomposition = await self._decompose_goal(goal, context)
            
            # Phase 2: Plan generation using LLM
            plan = await self._generate_plan(decomposition, context)
            
            # Phase 3: Execution with inference engines
            results = await self._execute_plan(plan, context)
            
            # Phase 4: Result synthesis
            final_result = await self._synthesize_results(results, context)
            
            return {
                "success": True,
                "session_id": context.session_id,
                "goal": goal,
                "result": final_result,
                "steps": len(self.reasoning_history[context.session_id]),
                "metadata": {
                    "execution_time": (datetime.now() - context.timestamp).total_seconds(),
                    "confidence": self._calculate_overall_confidence(context.session_id)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "session_id": context.session_id,
                "error": str(e),
                "steps_completed": len(self.reasoning_history.get(context.session_id, []))
            }
    
    async def _decompose_goal(self, goal: str, context: ReasoningContext) -> Dict[str, Any]:
        """
        Decompose goal into manageable steps using LLM
        """
        decomposition_prompt = f"""
        Analyze this goal and break it down into logical steps:
        
        Goal: {goal}
        
        Consider:
        - What are the main components of this goal?
        - What dependencies exist between steps?
        - What information or resources are needed?
        - What are potential challenges or constraints?
        
        Provide a structured breakdown with clear, actionable steps.
        """
        
        result = await self._execute_llm_step(
            step_id=f"decompose_{context.session_id}",
            description="Goal decomposition",
            prompt=decomposition_prompt,
            context=context
        )
        
        return result
    
    async def _generate_plan(self, decomposition: Dict[str, Any], context: ReasoningContext) -> Dict[str, Any]:
        """
        Generate execution plan based on decomposition
        """
        planning_prompt = f"""
        Based on this goal decomposition, create a detailed execution plan:
        
        Decomposition: {decomposition}
        
        Create a plan that includes:
        - Ordered sequence of actions
        - Resource requirements for each step
        - Success criteria
        - Risk mitigation strategies
        - Confidence estimates
        
        Focus on actionable steps that can be executed systematically.
        """
        
        result = await self._execute_llm_step(
            step_id=f"plan_{context.session_id}",
            description="Plan generation",
            prompt=planning_prompt,
            context=context
        )
        
        return result
    
    async def _execute_plan(self, plan: Dict[str, Any], context: ReasoningContext) -> List[Dict[str, Any]]:
        """
        Execute the generated plan step by step
        """
        results = []
        
        # For now, simulate plan execution by analyzing each step
        execution_prompt = f"""
        Execute this plan systematically:
        
        Plan: {plan}
        Goal: {context.goal}
        
        For each step in the plan:
        1. Analyze what needs to be done
        2. Identify required resources or information
        3. Execute or simulate the step
        4. Evaluate the outcome
        5. Adjust if necessary
        
        Provide detailed results for each step.
        """
        
        result = await self._execute_llm_step(
            step_id=f"execute_{context.session_id}",
            description="Plan execution",
            prompt=execution_prompt,
            context=context
        )
        
        results.append(result)
        return results
    
    async def _synthesize_results(self, results: List[Dict[str, Any]], context: ReasoningContext) -> Dict[str, Any]:
        """
        Synthesize execution results into final answer
        """
        synthesis_prompt = f"""
        Synthesize these execution results into a comprehensive final answer:
        
        Original Goal: {context.goal}
        Execution Results: {results}
        
        Provide:
        - Clear answer to the original goal
        - Summary of key findings
        - Confidence assessment
        - Recommendations for next steps
        - Any limitations or caveats
        
        Make the response actionable and complete.
        """
        
        result = await self._execute_llm_step(
            step_id=f"synthesize_{context.session_id}",
            description="Result synthesis",
            prompt=synthesis_prompt,
            context=context
        )
        
        return result
    
    async def _execute_llm_step(self, step_id: str, description: str, prompt: str, context: ReasoningContext) -> Dict[str, Any]:
        """
        Execute a single LLM reasoning step using the new architecture
        """
        start_time = datetime.now()
        
        step = ReasoningStep(
            step_id=step_id,
            description=description,
            parameters={"prompt": prompt},
            timestamp=start_time
        )
        
        try:
            # Use the modern LLM engine
            result = self.llm_engine.run(
                input_data=prompt,
                schema={},
                model_metadata={}
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if result.get("success"):
                step.result = result
                step.confidence = ConfidenceLevel.HIGH
                step.execution_time = execution_time
            else:
                step.error = result.get("error", "Unknown error")
                step.confidence = ConfidenceLevel.LOW
            
            # Add to reasoning history
            self.reasoning_history[context.session_id].append(step)
            
            return result
            
        except Exception as e:
            step.error = str(e)
            step.confidence = ConfidenceLevel.LOW
            self.reasoning_history[context.session_id].append(step)
            raise
    
    def _calculate_overall_confidence(self, session_id: str) -> str:
        """
        Calculate overall confidence based on step confidences
        """
        steps = self.reasoning_history.get(session_id, [])
        if not steps:
            return ConfidenceLevel.UNKNOWN.value
        
        confidences = [step.confidence for step in steps if step.confidence]
        if not confidences:
            return ConfidenceLevel.UNKNOWN.value
        
        # Simple majority vote
        high_count = sum(1 for c in confidences if c == ConfidenceLevel.HIGH)
        total_count = len(confidences)
        
        if high_count / total_count >= 0.7:
            return ConfidenceLevel.HIGH.value
        elif high_count / total_count >= 0.4:
            return ConfidenceLevel.MEDIUM.value
        else:
            return ConfidenceLevel.LOW.value
    
    def get_session_history(self, session_id: str) -> List[ReasoningStep]:
        """
        Get reasoning history for a session
        """
        return self.reasoning_history.get(session_id, [])
    
    def get_active_sessions(self) -> List[str]:
        """
        Get list of active session IDs
        """
        return list(self.active_sessions.keys())
