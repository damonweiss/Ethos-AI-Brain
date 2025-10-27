"""
Adaptive Intent Analyzer - Modern Intent Analysis Implementation
Built on the clean inference engine architecture
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field

from ..inference_engines.llm_engine import LLMEngine
from ...knowledge.common.knowledge_graph import KnowledgeGraph, GraphType


class ComplexityLevel(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


class IntentType(Enum):
    QUESTION = "question"
    TASK = "task"
    ANALYSIS = "analysis"
    CREATION = "creation"
    PROBLEM_SOLVING = "problem_solving"
    PLANNING = "planning"


# Pydantic models for structured responses
class IntentObjective(BaseModel):
    name: str = Field(description="Clear objective name")
    priority: str = Field(pattern="^(critical|high|medium|low)$")
    description: str = Field(description="Detailed objective description")
    success_criteria: List[str] = Field(description="How to measure success")


class IntentStakeholder(BaseModel):
    name: str = Field(description="Stakeholder name or role")
    role: str = Field(description="Their role in this intent")
    objectives: List[IntentObjective] = Field(description="Their specific objectives")
    influence_level: str = Field(pattern="^(high|medium|low)$")


class IntentConstraint(BaseModel):
    constraint_type: str = Field(description="Type of constraint")
    description: str = Field(description="Detailed constraint description")
    impact_level: str = Field(pattern="^(blocking|high|medium|low)$")
    mitigation_strategies: List[str] = Field(description="Ways to work around this constraint")


class AIActionableInsight(BaseModel):
    insight: str = Field(description="The insight or recommendation")
    actionable_by_ai: bool = Field(description="Can this be executed by AI agent with tools?")
    required_tools: List[str] = Field(description="Tools/APIs needed for AI execution")
    human_oversight_required: bool = Field(description="Requires human approval/input?")
    automation_level: str = Field(pattern="^(fully_automated|human_in_loop|human_required)$")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this insight")


class SimpleIntentResponse(BaseModel):
    complexity: str = Field(pattern="^simple$")
    intent_type: IntentType
    direct_answer: str = Field(description="Direct response to the intent")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Why this is considered simple")


class ComplexIntentResponse(BaseModel):
    complexity: str = Field(pattern="^(moderate|complex|highly_complex)$")
    intent_type: IntentType
    stakeholders: List[IntentStakeholder]
    objectives: List[IntentObjective]
    constraints: List[IntentConstraint]
    ai_actionable_insights: List[AIActionableInsight]
    recommended_approach: str = Field(description="Recommended approach to address this intent")
    estimated_effort: str = Field(description="Estimated time/effort required")


class AdaptiveIntentAnalyzer:
    """
    Modern intent analysis system built on clean inference architecture
    """
    
    def __init__(self):
        # Use the new clean architecture components
        self.llm_engine = LLMEngine(model="gpt-4o")
        
        # Analysis history
        self.analysis_history: Dict[str, Dict[str, Any]] = {}
    
    async def analyze_intent(self, user_input: str, context: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Main intent analysis entry point
        
        Returns:
            Tuple of (analysis_type, analysis_result)
            analysis_type: "simple_response" or "complex_analysis"
        """
        analysis_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            # Phase 1: Complexity assessment
            complexity_result = await self._assess_complexity(user_input, context)
            
            # Phase 2: Detailed analysis based on complexity
            if complexity_result.get("complexity") == "simple":
                analysis_type = "simple_response"
                analysis_result = await self._handle_simple_intent(user_input, context, complexity_result)
            else:
                analysis_type = "complex_analysis"
                analysis_result = await self._handle_complex_intent(user_input, context, complexity_result)
            
            # Store analysis history
            self.analysis_history[analysis_id] = {
                "user_input": user_input,
                "context": context,
                "analysis_type": analysis_type,
                "result": analysis_result,
                "timestamp": start_time,
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
            
            return analysis_type, analysis_result
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "analysis_id": analysis_id
            }
            return "error", error_result
    
    async def _assess_complexity(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Assess the complexity of the user input
        """
        complexity_prompt = f"""
        Analyze the complexity of this user request:
        
        User Input: "{user_input}"
        Context: {context or "None provided"}
        
        Determine if this is:
        - SIMPLE: Direct question with straightforward answer, single-step task
        - MODERATE: Multi-step task, requires some analysis, clear scope
        - COMPLEX: Multi-stakeholder, multiple objectives, requires planning
        - HIGHLY_COMPLEX: Strategic planning, multiple domains, long-term implications
        
        Consider:
        - Number of steps required
        - Number of stakeholders involved
        - Scope and scale of the request
        - Dependencies and constraints
        - Required expertise domains
        
        Respond with JSON format:
        {{
            "complexity": "simple|moderate|complex|highly_complex",
            "reasoning": "explanation of complexity assessment",
            "key_factors": ["factor1", "factor2", ...],
            "estimated_steps": number,
            "confidence": 0.0-1.0
        }}
        """
        
        result = self.llm_engine.run(
            input_data=complexity_prompt,
            schema={},
            model_metadata={}
        )
        
        if result.get("success"):
            # Extract the response content
            result_data = result.get("result", {})
            response_content = result_data.get("message", result_data.get("response", ""))
            
            try:
                # Try to parse as JSON
                import json
                if response_content:
                    return json.loads(response_content)
                else:
                    return {"complexity": "moderate", "reasoning": "Unable to parse response", "confidence": 0.5}
            except json.JSONDecodeError:
                # Fallback to text analysis
                if "simple" in response_content.lower():
                    return {"complexity": "simple", "reasoning": response_content, "confidence": 0.7}
                elif "complex" in response_content.lower():
                    return {"complexity": "complex", "reasoning": response_content, "confidence": 0.7}
                else:
                    return {"complexity": "moderate", "reasoning": response_content, "confidence": 0.6}
        else:
            return {"complexity": "moderate", "reasoning": "Analysis failed", "confidence": 0.3}
    
    async def _handle_simple_intent(self, user_input: str, context: Dict[str, Any], complexity_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle simple intents with direct responses
        """
        simple_prompt = f"""
        Provide a direct, helpful response to this simple request:
        
        User Input: "{user_input}"
        Context: {context or "None"}
        
        Complexity Assessment: {complexity_result.get("reasoning", "")}
        
        Provide a clear, concise, and actionable response. Be direct and helpful.
        If this is a question, answer it clearly.
        If this is a task, provide step-by-step guidance.
        """
        
        result = self.llm_engine.run(
            input_data=simple_prompt,
            schema={},
            model_metadata={}
        )
        
        if result.get("success"):
            result_data = result.get("result", {})
            response_content = result_data.get("message", result_data.get("response", ""))
            
            return {
                "success": True,
                "complexity": "simple",
                "intent_type": self._infer_intent_type(user_input),
                "direct_answer": response_content,
                "confidence": complexity_result.get("confidence", 0.8),
                "reasoning": complexity_result.get("reasoning", ""),
                "metadata": {
                    "cost": result.get("cost", 0),
                    "tokens": result.get("usage", {}).get("total_tokens", 0)
                }
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Failed to generate simple response")
            }
    
    async def _handle_complex_intent(self, user_input: str, context: Dict[str, Any], complexity_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle complex intents with detailed stakeholder analysis
        """
        complex_prompt = f"""
        Perform a comprehensive analysis of this complex request:
        
        User Input: "{user_input}"
        Context: {context or "None"}
        Complexity: {complexity_result.get("complexity", "complex")}
        Reasoning: {complexity_result.get("reasoning", "")}
        
        Provide a detailed analysis including:
        
        1. STAKEHOLDERS: Who are the key stakeholders involved?
           - Their roles and responsibilities
           - Their objectives and priorities
           - Their influence level (high/medium/low)
        
        2. OBJECTIVES: What are the main objectives?
           - Priority level (critical/high/medium/low)
           - Success criteria
           - Dependencies between objectives
        
        3. CONSTRAINTS: What limitations exist?
           - Resource constraints
           - Time constraints
           - Technical constraints
           - Policy/regulatory constraints
           - Impact level (blocking/high/medium/low)
        
        4. AI ACTIONABLE INSIGHTS: What can AI help with?
           - Specific actions AI can take
           - Required tools/APIs
           - Human oversight needs
           - Automation level (fully_automated/human_in_loop/human_required)
           - Confidence in each insight (0.0-1.0)
        
        5. RECOMMENDED APPROACH: How should this be tackled?
        
        6. ESTIMATED EFFORT: Time and resource requirements
        
        Be thorough and practical. Focus on actionable insights.
        """
        
        result = self.llm_engine.run(
            input_data=complex_prompt,
            schema={},
            model_metadata={}
        )
        
        if result.get("success"):
            result_data = result.get("result", {})
            response_content = result_data.get("message", result_data.get("response", ""))
            
            # For now, return structured text analysis
            # In the future, this could be enhanced with structured JSON parsing
            return {
                "success": True,
                "complexity": complexity_result.get("complexity", "complex"),
                "intent_type": self._infer_intent_type(user_input),
                "detailed_analysis": response_content,
                "stakeholder_count": self._estimate_stakeholders(response_content),
                "objective_count": self._estimate_objectives(response_content),
                "constraint_count": self._estimate_constraints(response_content),
                "ai_actionable_count": self._estimate_ai_actions(response_content),
                "confidence": complexity_result.get("confidence", 0.7),
                "metadata": {
                    "cost": result.get("cost", 0),
                    "tokens": result.get("usage", {}).get("total_tokens", 0),
                    "complexity_factors": complexity_result.get("key_factors", [])
                }
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Failed to generate complex analysis")
            }
    
    def _infer_intent_type(self, user_input: str) -> str:
        """
        Infer the type of intent from user input
        """
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ["what", "how", "why", "when", "where", "who", "?"]):
            return IntentType.QUESTION.value
        elif any(word in user_lower for word in ["analyze", "analysis", "examine", "evaluate"]):
            return IntentType.ANALYSIS.value
        elif any(word in user_lower for word in ["create", "build", "make", "generate", "design"]):
            return IntentType.CREATION.value
        elif any(word in user_lower for word in ["plan", "strategy", "roadmap", "schedule"]):
            return IntentType.PLANNING.value
        elif any(word in user_lower for word in ["solve", "fix", "resolve", "problem"]):
            return IntentType.PROBLEM_SOLVING.value
        else:
            return IntentType.TASK.value
    
    def _estimate_stakeholders(self, analysis_text: str) -> int:
        """Estimate number of stakeholders from analysis text"""
        # Simple heuristic - count stakeholder-related keywords
        stakeholder_keywords = ["stakeholder", "user", "customer", "team", "department", "role"]
        count = sum(analysis_text.lower().count(keyword) for keyword in stakeholder_keywords)
        return min(count, 10)  # Cap at reasonable number
    
    def _estimate_objectives(self, analysis_text: str) -> int:
        """Estimate number of objectives from analysis text"""
        objective_keywords = ["objective", "goal", "target", "aim", "purpose"]
        count = sum(analysis_text.lower().count(keyword) for keyword in objective_keywords)
        return min(count, 15)
    
    def _estimate_constraints(self, analysis_text: str) -> int:
        """Estimate number of constraints from analysis text"""
        constraint_keywords = ["constraint", "limitation", "restriction", "barrier", "challenge"]
        count = sum(analysis_text.lower().count(keyword) for keyword in constraint_keywords)
        return min(count, 10)
    
    def _estimate_ai_actions(self, analysis_text: str) -> int:
        """Estimate number of AI actionable insights from analysis text"""
        ai_keywords = ["ai can", "automate", "automated", "tool", "api", "insight"]
        count = sum(analysis_text.lower().count(keyword) for keyword in ai_keywords)
        return min(count, 12)
    
    def get_analysis_history(self) -> Dict[str, Dict[str, Any]]:
        """
        Get complete analysis history
        """
        return self.analysis_history
    
    def get_recent_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent analyses, sorted by timestamp
        """
        analyses = list(self.analysis_history.values())
        analyses.sort(key=lambda x: x["timestamp"], reverse=True)
        return analyses[:limit]
