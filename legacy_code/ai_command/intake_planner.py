#!/usr/bin/env python3
"""
Intake Planner - Major General's Strategic Assessment System
Analyzes user requests to determine optimal execution strategy
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional

from command_structures import (
    IntakeAssessment, IntentType, ComplexityLevel, 
    ExecutionStrategy, ZMQPattern
)

logger = logging.getLogger(__name__)

class IntakePlanner:
    """Major General's intake planning and strategy assessment system"""
    
    def __init__(self, meta_reasoning_engine):
        self.meta_reasoning = meta_reasoning_engine
        self.strategy_rules = self._initialize_strategy_rules()
        
    def _initialize_strategy_rules(self) -> Dict[str, Any]:
        """Initialize strategy selection rules"""
        return {
            "direct_response_triggers": [
                "what is", "define", "explain", "describe",
                "simple question", "basic concept"
            ],
            "tool_usage_triggers": [
                "calculate", "convert", "lookup", "search",
                "find documentation", "check status"
            ],
            "expert_consultation_triggers": [
                "security", "compliance", "architecture", "performance",
                "scalability", "best practices", "recommendations"
            ],
            "decomposition_triggers": [
                "build", "create", "design", "implement", "develop",
                "plan", "strategy", "roadmap", "system"
            ],
            "complexity_indicators": [
                "secure", "scalable", "enterprise", "production",
                "high-performance", "distributed", "microservices",
                "compliance", "regulations", "multi-", "complex"
            ]
        }
    
    async def assess_request(self, user_request: str) -> IntakeAssessment:
        """Perform comprehensive intake assessment"""
        
        # Create intake analysis prompt
        intake_prompt = self._create_intake_prompt(user_request)
        
        # Get meta-reasoning analysis
        analysis_result = await self.meta_reasoning.llm_backends["analyst"].complete(
            intake_prompt, {"request": user_request}
        )
        
        # Parse and structure the analysis
        structured_assessment = self._parse_analysis_result(analysis_result, user_request)
        
        # Apply strategy rules
        final_assessment = self._apply_strategy_rules(structured_assessment, user_request)
        
        logger.info(f"Intake assessment: {final_assessment.execution_strategy.value} "
                   f"(complexity: {final_assessment.complexity_level.value})")
        
        return final_assessment
    
    def _create_intake_prompt(self, user_request: str) -> str:
        """Create the intake analysis prompt for meta-reasoning"""
        
        return f"""
As Major General's Strategic Intake Analyst, assess this user request for optimal execution strategy.

USER REQUEST: "{user_request}"

Analyze the following dimensions:

1. INTENT CLASSIFICATION:
   - Information Request (what/how/why questions)
   - Analysis Request (analyze, evaluate, assess)
   - Design Request (design, architect, create)
   - Planning Request (plan, strategy, roadmap)
   - Problem Solving (solve, fix, troubleshoot)
   - Comparison Request (compare, versus, differences)
   - Recommendation Request (recommend, suggest, best)

2. COMPLEXITY ASSESSMENT:
   - Simple (1): Single domain, straightforward answer
   - Moderate (2): Multiple aspects, some interdependencies
   - Complex (3): Multi-domain, significant interdependencies
   - Expert (4): Requires deep specialization, critical decisions

3. DOMAIN ANALYSIS:
   - Technical domains involved (security, performance, architecture, etc.)
   - Business domains involved (strategy, compliance, operations, etc.)
   - Interdependencies between domains

4. RESOURCE REQUIREMENTS:
   - Can be answered with existing knowledge?
   - Requires specialist expertise?
   - Needs external research/tools?
   - Requires decomposition into subtasks?

5. EXECUTION STRATEGY RECOMMENDATION:
   - Direct Response: Single LLM can handle adequately
   - Tool Usage: Existing tools/knowledge base can answer
   - Sequential Analysis: Multiple specialists needed in sequence
   - Panel of Experts: Parallel expert consultation required
   - Full Decomposition: Complex DAG execution needed
   - Research Required: External knowledge gathering needed

Provide your assessment in this format:
INTENT: [intent_type]
COMPLEXITY: [1-4]
DOMAINS: [list of domains]
STRATEGY: [recommended_strategy]
SPECIALISTS: [list of required specialists if any]
TIME_ESTIMATE: [minutes]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation of your assessment]
"""

    def _parse_analysis_result(self, analysis_result: str, user_request: str) -> IntakeAssessment:
        """Parse the meta-reasoning analysis into structured assessment"""
        
        # Extract key information from analysis
        # This is a simplified parser - could be made more sophisticated
        
        try:
            # Default values
            intent_type = IntentType.INFORMATION_REQUEST
            complexity_level = ComplexityLevel.SIMPLE
            execution_strategy = ExecutionStrategy.DIRECT_RESPONSE
            required_specialists = []
            estimated_time = 1.0
            confidence = 0.8
            reasoning = analysis_result
            
            # Parse intent
            if any(word in user_request.lower() for word in ["analyze", "analysis", "assess"]):
                intent_type = IntentType.ANALYSIS_REQUEST
            elif any(word in user_request.lower() for word in ["design", "create", "build"]):
                intent_type = IntentType.DESIGN_REQUEST
            elif any(word in user_request.lower() for word in ["plan", "strategy", "roadmap"]):
                intent_type = IntentType.PLANNING_REQUEST
            elif any(word in user_request.lower() for word in ["recommend", "suggest", "best"]):
                intent_type = IntentType.RECOMMENDATION_REQUEST
            elif any(word in user_request.lower() for word in ["compare", "versus", "vs"]):
                intent_type = IntentType.COMPARISON_REQUEST
            elif any(word in user_request.lower() for word in ["solve", "fix", "problem"]):
                intent_type = IntentType.PROBLEM_SOLVING
            
            # Parse complexity based on keywords
            complexity_indicators = self.strategy_rules["complexity_indicators"]
            complexity_count = sum(1 for indicator in complexity_indicators 
                                 if indicator in user_request.lower())
            
            if complexity_count >= 3:
                complexity_level = ComplexityLevel.EXPERT_LEVEL
            elif complexity_count >= 2:
                complexity_level = ComplexityLevel.COMPLEX
            elif complexity_count >= 1:
                complexity_level = ComplexityLevel.MODERATE
            
            # Determine specialists needed
            if any(word in user_request.lower() for word in ["security", "secure"]):
                required_specialists.append("security_expert")
            if any(word in user_request.lower() for word in ["architecture", "design", "system"]):
                required_specialists.append("system_architect")
            if any(word in user_request.lower() for word in ["performance", "scalability", "speed"]):
                required_specialists.append("performance_engineer")
            if any(word in user_request.lower() for word in ["compliance", "regulation", "legal"]):
                required_specialists.append("compliance_officer")
            
            # Estimate time based on complexity
            time_estimates = {
                ComplexityLevel.SIMPLE: 1.0,
                ComplexityLevel.MODERATE: 3.0,
                ComplexityLevel.COMPLEX: 8.0,
                ComplexityLevel.EXPERT_LEVEL: 15.0
            }
            estimated_time = time_estimates[complexity_level]
            
            return IntakeAssessment(
                intent_type=intent_type,
                complexity_level=complexity_level,
                execution_strategy=execution_strategy,  # Will be refined by strategy rules
                required_specialists=required_specialists,
                estimated_time=estimated_time,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error parsing analysis result: {e}")
            # Return safe defaults
            return IntakeAssessment(
                intent_type=IntentType.INFORMATION_REQUEST,
                complexity_level=ComplexityLevel.SIMPLE,
                execution_strategy=ExecutionStrategy.DIRECT_RESPONSE,
                required_specialists=[],
                estimated_time=1.0,
                confidence=0.5,
                reasoning=f"Parse error: {e}"
            )
    
    def _apply_strategy_rules(self, assessment: IntakeAssessment, user_request: str) -> IntakeAssessment:
        """Apply strategy selection rules to refine the assessment"""
        
        request_lower = user_request.lower()
        
        # Rule 1: Simple information requests
        if (assessment.intent_type == IntentType.INFORMATION_REQUEST and 
            assessment.complexity_level == ComplexityLevel.SIMPLE):
            assessment.execution_strategy = ExecutionStrategy.DIRECT_RESPONSE
        
        # Rule 2: Tool usage triggers
        elif any(trigger in request_lower for trigger in self.strategy_rules["tool_usage_triggers"]):
            assessment.execution_strategy = ExecutionStrategy.TOOL_USAGE
        
        # Rule 3: Panel of Experts consultation (ZMQ Scatter-Gather)
        elif (len(assessment.required_specialists) >= 2 and 
              assessment.complexity_level >= ComplexityLevel.MODERATE):
            assessment.execution_strategy = ExecutionStrategy.PANEL_OF_EXPERTS
            assessment.zmq_pattern = ZMQPattern.SCATTER_GATHER
        
        # Rule 4: Sequential analysis for moderate complexity
        elif (len(assessment.required_specialists) >= 1 and 
              assessment.complexity_level == ComplexityLevel.MODERATE):
            assessment.execution_strategy = ExecutionStrategy.SEQUENTIAL_ANALYSIS
        
        # Rule 5: Full decomposition for complex builds/designs
        elif (assessment.intent_type in [IntentType.DESIGN_REQUEST, IntentType.PLANNING_REQUEST] and
              assessment.complexity_level >= ComplexityLevel.COMPLEX):
            assessment.execution_strategy = ExecutionStrategy.FULL_DECOMPOSITION
            assessment.decomposition_needed = True
        
        # Rule 6: Research required for unknown domains
        elif any(word in request_lower for word in ["latest", "current", "new", "recent"]):
            assessment.execution_strategy = ExecutionStrategy.RESEARCH_REQUIRED
            assessment.external_research_needed = True
        
        # Default: Direct response
        else:
            assessment.execution_strategy = ExecutionStrategy.DIRECT_RESPONSE
        
        return assessment
    
    def get_strategy_explanation(self, assessment: IntakeAssessment) -> str:
        """Get human-readable explanation of the chosen strategy"""
        
        explanations = {
            ExecutionStrategy.DIRECT_RESPONSE: 
                "This request can be handled with a single, comprehensive AI analysis.",
            
            ExecutionStrategy.TOOL_USAGE: 
                "This request can be answered using existing tools and knowledge bases.",
            
            ExecutionStrategy.SEQUENTIAL_ANALYSIS: 
                f"This request requires consultation with {len(assessment.required_specialists)} "
                f"specialists in sequence: {', '.join(assessment.required_specialists)}.",
            
            ExecutionStrategy.PANEL_OF_EXPERTS: 
                f"This request requires parallel consultation with multiple experts: "
                f"{', '.join(assessment.required_specialists)}.",
            
            ExecutionStrategy.FULL_DECOMPOSITION: 
                "This complex request requires breaking down into multiple coordinated tasks "
                "with specialized agents for each component.",
            
            ExecutionStrategy.RESEARCH_REQUIRED: 
                "This request requires gathering current information from external sources "
                "before analysis can proceed."
        }
        
        return explanations.get(assessment.execution_strategy, "Strategy explanation not available.")

# Demo function
async def demo_intake_planning():
    """Demo the intake planning system"""
    
    # Mock meta-reasoning for demo
    class MockMetaReasoning:
        class MockLLMBackend:
            async def complete(self, prompt, context):
                return "INTENT: analysis_request\nCOMPLEXITY: 3\nSTRATEGY: panel_of_experts"
        
        def __init__(self):
            self.llm_backends = {"analyst": self.MockLLMBackend()}
    
    # Test requests
    test_requests = [
        "What is a REST API?",
        "Build a secure fintech API for 10,000 transactions per second",
        "Compare microservices vs monolithic architecture",
        "Analyze the security implications of our current system",
        "Design a scalable e-commerce platform with real-time analytics"
    ]
    
    planner = IntakePlanner(MockMetaReasoning())
    
    print("🎖️  MAJOR GENERAL INTAKE PLANNING DEMO")
    print("=" * 60)
    
    for request in test_requests:
        print(f"\n👤 USER REQUEST: {request}")
        
        assessment = await planner.assess_request(request)
        
        print(f"📊 ASSESSMENT:")
        print(f"   Intent: {assessment.intent_type.value}")
        print(f"   Complexity: {assessment.complexity_level.value} ({assessment.complexity_level.value})")
        print(f"   Strategy: {assessment.execution_strategy.value}")
        print(f"   Specialists: {assessment.required_specialists}")
        print(f"   Time Estimate: {assessment.estimated_time} minutes")
        print(f"   Confidence: {assessment.confidence:.1%}")
        
        explanation = planner.get_strategy_explanation(assessment)
        print(f"💡 STRATEGY: {explanation}")

if __name__ == "__main__":
    asyncio.run(demo_intake_planning())
