#!/usr/bin/env python3
"""
AI_Brain - Pure Cognitive Intelligence System
Provides core thinking, reasoning, memory, and learning capabilities
Role-agnostic brain that can be used by any AI agent
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from command_structures import (
    IntakeAssessment, ExpertConsultation, CommandDirective,
    PlatoonMission, ExecutionStrategy, IntentType, ComplexityLevel
)
from meta_reasoning_engine import MetaReasoningEngine, ReasoningContext
from panel_of_experts import PanelOfExperts

logger = logging.getLogger(__name__)

"""
AI_Brain - Pure Cognitive Intelligence System
Provides core thinking, reasoning, memory, and learning capabilities
Role-agnostic brain that can be used by any AI agent
"""

class CognitiveCapability:
    """Defines core cognitive capabilities available to any brain"""
    META_REASONING = "meta_reasoning"
    STRATEGIC_ANALYSIS = "strategic_analysis"
    TACTICAL_ANALYSIS = "tactical_analysis"
    EXPERT_CONSULTATION = "expert_consultation"
    SITUATION_ASSESSMENT = "situation_assessment"
    MEMORY_ACCESS = "memory_access"
    LEARNING = "learning"
    PATTERN_RECOGNITION = "pattern_recognition"
    DECISION_MAKING = "decision_making"

class AI_Brain:
    """
    Pure Cognitive Intelligence System
    Role-agnostic brain providing core thinking capabilities
    """
    
    def __init__(self, capabilities: List[str] = None):
        self.capabilities = capabilities or [
            CognitiveCapability.META_REASONING,
            CognitiveCapability.MEMORY_ACCESS,
            CognitiveCapability.LEARNING,
            CognitiveCapability.DECISION_MAKING
        ]
        
        # Core cognitive systems - USE REAL AI
        self.meta_reasoning = MetaReasoningEngine(use_real_ai=True)
        self.panel_of_experts = None  # Initialized on demand
        
        # Intake assessment strategy rules
        self.strategy_rules = {
            "complexity_indicators": [
                "multi", "complex", "enterprise", "scalable", "distributed",
                "integration", "migration", "architecture", "security",
                "compliance", "performance", "optimization"
            ],
            "tool_usage_triggers": [
                "search", "lookup", "find", "research", "data", "database",
                "api", "external", "fetch", "retrieve"
            ],
            "expert_consultation_triggers": [
                "expert", "specialist", "consultation", "review", "audit",
                "assessment", "evaluation", "analysis"
            ]
        }
        
        # Memory systems
        self.episodic_memory = {}    # Experience history
        self.semantic_memory = {}    # Knowledge base
        self.working_memory = {}     # Current context
        self.procedural_memory = {}  # Learned procedures
        
        # Cognitive state
        self.attention_focus = None
        self.cognitive_load = 0.0
        self.decision_history = []
        self.learning_metrics = {}
        
        # Thinking patterns
        self.reasoning_patterns = {}
        self.problem_solving_strategies = []
        
        logger.info(f"🧠 AI_Brain initialized with capabilities: {capabilities}")
    
    def set_specialization_context(self, specialization: str):
        """Set cognitive specialization context (called by agent)"""
        self.specialization_context = specialization
        
        # Store in semantic memory
        self.semantic_memory['specialization'] = {
            'context': specialization,
            'timestamp': datetime.now()
        }
    
    async def think(self, prompt: str, context: Dict[str, Any] = None, specialization: str = None) -> Dict[str, Any]:
        """
        Core thinking function - pure cognitive reasoning
        Specialization context provided by the agent, not hardcoded in brain
        """
        # Update cognitive load
        self.cognitive_load += 0.1
        self.attention_focus = prompt
        
        # Create reasoning context
        reasoning_context = ReasoningContext(
            goal=prompt,
            constraints=context.get('constraints', {}) if context else {},
            user_preferences=context.get('preferences', {}) if context else {},
            metadata={
                'specialization': specialization or self.semantic_memory.get('specialization', {}).get('context'),
                'capabilities': self.capabilities,
                'cognitive_load': self.cognitive_load
            }
        )
        
        # Apply specialization if provided
        if specialization:
            specialized_prompt = f"Context: {specialization}\n\nTask: {prompt}"
        else:
            specialized_prompt = prompt
        
        # Use meta-reasoning
        thinking_result = await self.meta_reasoning.reason(specialized_prompt, reasoning_context)
        
        # Store in episodic memory
        thought_id = f"thought_{uuid.uuid4().hex[:8]}"
        self.episodic_memory[thought_id] = {
            'prompt': prompt,
            'context': context,
            'specialization': specialization,
            'result': thinking_result,
            'timestamp': datetime.now(),
            'cognitive_load': self.cognitive_load
        }
        
        # Update working memory (limited capacity)
        self.working_memory[thought_id] = thinking_result
        if len(self.working_memory) > 10:  # Limit working memory
            oldest_key = min(self.working_memory.keys())
            del self.working_memory[oldest_key]
        
        # Reduce cognitive load
        self.cognitive_load = max(0.0, self.cognitive_load - 0.05)
        
        return thinking_result
    
    async def assess_situation(self, situation: str) -> IntakeAssessment:
        """
        Cognitive assessment of situation complexity and strategy
        Integrated intake planning directly in brain
        """
        if CognitiveCapability.SITUATION_ASSESSMENT not in self.capabilities:
            raise ValueError("Brain does not have situation assessment capability")
        
        # Update cognitive state
        self.attention_focus = f"Assessing: {situation}"
        self.cognitive_load += 0.1
        
        # Create intake analysis prompt
        intake_prompt = self._create_intake_prompt(situation)
        
        print(f"\n🧠 BRAIN CALLING AI ANALYST FOR INTAKE ASSESSMENT:")
        print(f"📋 Situation: {situation}")
        print(f"🎯 Using: {self.meta_reasoning.llm_backends['analyst'].__class__.__name__}")
        
        # Get meta-reasoning analysis
        analysis_result = await self.meta_reasoning.llm_backends["analyst"].complete(
            intake_prompt, {"request": situation}
        )
        
        # Parse and structure the analysis
        structured_assessment = self._parse_analysis_result(analysis_result, situation)
        
        # Apply strategy rules
        final_assessment = self._apply_strategy_rules(structured_assessment, situation)
        
        # Store assessment in semantic memory
        assessment_id = f"assessment_{uuid.uuid4().hex[:8]}"
        self.semantic_memory[assessment_id] = {
            'situation': situation,
            'assessment': final_assessment.__dict__,
            'timestamp': datetime.now()
        }
        
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
              assessment.complexity_level in [ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT_LEVEL]):
            assessment.execution_strategy = ExecutionStrategy.PANEL_OF_EXPERTS
        
        # Rule 4: Sequential analysis for moderate complexity
        elif (assessment.complexity_level == ComplexityLevel.MODERATE and 
              len(assessment.required_specialists) >= 1):
            assessment.execution_strategy = ExecutionStrategy.SEQUENTIAL_ANALYSIS
        
        # Rule 5: Full decomposition for expert-level complexity
        elif assessment.complexity_level == ComplexityLevel.EXPERT_LEVEL:
            assessment.execution_strategy = ExecutionStrategy.FULL_DECOMPOSITION
        
        # Rule 6: Research required triggers
        elif any(trigger in request_lower for trigger in ["research", "latest", "current", "recent"]):
            assessment.execution_strategy = ExecutionStrategy.RESEARCH_REQUIRED
        
        return assessment
    
    async def enhanced_intake(self, mission: str) -> Dict[str, Any]:
        """
        Enhanced intake protocol - pure cognitive assessment
        Determines optimal approach without role-specific context
        """
        if CognitiveCapability.SITUATION_ASSESSMENT not in self.capabilities:
            raise ValueError("Brain does not have situation assessment capability")
        
        # Update cognitive state
        self.attention_focus = f"Enhanced intake: {mission}"
        self.cognitive_load += 0.1
        
        # Import schema for JSON prompting
        from dag_models import get_intake_response_schema, IntakeStrategy
        
        # Enhanced intake protocol with JSON prompting
        intake_protocol_prompt = f"""
You are a cognitive intake processor. Analyze this question and recommend the optimal approach.

QUESTION: "{mission}"

EVALUATION CRITERIA:
1. Can this be answered with direct LLM knowledge? (facts, definitions, simple calculations)
2. Does it require tools? (calculator, search, database lookup)
3. Is it ambiguous and needs user clarification?
4. Does it require expert consultation?
5. Is it complex enough for decomposition?
6. Would this benefit from MULTIPLE sequential approaches? (decomposition → expert consultation → synthesis)

IMPORTANT: For complex projects/planning questions, prefer MULTI_STAGE_STRATEGY over single approaches.

RESPOND WITH VALID JSON using this exact schema:
{get_intake_response_schema()}

Available strategies:
- direct_answer: Simple facts with >90% certainty
- tool_required: Calculator, search, database lookup needed
- hitl_clarification: Need user clarification/more context
- expert_consultation: Requires domain specialist input
- task_decomposition: Break into smaller tasks (specify decomposition_type)
- multi_stage_strategy: Multiple sequential approaches needed
- research_required: External information gathering needed

Decomposition types:
- simple: Break down task without expert input (straightforward subtasks)
- expert_guided: Break down with expert guidance (complex domain knowledge needed)
- hierarchical: Multi-level breakdown (nested subtasks)
- parallel: Independent parallel tasks (no dependencies)

Example responses:

For direct answers:
{{
  "strategy": "direct_answer",
  "reasoning": "Simple factual question with known answer",
  "parameters": {{"answer": "The actual answer to the question"}},
  "confidence": 0.95
}}

For expert consultation:
{{
  "strategy": "expert_consultation",
  "reasoning": "Requires specialized domain expertise",
  "parameters": {{
    "experts": [
      {{
        "role": "Operations Specialist",
        "domain": "business_operations",
        "system_prompt": "You are an Operations Specialist with 15 years of experience in business process optimization. Focus on efficiency, workflow analysis, and operational improvements. Provide specific, actionable recommendations.",
        "authority": "Can recommend process changes and efficiency improvements",
        "consultation_focus": "Analyze current operations and identify improvement opportunities"
      }},
      {{
        "role": "Financial Analyst", 
        "domain": "finance",
        "system_prompt": "You are a Senior Financial Analyst specializing in cost reduction and financial optimization. Focus on budget analysis, cost-benefit evaluation, and financial impact assessment.",
        "authority": "Can recommend financial strategies and budget allocations",
        "consultation_focus": "Evaluate financial implications and cost reduction opportunities"
      }}
    ],
    "consultation_question": "How should we approach this problem?",
    "coordination_method": "consensus"
  }},
  "confidence": 0.85
}}

For simple decomposition:
{{
  "strategy": "task_decomposition",
  "reasoning": "Task can be broken into clear, independent subtasks",
  "parameters": {{
    "decomposition_type": "simple",
    "subtasks": ["subtask1", "subtask2", "subtask3"],
    "expert_guidance": false
  }},
  "confidence": 0.80
}}

For expert-guided decomposition:
{{
  "strategy": "task_decomposition", 
  "reasoning": "Task requires domain expertise to properly decompose",
  "parameters": {{
    "decomposition_type": "expert_guided",
    "expert_guidance": true,
    "experts": [
      {{
        "role": "Domain Expert",
        "system_prompt": "You are a specialist in this domain...",
        "consultation_focus": "Help break down this complex task"
      }}
    ]
  }},
  "confidence": 0.85
}}

For complex multi-stage tasks:
{{
  "strategy": "multi_stage_strategy",
  "reasoning": "Complex planning question requiring decomposition followed by expert consultation and synthesis",
  "parameters": {{"complexity": "high", "domains": ["operations", "finance"]}},
  "confidence": 0.85
}}

IMPORTANT: For direct_answer strategy, ALWAYS include the actual answer in parameters.answer

RESPOND ONLY WITH VALID JSON.
"""
        
        # Get cognitive assessment
        response = await self.meta_reasoning.llm_backends["analyst"].complete(
            intake_protocol_prompt, {"mission": mission}
        )
        
        # Parse JSON response using Pydantic
        from dag_models import validate_intake_response
        
        try:
            # Debug: Show what we're trying to parse
            logger.info(f"Parsing intake response: {response[:200]}...")
            
            # Validate and parse the JSON response
            intake_response = validate_intake_response(response)
            
            intake_result = {
                "mission": mission,
                "raw_response": response,
                "strategy": intake_response.strategy.value,
                "parameters": intake_response.parameters,
                "reasoning": intake_response.reasoning,
                "confidence": intake_response.confidence
            }
            
        except ValueError as e:
            logger.warning(f"Failed to parse JSON intake response: {e}")
            logger.warning(f"Raw response was: {response}")
            
            # Fallback to legacy parsing
            intake_result = {
                "mission": mission,
                "raw_response": response,
                "strategy": "parse_error",
                "parameters": {"error": str(e), "raw_response": response[:500]},
                "reasoning": "Failed to parse JSON response",
                "confidence": 0.0
            }
        
        # Store in semantic memory
        intake_id = f"intake_{uuid.uuid4().hex[:8]}"
        self.semantic_memory[intake_id] = {
            'mission': mission,
            'intake_result': intake_result,
            'timestamp': datetime.now()
        }
        
        logger.info(f"Enhanced intake complete: {intake_result['strategy']}")
        return intake_result
    
    async def consult_experts(self, consultation: ExpertConsultation) -> Dict[str, Any]:
        """
        Cognitive expert consultation capability
        """
        if CognitiveCapability.EXPERT_CONSULTATION not in self.capabilities:
            raise ValueError("Brain does not have expert consultation capability")
        
        # Update cognitive state
        self.attention_focus = f"Consulting experts: {consultation.specific_question}"
        self.cognitive_load += 0.2  # Expert consultation is cognitively intensive
        
        # Initialize Panel of Experts if needed
        if self.panel_of_experts is None:
            self.panel_of_experts = PanelOfExperts()
            await self.panel_of_experts.initialize_panel()
        
        expert_responses = await self.panel_of_experts.consult_panel(consultation)
        
        # Store consultation in episodic memory
        consultation_id = consultation.consultation_id
        self.episodic_memory[consultation_id] = {
            'consultation': consultation.__dict__,
            'responses': expert_responses,
            'timestamp': datetime.now()
        }
        
        return expert_responses
    
    async def delegate_mission(self, mission_component: str, target_agent: str) -> CommandDirective:
        """
        Create delegation directive for subordinate agents
        Only available to agents with agent_delegation capability
        """
        if CognitiveCapability.AGENT_DELEGATION not in self.capabilities:
            raise ValueError("Brain does not have delegation capability")
        
        # Assess the mission component
        assessment = await self.assess_situation(mission_component)
        
        # Create command directive
        directive = CommandDirective(
            directive_id=f"cmd_{uuid.uuid4().hex[:8]}",
            target=target_agent,
            mission_component=mission_component,
            context=self.working_memory,
            intake_assessment=assessment
        )
        
        # Store in episodic memory
        self.episodic_memory[directive.directive_id] = directive
        
        return directive
    
    async def plan_tactical_mission(self, mission: str, resources: Dict[str, Any]) -> PlatoonMission:
        """
        Create tactical mission plan
        Only available to agents with tactical_planning capability
        """
        if CognitiveCapability.TACTICAL_ANALYSIS not in self.capabilities:
            raise ValueError("Brain does not have tactical analysis capability")
        
        # Assess mission from tactical perspective
        assessment = await self.assess_situation(mission)
        
        # Create tactical plan
        tactical_plan = await self.think(
            f"Create tactical execution plan for: {mission}",
            {"resources": resources, "planning_type": "tactical"}
        )
        
        # Create platoon mission
        platoon_mission = PlatoonMission(
            platoon_id=f"pl_{uuid.uuid4().hex[:8]}",
            mission_component=mission,
            tactical_objective=tactical_plan.get('objective', mission),
            intake_assessment=assessment,
            allocated_resources=resources
        )
        
        return platoon_mission
    
    def learn_from_experience(self, experience: Dict[str, Any]):
        """
        Learn from mission outcomes and update performance
        """
        if CognitiveCapability.LEARNING not in self.capabilities:
            return
        
        # Store in episodic memory
        experience_id = f"exp_{uuid.uuid4().hex[:8]}"
        self.episodic_memory[experience_id] = {
            'experience': experience,
            'timestamp': datetime.now(),
            'learning_context': 'experience_integration'
        }
        
        # Update performance metrics (simplified)
        if 'success' in experience:
            if 'success_rate' not in self.performance_metrics:
                self.performance_metrics['success_rate'] = []
            self.performance_metrics['success_rate'].append(experience['success'])
    
    def get_cognitive_status(self) -> Dict[str, Any]:
        """Get current cognitive status and metrics"""
        return {
            'capabilities': self.capabilities,
            'working_memory_items': len(self.working_memory),
            'episodic_memory_items': len(self.episodic_memory),
            'semantic_memory_items': len(self.semantic_memory),
            'procedural_memory_items': len(self.procedural_memory),
            'decisions_made': len(self.decision_history),
            'cognitive_load': self.cognitive_load,
            'attention_focus': self.attention_focus,
            'learning_metrics': self.learning_metrics,
            'meta_reasoning_backends': len(self.meta_reasoning.llm_backends)
        }
    
    async def shutdown(self):
        """Shutdown brain systems"""
        if self.panel_of_experts:
            await self.panel_of_experts.shutdown()
        
        logger.info(f"🧠 AI_Brain shutdown complete")

# Demo function for pure brain functionality
async def demo_ai_brain():
    """Demo pure AI_Brain cognitive capabilities"""
    
    print("🧠 AI_BRAIN COGNITIVE DEMO")
    print("=" * 40)
    
    # Create brain with different capability sets
    strategic_brain = AI_Brain([
        CognitiveCapability.META_REASONING,
        CognitiveCapability.STRATEGIC_ANALYSIS,
        CognitiveCapability.SITUATION_ASSESSMENT,
        CognitiveCapability.EXPERT_CONSULTATION,
        CognitiveCapability.MEMORY_ACCESS,
        CognitiveCapability.LEARNING
    ])
    
    tactical_brain = AI_Brain([
        CognitiveCapability.META_REASONING,
        CognitiveCapability.TACTICAL_ANALYSIS,
        CognitiveCapability.MEMORY_ACCESS,
        CognitiveCapability.LEARNING
    ])
    
    print(f"✅ Created brains with different capabilities")
    
    # Test thinking with specialization
    problem = "Analyze cybersecurity threats"
    
    print(f"\n🧠 Strategic Brain thinking about: {problem}")
    strategic_thought = await strategic_brain.think(
        problem, 
        specialization="Strategic military commander focused on high-level threat assessment"
    )
    
    print(f"🧠 Tactical Brain thinking about: {problem}")
    tactical_thought = await tactical_brain.think(
        problem,
        specialization="Tactical cybersecurity specialist focused on immediate threats"
    )
    
    # Show cognitive status
    print(f"\n📊 COGNITIVE STATUS:")
    strategic_status = strategic_brain.get_cognitive_status()
    tactical_status = tactical_brain.get_cognitive_status()
    
    print(f"   Strategic Brain: {strategic_status['working_memory_items']} thoughts, "
          f"load: {strategic_status['cognitive_load']:.2f}")
    print(f"   Tactical Brain: {tactical_status['working_memory_items']} thoughts, "
          f"load: {tactical_status['cognitive_load']:.2f}")
    
    # Cleanup
    await strategic_brain.shutdown()
    await tactical_brain.shutdown()

if __name__ == "__main__":
    asyncio.run(demo_ai_brain())
