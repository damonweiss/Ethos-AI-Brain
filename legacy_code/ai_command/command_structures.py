#!/usr/bin/env python3
"""
Command Structures - Shared data structures for AI Command System
Used by Major General, Platoon Leaders, and all agents
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime

class IntentType(Enum):
    INFORMATION_REQUEST = "information_request"      # "What is X?"
    ANALYSIS_REQUEST = "analysis_request"           # "Analyze X"
    DESIGN_REQUEST = "design_request"               # "Design X"
    PLANNING_REQUEST = "planning_request"           # "Plan X"
    PROBLEM_SOLVING = "problem_solving"             # "How do I solve X?"
    COMPARISON_REQUEST = "comparison_request"       # "Compare X vs Y"
    RECOMMENDATION_REQUEST = "recommendation_request" # "Recommend X for Y"

class ComplexityLevel(Enum):
    SIMPLE = 1          # Single domain, straightforward
    MODERATE = 2        # Multiple aspects, some complexity
    COMPLEX = 3         # Multi-domain, interdependent
    EXPERT_LEVEL = 4    # Requires deep specialization

class ExecutionStrategy(Enum):
    DIRECT_RESPONSE = "direct_response"             # Single LLM call
    TOOL_USAGE = "tool_usage"                      # Use existing tools/knowledge
    SEQUENTIAL_ANALYSIS = "sequential_analysis"     # Multiple specialists in sequence
    PANEL_OF_EXPERTS = "panel_of_experts"          # Parallel expert consultation (ZMQ Scatter-Gather)
    PLATOON_LEADER = "platoon_leader"              # Spawn tactical commander
    FULL_DECOMPOSITION = "full_decomposition"      # Complex DAG execution
    RESEARCH_REQUIRED = "research_required"        # Need external knowledge

class ZMQPattern(Enum):
    REQ_REP = "req_rep"                    # Request-Reply
    PUB_SUB = "pub_sub"                    # Publish-Subscribe
    PUSH_PULL = "push_pull"                # Pipeline
    DEALER_ROUTER = "dealer_router"        # Advanced routing
    SCATTER_GATHER = "scatter_gather"      # Panel of Experts pattern

@dataclass
class IntakeAssessment:
    """Shared assessment structure for all command levels"""
    intent_type: IntentType
    complexity_level: ComplexityLevel
    execution_strategy: ExecutionStrategy
    required_specialists: List[str]
    estimated_time: float  # minutes
    confidence: float      # 0.0 to 1.0
    reasoning: str
    suggested_tools: List[str] = None
    decomposition_needed: bool = False
    external_research_needed: bool = False
    zmq_pattern: ZMQPattern = ZMQPattern.REQ_REP
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.suggested_tools is None:
            self.suggested_tools = []

@dataclass
class CommandDirective:
    """Command from Major General to subordinates"""
    directive_id: str
    target: str                    # Agent ID or "panel_of_experts"
    mission_component: str         # Part of mission for this agent
    context: Dict[str, Any]        # Relevant context
    intake_assessment: IntakeAssessment
    priority: int = 1
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class ExpertConsultation:
    """Request for Panel of Experts consultation"""
    consultation_id: str
    mission_objective: str
    specific_question: str
    required_experts: List[str]
    intake_assessment: IntakeAssessment
    scatter_pattern: ZMQPattern = ZMQPattern.SCATTER_GATHER
    timeout_seconds: float = 30.0
    
@dataclass
class ExpertResponse:
    """Response from individual expert"""
    expert_type: str
    consultation_id: str
    analysis: str
    confidence: float
    recommendations: List[str]
    follow_up_needed: bool = False
    
@dataclass
class PlatoonMission:
    """Mission assignment for Platoon Leader"""
    platoon_id: str
    mission_component: str
    tactical_objective: str
    intake_assessment: IntakeAssessment
    allocated_resources: Dict[str, Any]
    reporting_schedule: str = "on_completion"
