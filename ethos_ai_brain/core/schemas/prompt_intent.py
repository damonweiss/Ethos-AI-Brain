"""
Prompt Intent Schema - Pydantic models for intent analysis.

Defines schemas for analyzing user prompts and extracting intent information.
"""

from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, validator


class IntentConfidence(str, Enum):
    """Intent confidence levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IntentCategory(str, Enum):
    """Categories of user intent."""
    INFORMATION_SEEKING = "information_seeking"
    TASK_EXECUTION = "task_execution"
    PROBLEM_SOLVING = "problem_solving"
    CREATIVE_GENERATION = "creative_generation"
    ANALYSIS_REQUEST = "analysis_request"
    PLANNING_REQUEST = "planning_request"
    DECISION_SUPPORT = "decision_support"


class IntentEntity(BaseModel):
    """Individual entity extracted from intent analysis."""
    entity_type: str = Field(description="Type of entity (goal, constraint, resource, action, stakeholder)")
    value: str = Field(description="The actual entity value")
    importance: float = Field(ge=0.0, le=1.0, description="Importance score for this entity")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in entity extraction")


class IntentRelationship(BaseModel):
    """Relationship between intent entities."""
    source_entity: str = Field(description="Source entity in the relationship")
    target_entity: str = Field(description="Target entity in the relationship")
    relationship_type: str = Field(description="Type of relationship (depends_on, requires, enables, etc.)")
    strength: float = Field(ge=0.0, le=1.0, description="Strength of the relationship")


class IntentAnalysis(BaseModel):
    """Comprehensive intent analysis result."""
    primary_intent: str = Field(description="Main user objective")
    secondary_intents: List[str] = Field(default_factory=list, description="Additional objectives")
    intent_category: IntentCategory = Field(description="Category of the primary intent")
    intent_confidence: IntentConfidence = Field(description="Confidence level in intent understanding")
    
    # Extracted entities
    goals: List[IntentEntity] = Field(default_factory=list, description="Goal entities")
    constraints: List[IntentEntity] = Field(default_factory=list, description="Constraint entities")
    resources: List[IntentEntity] = Field(default_factory=list, description="Resource entities")
    actions: List[IntentEntity] = Field(default_factory=list, description="Action entities")
    stakeholders: List[IntentEntity] = Field(default_factory=list, description="Stakeholder entities")
    
    # Relationships
    relationships: List[IntentRelationship] = Field(default_factory=list, description="Relationships between entities")
    
    # Analysis metadata
    success_criteria: List[str] = Field(default_factory=list, description="How to measure success")
    assumptions: List[str] = Field(default_factory=list, description="Implicit assumptions made")
    ambiguities: List[str] = Field(default_factory=list, description="Unclear aspects needing clarification")
    complexity_score: float = Field(ge=1.0, le=5.0, description="Complexity assessment 1-5")
    
    @validator('complexity_score')
    def validate_complexity_score(cls, v):
        return max(1.0, min(5.0, float(v)))
    
    @property
    def total_entities(self) -> int:
        """Total number of entities extracted."""
        return len(self.goals) + len(self.constraints) + len(self.resources) + len(self.actions) + len(self.stakeholders)
    
    @property
    def entity_summary(self) -> Dict[str, int]:
        """Summary of entity counts by type."""
        return {
            "goals": len(self.goals),
            "constraints": len(self.constraints),
            "resources": len(self.resources),
            "actions": len(self.actions),
            "stakeholders": len(self.stakeholders)
        }


class IntentScopeResult(BaseModel):
    """Result of initial intent scoping analysis."""
    can_answer_directly: bool = Field(description="Whether this can be answered with simple knowledge")
    direct_answer: Optional[str] = Field(None, description="Complete answer if can_answer_directly is True")
    
    # Intent analysis (if cannot answer directly)
    intent_analysis: Optional[IntentAnalysis] = Field(None, description="Detailed intent analysis")
    
    # Processing requirements
    requires_human_input: bool = Field(default=False, description="Needs human clarification")
    requires_external_data: bool = Field(default=False, description="Needs external data sources")
    estimated_complexity: IntentConfidence = Field(description="Estimated processing complexity")
    
    # Next steps
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended next steps")
    required_capabilities: List[str] = Field(default_factory=list, description="Required system capabilities")


class IntentGap(BaseModel):
    """Information gap identified during intent analysis."""
    gap_type: str = Field(description="Type of information gap")
    description: str = Field(description="Description of what information is missing")
    importance: IntentConfidence = Field(description="Importance of filling this gap")
    suggested_questions: List[str] = Field(default_factory=list, description="Questions to ask to fill the gap")
    can_proceed_without: bool = Field(description="Whether processing can continue without this information")
