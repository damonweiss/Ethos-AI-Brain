#!/usr/bin/env python3
"""
Pydantic models for DAG-based AI reasoning
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import json

class ApproachType(str, Enum):
    """Types of AI reasoning approaches"""
    DECOMPOSITION = "decomposition"
    EXPERT_CONSULTATION = "expert_consultation"
    HITL_CLARIFICATION = "hitl_clarification"
    RESEARCH = "research"
    SYNTHESIS = "synthesis"
    TOOL_REQUIRED = "tool_required"

class IntakeStrategy(str, Enum):
    """Brain intake strategy types"""
    DIRECT_ANSWER = "direct_answer"
    TOOL_REQUIRED = "tool_required"
    HITL_CLARIFICATION = "hitl_clarification"
    EXPERT_CONSULTATION = "expert_consultation"
    TASK_DECOMPOSITION = "task_decomposition"
    MULTI_STAGE_STRATEGY = "multi_stage_strategy"
    RESEARCH_REQUIRED = "research_required"

class DecompositionType(str, Enum):
    """Types of task decomposition"""
    SIMPLE = "simple"           # Break down without expert input
    EXPERT_GUIDED = "expert_guided"  # Break down with expert guidance
    HIERARCHICAL = "hierarchical"    # Multi-level breakdown
    PARALLEL = "parallel"       # Independent parallel tasks

class DAGNode(BaseModel):
    """Individual node in the DAG"""
    id: str = Field(..., description="Unique node identifier")
    approach: ApproachType = Field(..., description="Type of AI reasoning approach")
    purpose: str = Field(..., description="What this node accomplishes")
    resources: str = Field(..., description="AI capabilities needed")
    dependencies: List[str] = Field(default_factory=list, description="Node IDs this depends on")
    parallel_with: List[str] = Field(default_factory=list, description="Node IDs that can run in parallel")
    success_criteria: str = Field(..., description="How to know this node is complete")

    @field_validator('dependencies', mode='before')
    @classmethod
    def parse_dependencies(cls, v):
        if isinstance(v, str):
            if v.lower() in ['none', '']:
                return []
            return [dep.strip() for dep in v.split(',')]
        return v

    @field_validator('parallel_with', mode='before')
    @classmethod
    def parse_parallel_with(cls, v):
        if isinstance(v, str):
            if v.lower() in ['none', '']:
                return []
            return [dep.strip() for dep in v.split(',')]
        return v

class DAGPlan(BaseModel):
    """Complete DAG operation plan"""
    nodes: List[DAGNode] = Field(..., description="All nodes in the DAG")
    critical_path: str = Field(..., description="Sequence determining minimum completion")
    parallel_opportunities: str = Field(..., description="Which nodes can execute simultaneously")
    coordination_strategy: str = Field(..., description="How to manage parallel execution")
    risk_mitigation: str = Field(..., description="Failure handling strategies")
    commanding_officer: str = Field(..., description="Agent role creating this plan")

    def to_networkx(self):
        """Convert to NetworkX DAG"""
        import networkx as nx
        
        dag = nx.DiGraph()
        
        # Add nodes with attributes
        for node in self.nodes:
            dag.add_node(node.id, **node.dict())
        
        # Add dependency edges
        for node in self.nodes:
            for dep in node.dependencies:
                if dep and dep in [n.id for n in self.nodes]:
                    dag.add_edge(dep, node.id)
        
        return dag

class IntakeResponse(BaseModel):
    """Brain intake assessment response"""
    strategy: IntakeStrategy = Field(..., description="Recommended approach")
    reasoning: str = Field(..., description="Why this strategy was chosen")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy-specific parameters")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in assessment")

class ToolRequest(BaseModel):
    """Tool usage request"""
    tool: str = Field(..., description="Tool name (calculator, search, database)")
    query: str = Field(..., description="Specific query for tool")
    reasoning: str = Field(..., description="Why this tool is needed")

class HITLRequest(BaseModel):
    """Human-in-the-loop clarification request"""
    questions: List[str] = Field(..., description="Specific questions for user")
    context_needed: str = Field(..., description="What context is missing")
    reasoning: str = Field(..., description="Why clarification is needed")

class ExpertProfile(BaseModel):
    """Individual expert profile with ready-to-use prompt"""
    role: str = Field(..., description="Expert role/title")
    domain: str = Field(..., description="Area of expertise")
    system_prompt: str = Field(..., description="Complete LLM system prompt for this expert")
    authority: str = Field(..., description="What decisions this expert can make")
    consultation_focus: str = Field(..., description="What specific aspect they should address")

class ExpertConsultation(BaseModel):
    """Expert consultation request with ready-to-use expert prompts"""
    experts: List[ExpertProfile] = Field(..., description="Ready-to-deploy expert profiles")
    consultation_question: str = Field(..., description="Specific question for experts")
    coordination_method: str = Field(..., description="How to coordinate expert responses")
    reasoning: str = Field(..., description="Why expert consultation is needed")

# JSON Schema generators for prompts
def get_dag_node_schema() -> str:
    """Get JSON schema for DAG nodes to include in prompts"""
    return json.dumps(DAGNode.model_json_schema(), indent=2)

def get_intake_response_schema() -> str:
    """Get JSON schema for intake responses"""
    return json.dumps(IntakeResponse.model_json_schema(), indent=2)

def get_dag_plan_schema() -> str:
    """Get JSON schema for complete DAG plans"""
    return json.dumps(DAGPlan.model_json_schema(), indent=2)

# Validation helpers
def validate_dag_response(response_text: str) -> DAGPlan:
    """Validate and parse DAG response from LLM"""
    try:
        # Clean the response - extract JSON if it's embedded in other text
        cleaned_response = extract_json_from_text(response_text)
        data = json.loads(cleaned_response)
        return DAGPlan(**data)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Invalid DAG response format: {e}")

def validate_intake_response(response_text: str) -> IntakeResponse:
    """Validate and parse intake response from LLM"""
    try:
        # Clean the response - extract JSON if it's embedded in other text
        cleaned_response = extract_json_from_text(response_text)
        data = json.loads(cleaned_response)
        return IntakeResponse(**data)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Invalid intake response format: {e}")

def extract_json_from_text(text: str) -> str:
    """Extract JSON from text that might contain other content"""
    text = text.strip()
    
    # Handle common prefixes like "[Analyst]"
    if text.startswith('[') and ']' in text:
        bracket_end = text.find(']')
        if bracket_end != -1:
            # Remove the prefix and try the rest
            remaining_text = text[bracket_end + 1:].strip()
            if remaining_text.startswith('{'):
                text = remaining_text
    
    # If it starts with a bracket, assume it's pure JSON
    if text.startswith('{') or text.startswith('['):
        # Try to parse as-is first
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            # If it fails, try to fix truncated JSON
            return fix_truncated_json(text)
    
    # Look for JSON embedded in text
    import re
    
    # Find JSON object pattern - more flexible and greedy
    json_pattern = r'\{.*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if matches:
        # Try each match to see if it's valid JSON
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                # Try to fix truncated JSON
                fixed = fix_truncated_json(match)
                try:
                    json.loads(fixed)
                    return fixed
                except json.JSONDecodeError:
                    continue
    
    # If no valid JSON found, try to find content between first { and last }
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        potential_json = text[start_idx:end_idx + 1]
        try:
            json.loads(potential_json)
            return potential_json
        except json.JSONDecodeError:
            # Try to fix truncated JSON
            fixed = fix_truncated_json(potential_json)
            try:
                json.loads(fixed)
                return fixed
            except json.JSONDecodeError:
                pass
    
    # Last resort: try to fix the entire text as truncated JSON
    return fix_truncated_json(text)

def fix_truncated_json(text: str) -> str:
    """Attempt to fix truncated JSON by closing open structures"""
    text = text.strip()
    
    if not text.startswith('{'):
        return text
    
    # Count open/close braces and brackets
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')
    
    # Count open/close quotes (for strings)
    quote_count = text.count('"')
    
    # If we have unmatched quotes, try to close the string
    if quote_count % 2 == 1:
        text += '"'
    
    # Close any open arrays
    while open_brackets > close_brackets:
        text += ']'
        close_brackets += 1
    
    # Close any open objects
    while open_braces > close_braces:
        text += '}'
        close_braces += 1
    
    return text
