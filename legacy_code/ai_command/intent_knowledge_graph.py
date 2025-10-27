#!/usr/bin/env python3
"""
Intent Knowledge Graph - Specialized knowledge graph for user intent capture and analysis
Provides intent-specific query methods for requirements, constraints, and context analysis
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from knowledge_graph import KnowledgeGraph, GraphType

logger = logging.getLogger(__name__)

class GapType(Enum):
    """Types of gaps that can be identified in intent analysis"""
    RESOURCE = "resource"           # Missing people, budget, time, tools
    SKILL = "skill"                # Missing expertise or capabilities  
    KNOWLEDGE = "knowledge"        # Missing information or understanding
    HILT = "hilt"                  # Human In The Loop requirements
    TECHNOLOGY = "technology"      # Missing technical infrastructure
    PROCESS = "process"            # Missing workflows or procedures
    STAKEHOLDER = "stakeholder"    # Missing buy-in or decision authority
    EXTERNAL = "external"          # Dependencies on external parties

class GapSeverity(Enum):
    """Severity levels for identified gaps"""
    CRITICAL = "critical"          # Project cannot proceed without addressing
    HIGH = "high"                  # Significant impact on success
    MEDIUM = "medium"              # Moderate impact, workarounds possible
    LOW = "low"                    # Minor impact, nice to have

# ========================================
# Pydantic Models for Intent Graph Schema
# ========================================

class IntentObjective(BaseModel):
    """Schema for intent objectives"""
    id: str = Field(..., description="Unique identifier for the objective")
    name: str = Field(..., description="Human-readable name of the objective")
    description: Optional[str] = Field(None, description="Detailed description of what needs to be achieved")
    priority: str = Field(..., pattern="^(critical|high|medium|low)$", description="Priority level")
    measurable: bool = Field(False, description="Whether this objective can be measured")
    measurement_method: Optional[str] = Field(None, description="How to measure success")
    target_value: Optional[float] = Field(None, description="Target value for measurement")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    complexity: Optional[str] = Field("medium", pattern="^(low|medium|high|very_high)$")

class IntentConstraint(BaseModel):
    """Schema for intent constraints"""
    id: str = Field(..., description="Unique identifier for the constraint")
    constraint_type: str = Field(..., pattern="^(budget|timeline|technical|resource|regulatory|accessibility)$")
    constraint: str = Field(..., description="Description of the constraint")
    value: Optional[float] = Field(None, description="Numeric value if applicable")
    flexibility: str = Field("rigid", pattern="^(rigid|somewhat_flexible|flexible)$")
    time_pressure: Optional[str] = Field(None, pattern="^(low|moderate|tight|unrealistic)$")
    budget_breakdown: Optional[Dict[str, float]] = Field(None, description="Detailed budget allocation")

class IntentStakeholder(BaseModel):
    """Schema for intent stakeholders"""
    id: str = Field(..., description="Unique identifier for the stakeholder")
    name: str = Field(..., description="Name of the stakeholder")
    role: str = Field(..., description="Role in the project")
    influence_level: str = Field("medium", pattern="^(high|medium|low)$")
    support_level: str = Field("neutral", pattern="^(strong|cautious|neutral|weak|opposed|unknown)$")
    decision_authority: str = Field("none", pattern="^(final|approval|input|none)$")
    communication_preferences: Optional[Dict[str, Any]] = Field(None)

class IntentTechnicalRequirement(BaseModel):
    """Schema for technical requirements"""
    id: str = Field(..., description="Unique identifier for the requirement")
    name: str = Field(..., description="Name of the technical requirement")
    description: str = Field(..., description="Detailed description")
    complexity: str = Field("medium", pattern="^(low|medium|high|very_high)$")
    importance: str = Field("medium", pattern="^(critical|high|medium|low)$")

class IntentAssumption(BaseModel):
    """Schema for assumptions"""
    id: str = Field(..., description="Unique identifier for the assumption")
    assumption: str = Field(..., description="The assumption being made")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level (0-1)")
    impact_if_wrong: str = Field("medium", pattern="^(critical|high|medium|low)$")
    validation_method: Optional[str] = Field(None, description="How to validate this assumption")

class IntentDerivationData(BaseModel):
    """Schema for intent derivation metadata"""
    raw_user_prompt: str = Field("", description="Original user input")
    domain_context: str = Field("unknown", description="Identified domain")
    user_sentiment: str = Field("neutral", pattern="^(positive|neutral|negative|urgent|frustrated|excited)$")
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    ambiguities: List[Dict[str, Any]] = Field(default_factory=list)
    clarifying_questions: List[Dict[str, Any]] = Field(default_factory=list)
    implicit_requirements: List[Dict[str, Any]] = Field(default_factory=list)

class IntentGraphSchema(BaseModel):
    """Complete schema for Intent Knowledge Graph JSON"""
    graph_metadata: Dict[str, Any] = Field(..., description="Graph identification and metadata")
    intent_data: IntentDerivationData = Field(..., description="Intent derivation metadata")
    objectives: List[IntentObjective] = Field(default_factory=list, description="Primary and secondary objectives")
    constraints: List[IntentConstraint] = Field(default_factory=list, description="Project constraints")
    stakeholders: List[IntentStakeholder] = Field(default_factory=list, description="Project stakeholders")
    technical_requirements: List[IntentTechnicalRequirement] = Field(default_factory=list, description="Technical requirements")
    assumptions: List[IntentAssumption] = Field(default_factory=list, description="Project assumptions")
    relationships: List[Dict[str, str]] = Field(default_factory=list, description="Node relationships")
    
    @field_validator('objectives')
    @classmethod
    def validate_objectives(cls, v):
        if not v:
            raise ValueError("At least one objective is required")
        return v

class IntentKnowledgeGraph(KnowledgeGraph):
    """
    Specialized knowledge graph for user intent capture and analysis
    Provides intent-specific query methods that don't make sense for execution or RAG graphs
    """
    
    def __init__(self, graph_id: str, intent_name: str = None):
        """
        Initialize intent knowledge graph
        
        Args:
            graph_id: Unique identifier for this graph
            intent_name: Human-readable intent name
        """
        super().__init__(GraphType.INTENT, graph_id)
        self.intent_name = intent_name or graph_id
        self.intent_created_date = datetime.now()
        self.last_analysis_date = None
        
        # Intent derivation data structures (populated by agent brain)
        self.raw_user_prompt = ""
        self.domain_context = "unknown"
        self.intent_confidence_scores = {}  # {element_type: confidence_score}
        self.ambiguities = []  # List of ambiguous elements needing clarification
        self.clarifying_questions = []  # Questions to resolve ambiguities
        self.implicit_requirements = []  # Requirements not explicitly stated
        self.user_sentiment = "neutral"  # Overall user sentiment/urgency
        self.prompt_processing_metadata = {}  # LLM processing details
        
        logger.info(f"IntentKnowledgeGraph created: {graph_id} ({self.intent_name})")
    
    # ========================================
    # Intent-Specific Query Methods
    # ========================================
    
    def analyze_user_requirements(self) -> Dict[str, Any]:
        """
        Extract and structure user goals and needs from the intent graph
        Returns comprehensive requirements analysis
        """
        analysis = {
            'primary_objectives': [],
            'secondary_objectives': [],
            'success_criteria': [],
            'user_preferences': {},
            'stakeholder_needs': {},
            'functional_requirements': [],
            'non_functional_requirements': [],
            'business_context': {},
            'user_background': {}
        }
        
        if not self.nodes():
            return analysis
        
        # Analyze nodes for different requirement types
        for node_id in self.nodes():
            attrs = self.get_node_attributes(node_id)
            node_type = attrs.get('type', 'unknown')
            
            if node_type == 'primary_objective':
                analysis['primary_objectives'].append({
                    'id': node_id,
                    'description': attrs.get('description', ''),
                    'priority': attrs.get('priority', 'medium'),
                    'measurable': attrs.get('measurable', False)
                })
            elif node_type == 'success_criteria':
                analysis['success_criteria'].append({
                    'id': node_id,
                    'criteria': attrs.get('criteria', ''),
                    'measurement': attrs.get('measurement', ''),
                    'target_value': attrs.get('target_value')
                })
            elif node_type == 'user_preference':
                category = attrs.get('category', 'general')
                if category not in analysis['user_preferences']:
                    analysis['user_preferences'][category] = []
                analysis['user_preferences'][category].append({
                    'preference': attrs.get('preference', ''),
                    'importance': attrs.get('importance', 'medium'),
                    'rationale': attrs.get('rationale', '')
                })
        
        self.last_analysis_date = datetime.now()
        logger.info(f"Requirements analysis: {len(analysis['primary_objectives'])} objectives, "
                   f"{len(analysis['success_criteria'])} criteria")
        
        return analysis
    
    def identify_constraint_conflicts(self) -> List[Dict[str, Any]]:
        """
        Find conflicting or impossible requirements in the intent graph
        Returns list of constraint conflicts with severity and resolution suggestions
        """
        conflicts = []
        
        if not self.nodes():
            return conflicts
        
        # Find constraint nodes
        constraints = []
        for node_id in self.nodes():
            attrs = self.get_node_attributes(node_id)
            if attrs.get('type') == 'constraint':
                constraints.append({
                    'id': node_id,
                    'constraint': attrs.get('constraint', ''),
                    'constraint_type': attrs.get('constraint_type', 'unknown'),
                    'value': attrs.get('value'),
                    'flexibility': attrs.get('flexibility', 'rigid')
                })
        
        # Check for conflicts between constraints
        for i, constraint_a in enumerate(constraints):
            for constraint_b in constraints[i+1:]:
                conflict = self._detect_constraint_conflict(constraint_a, constraint_b)
                if conflict:
                    conflicts.append(conflict)
        
        # Check for impossible constraints
        for constraint in constraints:
            impossibility = self._detect_impossible_constraint(constraint)
            if impossibility:
                conflicts.append(impossibility)
        
        # Sort by severity
        conflicts.sort(key=lambda x: self._get_conflict_severity_score(x['severity']), reverse=True)
        
        logger.info(f"Found {len(conflicts)} constraint conflicts")
        return conflicts
    
    def assess_feasibility(self) -> Dict[str, Any]:
        """
        Evaluate achievability of user goals based on constraints and context
        Returns feasibility assessment with confidence scores
        """
        assessment = {
            'overall_feasibility': 0.0,
            'feasibility_factors': {},
            'risk_factors': [],
            'recommended_adjustments': [],
            'confidence_score': 0.0,
            'assessment_date': datetime.now().isoformat()
        }
        
        if not self.nodes():
            assessment['overall_feasibility'] = 1.0
            assessment['confidence_score'] = 0.0
            return assessment
        
        # Analyze different feasibility dimensions
        technical_feasibility = self._assess_technical_feasibility()
        resource_feasibility = self._assess_resource_feasibility()
        timeline_feasibility = self._assess_timeline_feasibility()
        stakeholder_feasibility = self._assess_stakeholder_feasibility()
        
        assessment['feasibility_factors'] = {
            'technical': technical_feasibility,
            'resource': resource_feasibility,
            'timeline': timeline_feasibility,
            'stakeholder': stakeholder_feasibility
        }
        
        # Calculate overall feasibility (weighted average)
        weights = {'technical': 0.3, 'resource': 0.25, 'timeline': 0.25, 'stakeholder': 0.2}
        assessment['overall_feasibility'] = sum(
            assessment['feasibility_factors'][factor] * weight
            for factor, weight in weights.items()
        )
        
        # Identify risk factors
        assessment['risk_factors'] = self._identify_feasibility_risks()
        
        # Generate recommendations
        assessment['recommended_adjustments'] = self._generate_feasibility_recommendations(assessment)
        
        # Calculate confidence based on data completeness
        assessment['confidence_score'] = self._calculate_assessment_confidence()
        
        logger.info(f"Feasibility assessment: {assessment['overall_feasibility']:.2f} "
                   f"(confidence: {assessment['confidence_score']:.2f})")
        
        return assessment
    
    def generate_success_metrics(self) -> List[Dict[str, Any]]:
        """
        Define measurable outcomes and KPIs based on user objectives
        Returns list of success metrics with measurement methods
        """
        metrics = []
        
        if not self.nodes():
            return metrics
        
        # Extract objectives and convert to measurable metrics
        for node_id in self.nodes():
            attrs = self.get_node_attributes(node_id)
            node_type = attrs.get('type', 'unknown')
            
            if node_type in ['primary_objective', 'secondary_objective', 'success_criteria']:
                metric = self._convert_objective_to_metric(node_id, attrs)
                if metric:
                    metrics.append(metric)
        
        # Add derived metrics based on relationships
        derived_metrics = self._generate_derived_metrics()
        metrics.extend(derived_metrics)
        
        # Sort by importance and measurability
        metrics.sort(key=lambda x: (x['importance_score'], x['measurability_score']), reverse=True)
        
        logger.info(f"Generated {len(metrics)} success metrics")
        return metrics
    
    def extract_decision_criteria(self) -> Dict[str, Any]:
        """
        Identify user priorities and trade-off preferences for decision making
        Returns structured decision criteria with weights and preferences
        """
        criteria = {
            'priority_framework': {},
            'trade_off_preferences': [],
            'decision_factors': {},
            'risk_tolerance': 'medium',
            'optimization_targets': [],
            'non_negotiables': [],
            'nice_to_haves': []
        }
        
        if not self.nodes():
            return criteria
        
        # Extract priority information
        priorities = self._extract_priority_information()
        criteria['priority_framework'] = priorities
        
        # Analyze trade-off preferences from relationships
        trade_offs = self._analyze_trade_off_preferences()
        criteria['trade_off_preferences'] = trade_offs
        
        # Extract decision factors
        factors = self._extract_decision_factors()
        criteria['decision_factors'] = factors
        
        # Determine risk tolerance
        criteria['risk_tolerance'] = self._determine_risk_tolerance()
        
        # Categorize requirements by flexibility
        categories = self._categorize_requirements_by_flexibility()
        criteria.update(categories)
        
        logger.info(f"Extracted decision criteria with {len(criteria['decision_factors'])} factors")
        return criteria
    
    def map_stakeholder_needs(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze different stakeholder requirements and potential conflicts
        Returns stakeholder mapping with needs and influence analysis
        """
        stakeholder_map = {}
        
        if not self.nodes():
            return stakeholder_map
        
        # Find stakeholder nodes
        for node_id in self.nodes():
            attrs = self.get_node_attributes(node_id)
            if attrs.get('type') == 'stakeholder':
                stakeholder_id = attrs.get('stakeholder_id', node_id)
                
                if stakeholder_id not in stakeholder_map:
                    stakeholder_map[stakeholder_id] = []
                
                stakeholder_info = {
                    'name': attrs.get('name', stakeholder_id),
                    'role': attrs.get('role', 'unknown'),
                    'influence_level': attrs.get('influence_level', 'medium'),
                    'support_level': attrs.get('support_level', 'neutral'),
                    'needs': self._extract_stakeholder_needs(node_id),
                    'concerns': attrs.get('concerns', []),
                    'success_criteria': attrs.get('success_criteria', []),
                    'communication_preferences': attrs.get('communication_preferences', {}),
                    'decision_authority': attrs.get('decision_authority', 'none')
                }
                
                stakeholder_map[stakeholder_id].append(stakeholder_info)
        
        logger.info(f"Mapped {len(stakeholder_map)} stakeholders")
        return stakeholder_map
    
    def capture_user_context(self) -> Dict[str, Any]:
        """
        Store background, assumptions, and sidebar details from user input
        Returns comprehensive context information
        """
        context = {
            'background_information': {},
            'assumptions': [],
            'constraints': [],
            'external_factors': [],
            'organizational_context': {},
            'technical_context': {},
            'business_context': {},
            'user_expertise_level': 'unknown',
            'previous_experience': [],
            'lessons_learned': [],
            'sidebar_details': {}
        }
        
        if not self.nodes():
            return context
        
        # Extract different types of context information
        for node_id in self.nodes():
            attrs = self.get_node_attributes(node_id)
            node_type = attrs.get('type', 'unknown')
            
            if node_type == 'background':
                context['background_information'][node_id] = {
                    'description': attrs.get('description', ''),
                    'relevance': attrs.get('relevance', 'medium'),
                    'source': attrs.get('source', 'user')
                }
            elif node_type == 'assumption':
                context['assumptions'].append({
                    'assumption': attrs.get('assumption', ''),
                    'confidence': attrs.get('confidence', 0.5),
                    'impact_if_wrong': attrs.get('impact_if_wrong', 'medium'),
                    'validation_method': attrs.get('validation_method', '')
                })
            elif node_type == 'external_factor':
                context['external_factors'].append({
                    'factor': attrs.get('factor', ''),
                    'impact': attrs.get('impact', 'unknown'),
                    'controllability': attrs.get('controllability', 'uncontrollable'),
                    'mitigation_strategy': attrs.get('mitigation_strategy', '')
                })
        
        # Extract user expertise and experience
        context['user_expertise_level'] = self._assess_user_expertise_level()
        context['previous_experience'] = self._extract_previous_experience()
        
        logger.info(f"Captured context: {len(context['assumptions'])} assumptions, "
                   f"{len(context['external_factors'])} external factors")
        
        return context
    
    def assess_gaps(self) -> List[Dict[str, Any]]:
        """
        Comprehensive gap analysis using enum-based classification
        Identifies what's missing to achieve user objectives
        """
        gaps = []
        
        if not self.nodes():
            return gaps
        
        # Analyze each objective for potential gaps
        objectives = [n for n in self.nodes() 
                     if self.get_node_attributes(n).get('type') in ['primary_objective', 'secondary_objective']]
        
        for obj_id in objectives:
            obj_attrs = self.get_node_attributes(obj_id)
            obj_gaps = self._analyze_objective_gaps(obj_id, obj_attrs)
            gaps.extend(obj_gaps)
        
        # Analyze constraints for gaps
        constraints = [n for n in self.nodes() 
                      if self.get_node_attributes(n).get('type') == 'constraint']
        
        for constraint_id in constraints:
            constraint_attrs = self.get_node_attributes(constraint_id)
            constraint_gaps = self._analyze_constraint_gaps(constraint_id, constraint_attrs)
            gaps.extend(constraint_gaps)
        
        # Analyze technical requirements for gaps
        tech_reqs = [n for n in self.nodes() 
                    if self.get_node_attributes(n).get('type') == 'technical_requirement']
        
        for tech_id in tech_reqs:
            tech_attrs = self.get_node_attributes(tech_id)
            tech_gaps = self._analyze_technical_gaps(tech_id, tech_attrs)
            gaps.extend(tech_gaps)
        
        # Analyze stakeholder gaps
        stakeholder_gaps = self._analyze_stakeholder_gaps()
        gaps.extend(stakeholder_gaps)
        
        # Sort by severity and impact
        gaps.sort(key=lambda x: (
            self._get_severity_score(x['severity']),
            x.get('impact_score', 0.5)
        ), reverse=True)
        
        logger.info(f"Identified {len(gaps)} gaps across {len(set(g['gap_type'] for g in gaps))} categories")
        return gaps
    
    def get_intent_completeness_score(self) -> float:
        """
        Calculate mathematical completeness score based on available data
        Returns 0.0-1.0 score indicating how complete the intent capture is
        """
        total_score = 0.0
        max_score = 0.0
        
        # Score based on objectives
        objectives = [n for n in self.nodes() 
                     if self.get_node_attributes(n).get('type') in ['primary_objective', 'secondary_objective']]
        if objectives:
            max_score += 0.3
            obj_completeness = sum(1 for obj in objectives 
                                 if self.get_node_attributes(obj).get('measurable', False)) / len(objectives)
            total_score += 0.3 * obj_completeness
        
        # Score based on constraints
        constraints = [n for n in self.nodes() 
                      if self.get_node_attributes(n).get('type') == 'constraint']
        if constraints:
            max_score += 0.2
            total_score += 0.2  # Having any constraints is good
        
        # Score based on stakeholders
        stakeholders = [n for n in self.nodes() 
                       if self.get_node_attributes(n).get('type') == 'stakeholder']
        if stakeholders:
            max_score += 0.2
            total_score += 0.2  # Having stakeholders mapped is good
        
        # Score based on context information
        if self.raw_user_prompt:
            max_score += 0.1
            total_score += 0.1
        
        if self.domain_context != "unknown":
            max_score += 0.1
            total_score += 0.1
        
        # Score based on ambiguity resolution
        if len(self.ambiguities) == 0:
            max_score += 0.1
            total_score += 0.1
        elif len(self.clarifying_questions) > 0:
            max_score += 0.1
            total_score += 0.05  # Partial credit for having questions ready
        
        return total_score / max_score if max_score > 0 else 0.0
    
    def calculate_confidence_distribution(self) -> Dict[str, float]:
        """
        Analyze distribution of confidence scores across intent elements
        Returns statistical summary of confidence levels
        """
        if not self.intent_confidence_scores:
            return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0, 'count': 0}
        
        scores = list(self.intent_confidence_scores.values())
        mean_confidence = sum(scores) / len(scores)
        min_confidence = min(scores)
        max_confidence = max(scores)
        
        # Calculate standard deviation
        variance = sum((x - mean_confidence) ** 2 for x in scores) / len(scores)
        std_confidence = variance ** 0.5
        
        return {
            'mean': mean_confidence,
            'min': min_confidence,
            'max': max_confidence,
            'std': std_confidence,
            'count': len(scores)
        }
    
    def get_ambiguity_severity_distribution(self) -> Dict[str, int]:
        """
        Analyze severity distribution of ambiguities
        Returns count by severity level
        """
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for ambiguity in self.ambiguities:
            severity = ambiguity.get('severity', 'medium')
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        return severity_counts
    
    def analyze_implicit_requirements_coverage(self) -> Dict[str, Any]:
        """
        Analyze how well implicit requirements are captured in explicit nodes
        Returns coverage analysis
        """
        analysis = {
            'total_implicit': len(self.implicit_requirements),
            'explicitly_captured': 0,
            'still_implicit': 0,
            'coverage_ratio': 0.0,
            'uncaptured_requirements': []
        }
        
        if not self.implicit_requirements:
            return analysis
        
        # Check which implicit requirements have corresponding explicit nodes
        explicit_descriptions = set()
        for node_id in self.nodes():
            attrs = self.get_node_attributes(node_id)
            if 'description' in attrs:
                explicit_descriptions.add(attrs['description'].lower())
        
        for impl_req in self.implicit_requirements:
            req_text = impl_req.get('requirement', '').lower()
            if any(req_text in desc for desc in explicit_descriptions):
                analysis['explicitly_captured'] += 1
            else:
                analysis['still_implicit'] += 1
                analysis['uncaptured_requirements'].append(impl_req)
        
        analysis['coverage_ratio'] = (analysis['explicitly_captured'] / 
                                    analysis['total_implicit'] if analysis['total_implicit'] > 0 else 0.0)
        
        return analysis
    
    # ========================================
    # Helper Methods
    # ========================================
    
    def _detect_constraint_conflict(self, constraint_a: Dict, constraint_b: Dict) -> Optional[Dict[str, Any]]:
        """Detect conflicts between two constraints"""
        # Simple conflict detection - can be enhanced with domain-specific logic
        if constraint_a['constraint_type'] == constraint_b['constraint_type']:
            if constraint_a['value'] != constraint_b['value']:
                return {
                    'type': 'value_conflict',
                    'severity': 'high',
                    'constraint_a': constraint_a,
                    'constraint_b': constraint_b,
                    'description': f"Conflicting values for {constraint_a['constraint_type']}",
                    'resolution_suggestions': [
                        'Clarify which constraint has higher priority',
                        'Find a compromise value',
                        'Make one constraint conditional'
                    ]
                }
        return None
    
    def _detect_impossible_constraint(self, constraint: Dict) -> Optional[Dict[str, Any]]:
        """Detect impossible or unrealistic constraints"""
        # Basic impossibility detection
        if constraint['constraint_type'] == 'budget' and constraint['value'] <= 0:
            return {
                'type': 'impossible_constraint',
                'severity': 'critical',
                'constraint': constraint,
                'description': 'Budget constraint cannot be zero or negative',
                'resolution_suggestions': ['Set a realistic budget amount']
            }
        return None
    
    def _get_conflict_severity_score(self, severity: str) -> int:
        """Convert severity to numeric score for sorting"""
        return {'critical': 3, 'high': 2, 'medium': 1, 'low': 0}.get(severity, 0)
    
    def _assess_technical_feasibility(self) -> float:
        """Assess technical feasibility of requirements"""
        # Simplified assessment - can be enhanced with domain expertise
        technical_nodes = [n for n in self.nodes() 
                          if self.get_node_attributes(n).get('type') == 'technical_requirement']
        
        if not technical_nodes:
            return 0.8  # Default moderate feasibility
        
        # Simple scoring based on complexity indicators
        complexity_scores = []
        for node_id in technical_nodes:
            attrs = self.get_node_attributes(node_id)
            complexity = attrs.get('complexity', 'medium')
            score = {'low': 0.9, 'medium': 0.7, 'high': 0.4, 'very_high': 0.2}.get(complexity, 0.7)
            complexity_scores.append(score)
        
        return sum(complexity_scores) / len(complexity_scores)
    
    def _assess_resource_feasibility(self) -> float:
        """Assess resource availability and constraints"""
        resource_nodes = [n for n in self.nodes() 
                         if self.get_node_attributes(n).get('type') == 'resource_constraint']
        
        if not resource_nodes:
            return 0.7  # Default moderate feasibility
        
        # Simple resource adequacy assessment
        adequacy_scores = []
        for node_id in resource_nodes:
            attrs = self.get_node_attributes(node_id)
            adequacy = attrs.get('adequacy', 'adequate')
            score = {'abundant': 1.0, 'adequate': 0.8, 'tight': 0.5, 'insufficient': 0.2}.get(adequacy, 0.7)
            adequacy_scores.append(score)
        
        return sum(adequacy_scores) / len(adequacy_scores)
    
    def _assess_timeline_feasibility(self) -> float:
        """Assess timeline constraints and deadlines"""
        timeline_nodes = [n for n in self.nodes() 
                         if self.get_node_attributes(n).get('type') == 'timeline_constraint']
        
        if not timeline_nodes:
            return 0.8  # Default moderate feasibility
        
        # Simple timeline pressure assessment
        pressure_scores = []
        for node_id in timeline_nodes:
            attrs = self.get_node_attributes(node_id)
            pressure = attrs.get('time_pressure', 'moderate')
            score = {'relaxed': 0.9, 'moderate': 0.7, 'tight': 0.4, 'unrealistic': 0.1}.get(pressure, 0.7)
            pressure_scores.append(score)
        
        return sum(pressure_scores) / len(pressure_scores)
    
    def _assess_stakeholder_feasibility(self) -> float:
        """Assess stakeholder alignment and support"""
        stakeholder_nodes = [n for n in self.nodes() 
                           if self.get_node_attributes(n).get('type') == 'stakeholder']
        
        if not stakeholder_nodes:
            return 0.8  # Default moderate feasibility
        
        # Simple stakeholder support assessment
        support_scores = []
        for node_id in stakeholder_nodes:
            attrs = self.get_node_attributes(node_id)
            support = attrs.get('support_level', 'neutral')
            score = {'strong': 1.0, 'moderate': 0.8, 'neutral': 0.6, 'weak': 0.3, 'opposed': 0.1}.get(support, 0.6)
            support_scores.append(score)
        
        return sum(support_scores) / len(support_scores)
    
    def _identify_feasibility_risks(self) -> List[Dict[str, Any]]:
        """Identify specific risks to feasibility"""
        risks = []
        
        # Check for high-risk patterns
        for node_id in self.nodes():
            attrs = self.get_node_attributes(node_id)
            
            if attrs.get('complexity') == 'very_high':
                risks.append({
                    'type': 'technical_complexity',
                    'description': f"Very high complexity requirement: {node_id}",
                    'impact': 'high',
                    'mitigation': 'Break down into smaller components'
                })
            
            if attrs.get('time_pressure') == 'unrealistic':
                risks.append({
                    'type': 'timeline_pressure',
                    'description': f"Unrealistic timeline constraint: {node_id}",
                    'impact': 'high',
                    'mitigation': 'Negotiate timeline extension or reduce scope'
                })
        
        return risks
    
    def _generate_feasibility_recommendations(self, assessment: Dict) -> List[str]:
        """Generate recommendations to improve feasibility"""
        recommendations = []
        
        if assessment['feasibility_factors']['technical'] < 0.6:
            recommendations.append("Consider simplifying technical requirements or breaking them into phases")
        
        if assessment['feasibility_factors']['resource'] < 0.6:
            recommendations.append("Secure additional resources or reduce scope to match available resources")
        
        if assessment['feasibility_factors']['timeline'] < 0.6:
            recommendations.append("Extend timeline or prioritize most critical features for initial delivery")
        
        if assessment['feasibility_factors']['stakeholder'] < 0.6:
            recommendations.append("Improve stakeholder alignment through better communication and expectation management")
        
        return recommendations
    
    def _calculate_assessment_confidence(self) -> float:
        """Calculate confidence in feasibility assessment based on data completeness"""
        total_nodes = len(self.nodes())
        if total_nodes == 0:
            return 0.0
        
        # Count nodes with sufficient detail
        detailed_nodes = 0
        for node_id in self.nodes():
            attrs = self.get_node_attributes(node_id)
            if len(attrs) >= 3:  # Has at least 3 attributes
                detailed_nodes += 1
        
        return min(detailed_nodes / total_nodes, 1.0)
    
    def _convert_objective_to_metric(self, node_id: str, attrs: Dict) -> Optional[Dict[str, Any]]:
        """Convert an objective node to a measurable metric"""
        description = attrs.get('description', '')
        if not description:
            return None
        
        return {
            'id': f"metric_{node_id}",
            'name': attrs.get('name', node_id),
            'description': description,
            'measurement_method': attrs.get('measurement_method', 'manual assessment'),
            'target_value': attrs.get('target_value'),
            'unit': attrs.get('unit', 'score'),
            'frequency': attrs.get('measurement_frequency', 'end of project'),
            'importance_score': self._calculate_importance_score(attrs),
            'measurability_score': self._calculate_measurability_score(attrs)
        }
    
    def _calculate_importance_score(self, attrs: Dict) -> float:
        """Calculate importance score for a metric"""
        priority = attrs.get('priority', 'medium')
        return {'critical': 1.0, 'high': 0.8, 'medium': 0.6, 'low': 0.4}.get(priority, 0.6)
    
    def _calculate_measurability_score(self, attrs: Dict) -> float:
        """Calculate how measurable a metric is"""
        if attrs.get('target_value') and attrs.get('measurement_method'):
            return 1.0
        elif attrs.get('target_value') or attrs.get('measurement_method'):
            return 0.7
        else:
            return 0.3
    
    def _generate_derived_metrics(self) -> List[Dict[str, Any]]:
        """Generate derived metrics based on graph relationships"""
        # Placeholder for relationship-based metric generation
        return []
    
    def _extract_priority_information(self) -> Dict[str, Any]:
        """Extract priority framework from intent graph"""
        framework = {
            'primary_priorities': [],
            'secondary_priorities': [],
            'priority_order': []
        }
        
        for node_id in self.nodes():
            attrs = self.get_node_attributes(node_id)
            priority = attrs.get('priority', 'medium')
            
            if priority in ['critical', 'high']:
                framework['primary_priorities'].append(node_id)
            else:
                framework['secondary_priorities'].append(node_id)
        
        return framework
    
    def _analyze_trade_off_preferences(self) -> List[Dict[str, Any]]:
        """Analyze trade-off preferences from graph relationships"""
        # Placeholder for trade-off analysis
        return []
    
    def _extract_decision_factors(self) -> Dict[str, float]:
        """Extract decision factors with weights"""
        factors = {}
        
        for node_id in self.nodes():
            attrs = self.get_node_attributes(node_id)
            if attrs.get('type') == 'decision_factor':
                factor_name = attrs.get('factor_name', node_id)
                weight = attrs.get('weight', 0.5)
                factors[factor_name] = weight
        
        return factors
    
    def _determine_risk_tolerance(self) -> str:
        """Determine overall risk tolerance from user preferences"""
        risk_nodes = [n for n in self.nodes() 
                     if self.get_node_attributes(n).get('type') == 'risk_preference']
        
        if not risk_nodes:
            return 'medium'
        
        # Simple majority vote on risk tolerance
        tolerances = [self.get_node_attributes(n).get('risk_tolerance', 'medium') 
                     for n in risk_nodes]
        
        return max(set(tolerances), key=tolerances.count)
    
    def _categorize_requirements_by_flexibility(self) -> Dict[str, List[str]]:
        """Categorize requirements by flexibility level"""
        categories = {
            'non_negotiables': [],
            'nice_to_haves': [],
            'optimization_targets': []
        }
        
        for node_id in self.nodes():
            attrs = self.get_node_attributes(node_id)
            flexibility = attrs.get('flexibility', 'flexible')
            
            if flexibility == 'rigid':
                categories['non_negotiables'].append(node_id)
            elif flexibility == 'optional':
                categories['nice_to_haves'].append(node_id)
            elif flexibility == 'optimizable':
                categories['optimization_targets'].append(node_id)
        
        return categories
    
    def _extract_stakeholder_needs(self, stakeholder_node_id: str) -> List[Dict[str, Any]]:
        """Extract needs for a specific stakeholder"""
        needs = []
        
        # Find connected need nodes
        for successor in self.successors(stakeholder_node_id):
            attrs = self.get_node_attributes(successor)
            if attrs.get('type') == 'stakeholder_need':
                needs.append({
                    'need': attrs.get('need', ''),
                    'importance': attrs.get('importance', 'medium'),
                    'rationale': attrs.get('rationale', '')
                })
        
        return needs
    
    def _assess_user_expertise_level(self) -> str:
        """Assess user expertise level from context clues"""
        expertise_indicators = []
        
        for node_id in self.nodes():
            attrs = self.get_node_attributes(node_id)
            if 'expertise_level' in attrs:
                expertise_indicators.append(attrs['expertise_level'])
        
        if not expertise_indicators:
            return 'unknown'
        
        # Return most common expertise level
        return max(set(expertise_indicators), key=expertise_indicators.count)
    
    def _extract_previous_experience(self) -> List[Dict[str, Any]]:
        """Extract previous experience information"""
        experiences = []
        
        for node_id in self.nodes():
            attrs = self.get_node_attributes(node_id)
            if attrs.get('type') == 'previous_experience':
                experiences.append({
                    'experience': attrs.get('experience', ''),
                    'outcome': attrs.get('outcome', 'unknown'),
                    'lessons_learned': attrs.get('lessons_learned', []),
                    'relevance': attrs.get('relevance', 'medium')
                })
        
        return experiences
    
    def _analyze_objective_gaps(self, obj_id: str, obj_attrs: Dict) -> List[Dict[str, Any]]:
        """Analyze gaps for a specific objective"""
        gaps = []
        
        # Check for missing measurement methods
        if obj_attrs.get('measurable', False) and not obj_attrs.get('measurement_method'):
            gaps.append({
                'gap_type': GapType.KNOWLEDGE.value,
                'severity': GapSeverity.MEDIUM.value,
                'objective_id': obj_id,
                'description': f"No measurement method defined for objective: {obj_attrs.get('name', obj_id)}",
                'impact_score': 0.6,
                'mitigation': "Define specific measurement criteria and methods",
                'hilt_required': True,
                'estimated_effort': "1-2 hours of planning"
            })
        
        # Check for missing target values
        if obj_attrs.get('measurable', False) and not obj_attrs.get('target_value'):
            gaps.append({
                'gap_type': GapType.KNOWLEDGE.value,
                'severity': GapSeverity.MEDIUM.value,
                'objective_id': obj_id,
                'description': f"No target value defined for objective: {obj_attrs.get('name', obj_id)}",
                'impact_score': 0.7,
                'mitigation': "Set specific, achievable target values",
                'hilt_required': True,
                'estimated_effort': "30 minutes of analysis"
            })
        
        # Check for high complexity objectives without breakdown
        if obj_attrs.get('complexity') == 'high' and obj_attrs.get('priority') == 'critical':
            gaps.append({
                'gap_type': GapType.PROCESS.value,
                'severity': GapSeverity.HIGH.value,
                'objective_id': obj_id,
                'description': f"High complexity critical objective may need decomposition: {obj_attrs.get('name', obj_id)}",
                'impact_score': 0.8,
                'mitigation': "Break down into smaller, manageable sub-objectives",
                'hilt_required': True,
                'estimated_effort': "2-4 hours of planning"
            })
        
        return gaps
    
    def _analyze_constraint_gaps(self, constraint_id: str, constraint_attrs: Dict) -> List[Dict[str, Any]]:
        """Analyze gaps for constraints"""
        gaps = []
        
        # Check for budget constraints without detailed breakdown
        if constraint_attrs.get('constraint_type') == 'budget':
            if not constraint_attrs.get('budget_breakdown'):
                gaps.append({
                    'gap_type': GapType.KNOWLEDGE.value,
                    'severity': GapSeverity.HIGH.value,
                    'constraint_id': constraint_id,
                    'description': "Budget constraint lacks detailed breakdown",
                    'impact_score': 0.8,
                    'mitigation': "Create detailed budget allocation across categories",
                    'hilt_required': True,
                    'estimated_effort': "1-2 hours of financial planning"
                })
        
        # Check for timeline constraints without milestone planning
        if constraint_attrs.get('constraint_type') == 'timeline':
            if constraint_attrs.get('time_pressure') in ['tight', 'unrealistic']:
                gaps.append({
                    'gap_type': GapType.RESOURCE.value,
                    'severity': GapSeverity.CRITICAL.value,
                    'constraint_id': constraint_id,
                    'description': "Timeline constraint may require additional resources",
                    'impact_score': 0.9,
                    'mitigation': "Assess resource requirements for timeline adherence",
                    'hilt_required': True,
                    'estimated_effort': "Half day of project planning"
                })
        
        return gaps
    
    def _analyze_technical_gaps(self, tech_id: str, tech_attrs: Dict) -> List[Dict[str, Any]]:
        """Analyze gaps for technical requirements"""
        gaps = []
        
        # Check for high complexity technical requirements
        if tech_attrs.get('complexity') in ['high', 'very_high']:
            gaps.append({
                'gap_type': GapType.SKILL.value,
                'severity': GapSeverity.HIGH.value,
                'technical_requirement_id': tech_id,
                'description': f"High complexity requirement may need specialized expertise: {tech_attrs.get('description', tech_id)}",
                'impact_score': 0.8,
                'mitigation': "Identify required skills and available expertise",
                'hilt_required': True,
                'estimated_effort': "Skills assessment and potential hiring/training"
            })
        
        # Check for security-related requirements
        if 'security' in tech_attrs.get('description', '').lower():
            gaps.append({
                'gap_type': GapType.HILT.value,
                'severity': GapSeverity.CRITICAL.value,
                'technical_requirement_id': tech_id,
                'description': "Security requirements need expert validation and ongoing monitoring",
                'impact_score': 0.95,
                'mitigation': "Engage security expert for validation and compliance",
                'hilt_required': True,
                'estimated_effort': "Security audit and ongoing monitoring setup"
            })
        
        # Check for integration requirements
        if 'integration' in tech_attrs.get('description', '').lower():
            gaps.append({
                'gap_type': GapType.EXTERNAL.value,
                'severity': GapSeverity.MEDIUM.value,
                'technical_requirement_id': tech_id,
                'description': "Integration requirements depend on external systems",
                'impact_score': 0.6,
                'mitigation': "Validate external system availability and compatibility",
                'hilt_required': False,
                'estimated_effort': "API documentation review and testing"
            })
        
        return gaps
    
    def _analyze_stakeholder_gaps(self) -> List[Dict[str, Any]]:
        """Analyze stakeholder-related gaps"""
        gaps = []
        
        stakeholders = [n for n in self.nodes() 
                       if self.get_node_attributes(n).get('type') == 'stakeholder']
        
        # Check for stakeholders with weak support
        weak_support_stakeholders = []
        for stakeholder_id in stakeholders:
            attrs = self.get_node_attributes(stakeholder_id)
            if attrs.get('support_level') in ['weak', 'opposed']:
                weak_support_stakeholders.append(stakeholder_id)
        
        if weak_support_stakeholders:
            gaps.append({
                'gap_type': GapType.STAKEHOLDER.value,
                'severity': GapSeverity.HIGH.value,
                'description': f"Stakeholders with weak support: {', '.join(weak_support_stakeholders)}",
                'impact_score': 0.8,
                'mitigation': "Develop stakeholder engagement and alignment strategy",
                'hilt_required': True,
                'estimated_effort': "Stakeholder meetings and relationship building"
            })
        
        # Check for high influence stakeholders without strong support
        high_influence_neutral = []
        for stakeholder_id in stakeholders:
            attrs = self.get_node_attributes(stakeholder_id)
            if (attrs.get('influence_level') == 'high' and 
                attrs.get('support_level') in ['neutral', 'unknown']):
                high_influence_neutral.append(stakeholder_id)
        
        if high_influence_neutral:
            gaps.append({
                'gap_type': GapType.STAKEHOLDER.value,
                'severity': GapSeverity.MEDIUM.value,
                'description': f"High influence stakeholders with neutral support: {', '.join(high_influence_neutral)}",
                'impact_score': 0.7,
                'mitigation': "Secure explicit support from high influence stakeholders",
                'hilt_required': True,
                'estimated_effort': "Stakeholder presentations and buy-in sessions"
            })
        
        return gaps
    
    def _get_severity_score(self, severity: str) -> int:
        """Convert severity to numeric score for sorting"""
        return {
            GapSeverity.CRITICAL.value: 4,
            GapSeverity.HIGH.value: 3,
            GapSeverity.MEDIUM.value: 2,
            GapSeverity.LOW.value: 1
        }.get(severity, 0)
    
    # ========================================
    # Intent Management Methods
    # ========================================
    
    def add_objective(self, objective_id: str, **attributes) -> None:
        """Add an objective to the intent graph"""
        intent_attrs = {
            'type': 'primary_objective',
            'priority': 'medium',
            'measurable': False,
            'created_date': datetime.now().isoformat(),
            **attributes
        }
        
        self.add_node(objective_id, **intent_attrs)
        logger.info(f"Added intent objective: {objective_id}")
    
    def add_constraint(self, constraint_id: str, **attributes) -> None:
        """Add a constraint to the intent graph"""
        constraint_attrs = {
            'type': 'constraint',
            'constraint_type': 'general',
            'flexibility': 'rigid',
            'created_date': datetime.now().isoformat(),
            **attributes
        }
        
        self.add_node(constraint_id, **constraint_attrs)
        logger.info(f"Added intent constraint: {constraint_id}")
    
    def add_stakeholder(self, stakeholder_id: str, **attributes) -> None:
        """Add a stakeholder to the intent graph"""
        stakeholder_attrs = {
            'type': 'stakeholder',
            'influence_level': 'medium',
            'support_level': 'neutral',
            'created_date': datetime.now().isoformat(),
            **attributes
        }
        
        self.add_node(stakeholder_id, **stakeholder_attrs)
        logger.info(f"Added stakeholder: {stakeholder_id}")
    
    def link_stakeholder_need(self, stakeholder_id: str, need_id: str, relationship: str = 'has_need') -> None:
        """Link a stakeholder to their needs"""
        self.add_edge(stakeholder_id, need_id, relationship=relationship)
        logger.info(f"Linked stakeholder {stakeholder_id} to need {need_id}")
    
    def link_constraint_to_objectives(self, constraint_id: str, objective_ids: List[str] = None, relationship: str = 'constrains') -> None:
        """
        Link a constraint to the objectives it affects
        If objective_ids is None, links to all objectives in the graph
        """
        if objective_ids is None:
            # Auto-link to all objectives
            objective_ids = [n for n in self.nodes() 
                           if self.get_node_attributes(n).get('type') in ['primary_objective', 'secondary_objective']]
        
        linked_count = 0
        for obj_id in objective_ids:
            if obj_id in self.nodes():
                self.add_edge(constraint_id, obj_id, relationship=relationship)
                linked_count += 1
        
        logger.info(f"Linked constraint {constraint_id} to {linked_count} objectives")
    
    def link_stakeholder_to_concerns(self, stakeholder_id: str, concern_ids: List[str] = None, relationship: str = 'concerned_about') -> None:
        """
        Link a stakeholder to their areas of concern (objectives, constraints, etc.)
        If concern_ids is None, links based on stakeholder role and influence
        """
        if concern_ids is None:
            # Auto-determine concerns based on stakeholder attributes
            stakeholder_attrs = self.get_node_attributes(stakeholder_id)
            role = stakeholder_attrs.get('role', '')
            influence = stakeholder_attrs.get('influence_level', 'medium')
            
            concern_ids = []
            
            # High influence stakeholders care about critical objectives
            if influence == 'high':
                critical_objectives = [n for n in self.nodes() 
                                     if (self.get_node_attributes(n).get('type') in ['primary_objective', 'secondary_objective'] 
                                         and self.get_node_attributes(n).get('priority') == 'critical')]
                concern_ids.extend(critical_objectives)
            
            # Decision makers care about constraints
            if 'decision' in role.lower():
                constraints = [n for n in self.nodes() 
                             if self.get_node_attributes(n).get('type') == 'constraint']
                concern_ids.extend(constraints)
            
            # Financial roles care about budget constraints
            if 'financial' in role.lower() or 'owner' in role.lower():
                budget_constraints = [n for n in self.nodes() 
                                    if (self.get_node_attributes(n).get('type') == 'constraint' 
                                        and self.get_node_attributes(n).get('constraint_type') == 'budget')]
                concern_ids.extend(budget_constraints)
        
        linked_count = 0
        for concern_id in concern_ids:
            if concern_id in self.nodes():
                self.add_edge(stakeholder_id, concern_id, relationship=relationship)
                linked_count += 1
        
        logger.info(f"Linked stakeholder {stakeholder_id} to {linked_count} concerns")
    
    def auto_link_graph_relationships(self) -> Dict[str, int]:
        """
        Automatically create logical relationships in the intent graph
        Returns statistics on relationships created
        """
        stats = {
            'constraint_links': 0,
            'stakeholder_links': 0,
            'objective_dependencies': 0
        }
        
        # Link all constraints to relevant objectives
        constraints = [n for n in self.nodes() 
                      if self.get_node_attributes(n).get('type') == 'constraint']
        
        for constraint_id in constraints:
            constraint_attrs = self.get_node_attributes(constraint_id)
            constraint_type = constraint_attrs.get('constraint_type', 'general')
            
            # Determine which objectives this constraint affects
            relevant_objectives = []
            
            if constraint_type == 'budget':
                # Budget constraints affect all objectives
                relevant_objectives = [n for n in self.nodes() 
                                     if self.get_node_attributes(n).get('type') in ['primary_objective', 'secondary_objective']]
            elif constraint_type == 'timeline':
                # Timeline constraints affect all objectives AND technical requirements
                relevant_objectives = [n for n in self.nodes() 
                                     if self.get_node_attributes(n).get('type') in ['primary_objective', 'secondary_objective', 'technical_requirement']]
            elif constraint_type in ['technical', 'security']:
                # Technical constraints affect technical objectives
                relevant_objectives = [n for n in self.nodes() 
                                     if (self.get_node_attributes(n).get('type') in ['primary_objective', 'secondary_objective']
                                         and 'technical' in self.get_node_attributes(n).get('description', '').lower())]
            else:
                # General constraints affect primary objectives
                relevant_objectives = [n for n in self.nodes() 
                                     if self.get_node_attributes(n).get('type') == 'primary_objective']
            
            if relevant_objectives:
                self.link_constraint_to_objectives(constraint_id, relevant_objectives)
                stats['constraint_links'] += len(relevant_objectives)
        
        # Link stakeholders to their concerns
        stakeholders = [n for n in self.nodes() 
                       if self.get_node_attributes(n).get('type') == 'stakeholder']
        
        for stakeholder_id in stakeholders:
            self.link_stakeholder_to_concerns(stakeholder_id)
            # Count new edges from this stakeholder
            stakeholder_edges = len([e for e in self.edges() if e[0] == stakeholder_id])
            stats['stakeholder_links'] += stakeholder_edges
        
        # Create objective dependencies based on priority and type
        primary_objectives = [n for n in self.nodes() 
                            if self.get_node_attributes(n).get('type') == 'primary_objective']
        secondary_objectives = [n for n in self.nodes() 
                              if self.get_node_attributes(n).get('type') == 'secondary_objective']
        
        # Secondary objectives often depend on primary objectives
        for secondary_id in secondary_objectives:
            for primary_id in primary_objectives:
                # Add dependency if it makes logical sense
                secondary_attrs = self.get_node_attributes(secondary_id)
                primary_attrs = self.get_node_attributes(primary_id)
                
                # Simple heuristic: if secondary mentions primary in description
                if (primary_attrs.get('name', '').lower() in secondary_attrs.get('description', '').lower() or
                    any(word in secondary_attrs.get('description', '').lower() 
                        for word in primary_attrs.get('name', '').lower().split())):
                    self.add_edge(primary_id, secondary_id, relationship='enables')
                    stats['objective_dependencies'] += 1
        
        logger.info(f"Auto-linked relationships: {stats}")
        return stats
    
    def link_technical_requirements_to_objectives(self) -> None:
        """Link technical requirements to objectives they enable"""
        tech_reqs = [n for n in self.nodes() 
                    if self.get_node_attributes(n).get('type') == 'technical_requirement']
        objectives = [n for n in self.nodes() 
                     if self.get_node_attributes(n).get('type') in ['primary_objective', 'secondary_objective']]
        
        linked_count = 0
        for tech_id in tech_reqs:
            tech_attrs = self.get_node_attributes(tech_id)
            tech_desc = tech_attrs.get('description', '').lower()
            tech_name = tech_attrs.get('name', '').lower()
            
            for obj_id in objectives:
                obj_attrs = self.get_node_attributes(obj_id)
                obj_desc = obj_attrs.get('description', '').lower()
                obj_name = obj_attrs.get('name', '').lower()
                
                # Link if technical requirement mentions objective or vice versa
                if (any(word in tech_desc for word in obj_name.split()) or
                    any(word in obj_desc for word in tech_name.split()) or
                    tech_attrs.get('importance') == 'critical'):
                    self.add_edge(tech_id, obj_id, relationship='enables')
                    linked_count += 1
        
        logger.info(f"Linked {linked_count} technical requirements to objectives")
    
    def link_assumptions_to_affected_elements(self) -> None:
        """Link assumptions to constraints/objectives they impact"""
        assumptions = [n for n in self.nodes() 
                      if self.get_node_attributes(n).get('type') == 'assumption']
        
        linked_count = 0
        for assumption_id in assumptions:
            assumption_attrs = self.get_node_attributes(assumption_id)
            assumption_text = assumption_attrs.get('assumption', '').lower()
            
            # Link to related constraints
            constraints = [n for n in self.nodes() 
                          if self.get_node_attributes(n).get('type') == 'constraint']
            
            for constraint_id in constraints:
                constraint_attrs = self.get_node_attributes(constraint_id)
                constraint_text = constraint_attrs.get('constraint', '').lower()
                constraint_type = constraint_attrs.get('constraint_type', '')
                
                # Link if assumption mentions constraint type or content
                if (constraint_type in assumption_text or
                    any(word in assumption_text for word in constraint_text.split()[:3])):
                    self.add_edge(assumption_id, constraint_id, relationship='affects')
                    linked_count += 1
            
            # Link to related objectives
            objectives = [n for n in self.nodes() 
                         if self.get_node_attributes(n).get('type') in ['primary_objective', 'secondary_objective']]
            
            for obj_id in objectives:
                obj_attrs = self.get_node_attributes(obj_id)
                obj_desc = obj_attrs.get('description', '').lower()
                
                if any(word in assumption_text for word in obj_desc.split()[:5]):
                    self.add_edge(assumption_id, obj_id, relationship='impacts')
                    linked_count += 1
        
        logger.info(f"Linked {linked_count} assumptions to affected elements")
    
    def link_success_criteria_to_objectives(self) -> None:
        """Link success criteria to the objectives they measure"""
        success_criteria = [n for n in self.nodes() 
                           if self.get_node_attributes(n).get('type') == 'success_criteria']
        objectives = [n for n in self.nodes() 
                     if self.get_node_attributes(n).get('type') in ['primary_objective', 'secondary_objective']]
        
        linked_count = 0
        for criteria_id in success_criteria:
            criteria_attrs = self.get_node_attributes(criteria_id)
            criteria_text = criteria_attrs.get('criteria', '').lower()
            
            for obj_id in objectives:
                obj_attrs = self.get_node_attributes(obj_id)
                obj_name = obj_attrs.get('name', '').lower()
                obj_desc = obj_attrs.get('description', '').lower()
                
                # Link if criteria mentions objective
                if (any(word in criteria_text for word in obj_name.split()) or
                    any(word in obj_desc for word in criteria_text.split()[:5])):
                    self.add_edge(criteria_id, obj_id, relationship='measures')
                    linked_count += 1
        
        logger.info(f"Linked {linked_count} success criteria to objectives")
    
    def link_background_to_relevant_elements(self) -> None:
        """Link background/context to relevant objectives/constraints"""
        background_nodes = [n for n in self.nodes() 
                           if self.get_node_attributes(n).get('type') in ['background', 'previous_experience', 'external_factor']]
        
        linked_count = 0
        for bg_id in background_nodes:
            bg_attrs = self.get_node_attributes(bg_id)
            bg_text = (bg_attrs.get('description', '') + ' ' + 
                      bg_attrs.get('experience', '') + ' ' +
                      bg_attrs.get('factor', '')).lower()
            
            relevance = bg_attrs.get('relevance', 'medium')
            
            # Only link high relevance background items
            if relevance in ['high', 'critical']:
                # Link to constraints
                constraints = [n for n in self.nodes() 
                              if self.get_node_attributes(n).get('type') == 'constraint']
                
                for constraint_id in constraints:
                    constraint_attrs = self.get_node_attributes(constraint_id)
                    constraint_text = constraint_attrs.get('constraint', '').lower()
                    
                    if any(word in bg_text for word in constraint_text.split()[:3]):
                        self.add_edge(bg_id, constraint_id, relationship='informs')
                        linked_count += 1
                
                # Link to objectives
                objectives = [n for n in self.nodes() 
                             if self.get_node_attributes(n).get('type') in ['primary_objective', 'secondary_objective']]
                
                for obj_id in objectives:
                    obj_attrs = self.get_node_attributes(obj_id)
                    obj_desc = obj_attrs.get('description', '').lower()
                    
                    if any(word in bg_text for word in obj_desc.split()[:5]):
                        self.add_edge(bg_id, obj_id, relationship='contextualizes')
                        linked_count += 1
        
        logger.info(f"Linked {linked_count} background elements to relevant items")
    
    def link_stakeholder_needs_to_objectives(self) -> None:
        """Link stakeholder needs to objectives they influence"""
        stakeholder_needs = [n for n in self.nodes() 
                            if self.get_node_attributes(n).get('type') == 'stakeholder_need']
        objectives = [n for n in self.nodes() 
                     if self.get_node_attributes(n).get('type') in ['primary_objective', 'secondary_objective']]
        
        linked_count = 0
        for need_id in stakeholder_needs:
            need_attrs = self.get_node_attributes(need_id)
            need_text = need_attrs.get('need', '').lower()
            importance = need_attrs.get('importance', 'medium')
            
            for obj_id in objectives:
                obj_attrs = self.get_node_attributes(obj_id)
                obj_desc = obj_attrs.get('description', '').lower()
                obj_name = obj_attrs.get('name', '').lower()
                
                # Link if need mentions objective or is high importance
                if (any(word in need_text for word in obj_name.split()) or
                    any(word in obj_desc for word in need_text.split()[:5]) or
                    importance == 'high'):
                    self.add_edge(need_id, obj_id, relationship='influences')
                    linked_count += 1
        
        logger.info(f"Linked {linked_count} stakeholder needs to objectives")
    
    def link_timeline_constraints_to_technical_requirements(self) -> None:
        """Link timeline constraints to technical requirements they affect"""
        timeline_constraints = [n for n in self.nodes() 
                               if (self.get_node_attributes(n).get('type') == 'timeline_constraint' or
                                   (self.get_node_attributes(n).get('type') == 'constraint' and 
                                    self.get_node_attributes(n).get('constraint_type') == 'timeline'))]
        
        tech_requirements = [n for n in self.nodes() 
                            if self.get_node_attributes(n).get('type') == 'technical_requirement']
        
        linked_count = 0
        for timeline_id in timeline_constraints:
            timeline_attrs = self.get_node_attributes(timeline_id)
            
            for tech_id in tech_requirements:
                tech_attrs = self.get_node_attributes(tech_id)
                
                # Link timeline constraints to all technical requirements
                # (timeline affects when technical work can be completed)
                self.add_edge(timeline_id, tech_id, relationship='constrains')
                linked_count += 1
        
        logger.info(f"Linked {linked_count} timeline constraints to technical requirements")
    
    def comprehensive_auto_link(self) -> Dict[str, int]:
        """
        Perform comprehensive auto-linking of all node types
        Returns statistics on all relationships created
        """
        print("Performing comprehensive auto-linking...")
        
        # Get initial edge count
        initial_edges = len(self.edges())
        
        # Run all linking functions
        self.auto_link_graph_relationships()
        self.link_technical_requirements_to_objectives()
        self.link_assumptions_to_affected_elements()
        self.link_success_criteria_to_objectives()
        self.link_background_to_relevant_elements()
        self.link_stakeholder_needs_to_objectives()
        self.link_timeline_constraints_to_technical_requirements()
        
        # Calculate final statistics
        final_edges = len(self.edges())
        total_links_added = final_edges - initial_edges
        
        # Get edge type distribution
        edge_types = {}
        for source, target, attrs in self.graph.edges(data=True):
            rel_type = attrs.get('relationship', 'unknown')
            edge_types[rel_type] = edge_types.get(rel_type, 0) + 1
        
        stats = {
            'initial_edges': initial_edges,
            'final_edges': final_edges,
            'total_links_added': total_links_added,
            'edge_type_distribution': edge_types
        }
        
        logger.info(f"Comprehensive linking complete: {stats}")
        return stats
    
    # ========================================
    # Intent Derivation Data Management
    # ========================================
    
    def set_raw_user_prompt(self, prompt: str) -> None:
        """Store the original user prompt (set by agent brain)"""
        self.raw_user_prompt = prompt
        logger.info(f"Raw user prompt stored ({len(prompt)} characters)")
    
    def set_domain_context(self, domain: str) -> None:
        """Set the identified domain context (set by agent brain)"""
        self.domain_context = domain
        logger.info(f"Domain context set to: {domain}")
    
    def add_confidence_score(self, element_type: str, confidence: float) -> None:
        """Add confidence score for an intent element (set by agent brain)"""
        self.intent_confidence_scores[element_type] = max(0.0, min(1.0, confidence))
        logger.info(f"Confidence score for {element_type}: {confidence:.2f}")
    
    def add_ambiguity(self, ambiguity_data: Dict[str, Any]) -> None:
        """Add an identified ambiguity (set by agent brain)"""
        required_fields = ['element', 'description', 'severity']
        if all(field in ambiguity_data for field in required_fields):
            self.ambiguities.append(ambiguity_data)
            logger.info(f"Added ambiguity: {ambiguity_data['element']} ({ambiguity_data['severity']})")
        else:
            logger.warning(f"Ambiguity data missing required fields: {required_fields}")
    
    def add_clarifying_question(self, question: str, related_element: str = None) -> None:
        """Add a clarifying question (set by agent brain)"""
        question_data = {
            'question': question,
            'related_element': related_element,
            'created_date': datetime.now().isoformat()
        }
        self.clarifying_questions.append(question_data)
        logger.info(f"Added clarifying question for {related_element}: {question}")
    
    def add_implicit_requirement(self, requirement_data: Dict[str, Any]) -> None:
        """Add an implicit requirement (set by agent brain)"""
        required_fields = ['requirement', 'confidence', 'rationale']
        if all(field in requirement_data for field in required_fields):
            self.implicit_requirements.append(requirement_data)
            logger.info(f"Added implicit requirement: {requirement_data['requirement']}")
        else:
            logger.warning(f"Implicit requirement data missing required fields: {required_fields}")
    
    def set_user_sentiment(self, sentiment: str) -> None:
        """Set overall user sentiment (set by agent brain)"""
        valid_sentiments = ['positive', 'neutral', 'negative', 'urgent', 'frustrated', 'excited']
        if sentiment in valid_sentiments:
            self.user_sentiment = sentiment
            logger.info(f"User sentiment set to: {sentiment}")
        else:
            logger.warning(f"Invalid sentiment: {sentiment}. Valid options: {valid_sentiments}")
    
    def set_processing_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set LLM processing metadata (set by agent brain)"""
        self.prompt_processing_metadata = metadata
        logger.info(f"Processing metadata updated with {len(metadata)} fields")
    
    def clear_ambiguities(self) -> None:
        """Clear resolved ambiguities"""
        cleared_count = len(self.ambiguities)
        self.ambiguities = []
        logger.info(f"Cleared {cleared_count} ambiguities")
    
    def remove_clarifying_question(self, question_index: int) -> None:
        """Remove a clarifying question (after it's been answered)"""
        if 0 <= question_index < len(self.clarifying_questions):
            removed = self.clarifying_questions.pop(question_index)
            logger.info(f"Removed clarifying question: {removed['question']}")
        else:
            logger.warning(f"Invalid question index: {question_index}")
    
    def get_intent_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of intent data structures"""
        return {
            'raw_prompt_length': len(self.raw_user_prompt),
            'domain_context': self.domain_context,
            'confidence_scores': self.intent_confidence_scores,
            'ambiguity_count': len(self.ambiguities),
            'clarifying_questions_count': len(self.clarifying_questions),
            'implicit_requirements_count': len(self.implicit_requirements),
            'user_sentiment': self.user_sentiment,
            'completeness_score': self.get_intent_completeness_score(),
            'confidence_distribution': self.calculate_confidence_distribution(),
            'ambiguity_severity_distribution': self.get_ambiguity_severity_distribution(),
            'implicit_coverage': self.analyze_implicit_requirements_coverage()
        }
    
    # ========================================
    # LLM Query Generation & Schema Methods
    # ========================================
    
    @classmethod
    def get_schema_template(cls) -> Dict[str, Any]:
        """
        Get the complete JSON schema template for intent graphs
        This tells LLMs exactly what structure to generate
        """
        return IntentGraphSchema.schema()
    
    @classmethod
    def get_llm_query_prompt(cls, user_input: str, domain_context: str = "unknown") -> str:
        """
        Generate a comprehensive LLM prompt for creating intent graph JSON
        
        Args:
            user_input: The user's original request/prompt
            domain_context: Optional domain context (e.g., "mobile_app", "wedding_planning")
            
        Returns:
            Structured prompt for LLM to generate proper intent graph JSON
        """
        
        prompt = f"""
You are an expert intent analysis system. Analyze the following user request and generate a comprehensive Intent Knowledge Graph in JSON format.

USER REQUEST: "{user_input}"
DOMAIN CONTEXT: {domain_context}

Generate a JSON structure that captures the user's complete intent with the following components:

## REQUIRED JSON STRUCTURE:

```json
{{
  "graph_metadata": {{
    "graph_id": "unique_identifier",
    "graph_type": "intent",
    "intent_name": "descriptive_name",
    "created_date": "ISO_timestamp"
  }},
  "intent_data": {{
    "raw_user_prompt": "{user_input}",
    "domain_context": "{domain_context}",
    "user_sentiment": "positive|neutral|negative|urgent|frustrated|excited",
    "confidence_scores": {{"objectives": 0.9, "constraints": 0.8}},
    "ambiguities": [
      {{"element": "element_name", "description": "what's unclear", "severity": "high|medium|low"}}
    ],
    "clarifying_questions": [
      {{"question": "specific question", "related_element": "element_id"}}
    ],
    "implicit_requirements": [
      {{"requirement": "inferred need", "confidence": 0.8, "rationale": "why inferred"}}
    ]
  }},
  "objectives": [
    {{
      "id": "unique_id",
      "name": "objective_name",
      "description": "detailed_description",
      "priority": "critical|high|medium|low",
      "measurable": true|false,
      "measurement_method": "how_to_measure",
      "target_value": numeric_target,
      "unit": "measurement_unit",
      "complexity": "low|medium|high|very_high"
    }}
  ],
  "constraints": [
    {{
      "id": "unique_id",
      "constraint_type": "budget|timeline|technical|resource|regulatory|accessibility",
      "constraint": "description",
      "value": numeric_value,
      "flexibility": "rigid|somewhat_flexible|flexible",
      "time_pressure": "low|moderate|tight|unrealistic",
      "budget_breakdown": {{"category1": amount1, "category2": amount2}}
    }}
  ],
  "stakeholders": [
    {{
      "id": "unique_id",
      "name": "stakeholder_name",
      "role": "their_role",
      "influence_level": "high|medium|low",
      "support_level": "strong|cautious|neutral|weak|opposed|unknown",
      "decision_authority": "final|approval|input|none"
    }}
  ],
  "technical_requirements": [
    {{
      "id": "unique_id",
      "name": "requirement_name",
      "description": "detailed_description",
      "complexity": "low|medium|high|very_high",
      "importance": "critical|high|medium|low"
    }}
  ],
  "assumptions": [
    {{
      "id": "unique_id",
      "assumption": "what_we_assume",
      "confidence": 0.0_to_1.0,
      "impact_if_wrong": "critical|high|medium|low",
      "validation_method": "how_to_verify"
    }}
  ],
  "relationships": [
    {{
      "source": "source_node_id",
      "target": "target_node_id",
      "relationship": "constrains|enables|influences|depends_on"
    }}
  ]
}}
```

## ANALYSIS GUIDELINES:

1. **OBJECTIVES**: Extract all goals (primary and secondary). Make them SMART when possible.
2. **CONSTRAINTS**: Identify budget, timeline, technical, and resource limitations.
3. **STAKEHOLDERS**: Include decision makers, influencers, and affected parties.
4. **TECHNICAL REQUIREMENTS**: Any technical needs, integrations, or specifications.
5. **ASSUMPTIONS**: What the user assumes to be true but might not be.
6. **RELATIONSHIPS**: How elements connect (constraints limit objectives, stakeholders influence decisions).

## INTENT ANALYSIS:
- Analyze user sentiment and urgency level
- Identify ambiguous elements that need clarification
- Generate specific clarifying questions
- Infer implicit requirements not explicitly stated
- Assess confidence in your analysis

Generate ONLY the JSON structure. Be comprehensive but realistic. Include confidence scores for your analysis quality.
"""
        
        return prompt.strip()
    
    @classmethod
    def get_llm_refinement_prompt(cls, current_json: Dict[str, Any], user_feedback: str) -> str:
        """
        Generate a prompt for LLM to refine existing intent graph based on user feedback
        
        Args:
            current_json: Current intent graph JSON
            user_feedback: User's clarifications or corrections
            
        Returns:
            Prompt for LLM to update the intent graph
        """
        
        prompt = f"""
You are refining an existing Intent Knowledge Graph based on user feedback.

CURRENT INTENT GRAPH:
```json
{json.dumps(current_json, indent=2)}
```

USER FEEDBACK: "{user_feedback}"

Update the intent graph JSON to incorporate the user's feedback. Follow these guidelines:

1. **PRESERVE** existing structure and IDs where possible
2. **UPDATE** elements that the user corrected or clarified
3. **ADD** new elements if the user provided additional information
4. **REMOVE** elements if the user indicated they're incorrect
5. **ADJUST** confidence scores based on the clarification
6. **RESOLVE** ambiguities that the user addressed
7. **UPDATE** relationships if new connections are implied

Return the complete updated JSON structure with the same format as the original.
Focus on the specific feedback provided while maintaining the overall intent structure.
"""
        
        return prompt.strip()
    
    def validate_and_import_json(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate JSON data against intent graph schema and import if valid
        
        Args:
            json_data: JSON data to validate and import
            
        Returns:
            Validation results and import statistics
        """
        validation_result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'import_stats': None
        }
        
        try:
            # Validate against Pydantic schema
            validated_data = IntentGraphSchema.model_validate(json_data)
            validation_result['valid'] = True
            
            # Import the validated data
            import_stats = self._import_validated_intent_data(validated_data)
            validation_result['import_stats'] = import_stats
            
            logger.info(f"Successfully validated and imported intent graph JSON")
            
        except Exception as e:
            validation_result['errors'].append(str(e))
            logger.error(f"Intent graph JSON validation failed: {str(e)}")
        
        return validation_result
    
    def _import_validated_intent_data(self, validated_data: IntentGraphSchema) -> Dict[str, int]:
        """Import validated Pydantic data into the intent graph"""
        stats = {
            'objectives_added': 0,
            'constraints_added': 0,
            'stakeholders_added': 0,
            'technical_requirements_added': 0,
            'assumptions_added': 0,
            'relationships_added': 0
        }
        
        # Import intent derivation data
        intent_data = validated_data.intent_data
        self.set_raw_user_prompt(intent_data.raw_user_prompt)
        self.set_domain_context(intent_data.domain_context)
        self.set_user_sentiment(intent_data.user_sentiment)
        
        for element_type, confidence in intent_data.confidence_scores.items():
            self.add_confidence_score(element_type, confidence)
        
        for ambiguity in intent_data.ambiguities:
            self.add_ambiguity(ambiguity)
        
        for question in intent_data.clarifying_questions:
            self.add_clarifying_question(question['question'], question.get('related_element'))
        
        for req in intent_data.implicit_requirements:
            self.add_implicit_requirement(req)
        
        # Import objectives
        for obj in validated_data.objectives:
            self.add_objective(
                obj.id,
                name=obj.name,
                description=obj.description,
                priority=obj.priority,
                measurable=obj.measurable,
                measurement_method=obj.measurement_method,
                target_value=obj.target_value,
                unit=obj.unit,
                complexity=obj.complexity
            )
            stats['objectives_added'] += 1
        
        # Import constraints
        for constraint in validated_data.constraints:
            self.add_constraint(
                constraint.id,
                constraint_type=constraint.constraint_type,
                constraint=constraint.constraint,
                value=constraint.value,
                flexibility=constraint.flexibility,
                time_pressure=constraint.time_pressure,
                budget_breakdown=constraint.budget_breakdown
            )
            stats['constraints_added'] += 1
        
        # Import stakeholders
        for stakeholder in validated_data.stakeholders:
            self.add_stakeholder(
                stakeholder.id,
                name=stakeholder.name,
                role=stakeholder.role,
                influence_level=stakeholder.influence_level,
                support_level=stakeholder.support_level,
                decision_authority=stakeholder.decision_authority,
                communication_preferences=stakeholder.communication_preferences
            )
            stats['stakeholders_added'] += 1
        
        # Import technical requirements
        for tech_req in validated_data.technical_requirements:
            self.add_node(
                tech_req.id,
                type="technical_requirement",
                name=tech_req.name,
                description=tech_req.description,
                complexity=tech_req.complexity,
                importance=tech_req.importance
            )
            stats['technical_requirements_added'] += 1
        
        # Import assumptions
        for assumption in validated_data.assumptions:
            self.add_node(
                assumption.id,
                type="assumption",
                assumption=assumption.assumption,
                confidence=assumption.confidence,
                impact_if_wrong=assumption.impact_if_wrong,
                validation_method=assumption.validation_method
            )
            stats['assumptions_added'] += 1
        
        # Import relationships
        for rel in validated_data.relationships:
            self.add_edge(rel['source'], rel['target'], relationship=rel['relationship'])
            stats['relationships_added'] += 1
        
        return stats
