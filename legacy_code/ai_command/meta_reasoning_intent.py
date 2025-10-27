#!/usr/bin/env python3
"""
Intent Analysis Engine - Sophisticated Intent-Specific Meta-Reasoning
Subclass of MetaReasoningBase with AdaptiveIntentRunner's advanced capabilities

Features:
- Intent-specific Pydantic models with strict validation
- Sophisticated heuristic linking from AdaptiveIntentRunner
- AI actionability constraints and insights
- Advanced systems thinking and graph analysis
"""

import json
import time
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, ValidationError

# Import base class and core components
from meta_reasoning_base import MetaReasoningBase, ReasoningContext
from knowledge_graph import KnowledgeGraph, GraphType

# Import intent-specific models from intent_knowledge_graph
from intent_knowledge_graph import (
    IntentConstraint, 
    IntentTechnicalRequirement, 
    IntentAssumption,
    IntentKnowledgeGraph
)

# ============================================================================
# INTENT-SPECIFIC PYDANTIC MODELS (from AdaptiveIntentRunner)
# ============================================================================

class AIActionableInsight(BaseModel):
    """AI Agent Actionability Constraints"""
    insight: str = Field(description="The insight or recommendation")
    actionable_by_ai: bool = Field(description="Can this be executed by AI agent with tools?")
    required_tools: List[str] = Field(description="Tools/APIs needed for AI execution")
    automation_level: str = Field(pattern="^(fully_automated|human_in_loop|human_required)$")

# Enhanced response models with AI constraints
class AIConstraintsResponse(BaseModel):
    constraints: List[IntentConstraint]
    ai_actionable_insights: List[AIActionableInsight] = Field(
        description="AI-actionable recommendations for constraint management"
    )

class AITechnicalRequirementsResponse(BaseModel):
    technical_requirements: List[IntentTechnicalRequirement]
    ai_actionable_insights: List[AIActionableInsight] = Field(
        description="AI-actionable recommendations for technical implementation"
    )

class AIAssumptionsResponse(BaseModel):
    assumptions: List[IntentAssumption]
    ai_actionable_insights: List[AIActionableInsight] = Field(
        description="AI-actionable recommendations for assumption validation"
    )

# Fallback models for compatibility
class ConstraintsResponse(BaseModel):
    constraints: List[IntentConstraint]

class TechnicalRequirementsResponse(BaseModel):
    technical_requirements: List[IntentTechnicalRequirement]

class AssumptionsResponse(BaseModel):
    assumptions: List[IntentAssumption]

# Comprehensive stakeholder-objective expansion model
class StakeholderObjectiveExpansion(BaseModel):
    """Complete stakeholder-objective expansion focusing on what LLMs can reliably identify"""
    constraints: List[IntentConstraint] = Field(default_factory=list)
    technical_requirements: List[IntentTechnicalRequirement] = Field(default_factory=list)
    assumptions: List[IntentAssumption] = Field(default_factory=list)
    dependencies: List[Dict[str, Any]] = Field(default_factory=list, description="Dependencies for this objective")
    # REMOVED: success_metrics - LLMs struggle with realistic targets and timeframes

# Additional node-type specific models
class AISuccessMetricsResponse(BaseModel):
    """Success metrics with AI actionability insights"""
    success_metrics: List[Dict[str, Any]] = Field(default_factory=list, description="Success metrics and KPIs")
    ai_actionable_insights: List[AIActionableInsight] = Field(default_factory=list)

class AIStakeholdersResponse(BaseModel):
    """Stakeholder analysis with AI actionability insights"""
    stakeholders: List[Dict[str, Any]] = Field(default_factory=list, description="Stakeholder analysis")
    ai_actionable_insights: List[AIActionableInsight] = Field(default_factory=list)

class AIDependenciesResponse(BaseModel):
    """Dependencies with AI actionability insights"""
    dependencies: List[Dict[str, Any]] = Field(default_factory=list, description="Project dependencies")
    ai_actionable_insights: List[AIActionableInsight] = Field(default_factory=list)

# ============================================================================
# INTENT ANALYSIS ENGINE - SOPHISTICATED SUBCLASS
# ============================================================================

class IntentAnalysisEngine(MetaReasoningBase):
    """
    Intent-specific meta-reasoning engine with sophisticated heuristic analysis
    
    Incorporates all advanced capabilities from AdaptiveIntentRunner:
    - Sophisticated heuristic linking and systems thinking
    - Intent-specific Pydantic models with strict validation
    - AI actionability constraints and insights
    - Advanced graph analysis and domain intelligence
    """
    
    def __init__(self, use_real_ai: bool = True, base_port: int = 5900):
        super().__init__(use_real_ai, base_port)
        
        # Map content types to Pydantic models for structured output
        self.pydantic_models = {
            "stakeholder_objective_expansion": StakeholderObjectiveExpansion,
            "node_type_constraints": AIConstraintsResponse,
            "node_type_technical_requirements": AITechnicalRequirementsResponse,
            "node_type_assumptions": AIAssumptionsResponse,
            "node_type_success_metrics": AISuccessMetricsResponse,
            "node_type_stakeholders": AIStakeholdersResponse,
            "node_type_dependencies": AIDependenciesResponse
        }
    
    def _validate_with_pydantic(self, parsed_json: Dict, content_type: str) -> Dict:
        """DEPRECATED: Structured output eliminates need for post-hoc validation"""
        print(f"⚠️ DEPRECATED: Post-hoc validation called for {content_type}")
        print(f"   This should not happen with structured output - investigate the call path")
        return parsed_json
    
    # REMOVED: All fallback validation methods - structured output eliminates the need
        
    async def analyze_complexity(self, user_input: str) -> Dict[str, Any]:
        """Pass-through to base class complexity analysis"""
        return await super().analyze_complexity(user_input)
        
    # ============================================================================
    # INTENT-SPECIFIC NODE TYPE HANDLING (from AdaptiveIntentRunner)
    # ============================================================================
    
    def get_node_type_specs(self) -> Dict[str, Dict[str, str]]:
        """Intent-specific node type specifications with heuristic tags"""
        return {
            "constraints": {
                "focus": "limitations, restrictions, and boundaries that affect the project",
                "examples": "budget limits, timeline restrictions, regulatory requirements, resource constraints",
                "heuristic_tags": "cascade_potential, rigid_constraint, bottleneck_risk"
            },
            "technical_requirements": {
                "focus": "technical capabilities, systems, and infrastructure needed",
                "examples": "software systems, hardware requirements, integrations, APIs, databases",
                "heuristic_tags": "bottleneck_potential, dependency_critical, temporal_sequence"
            },
            "assumptions": {
                "focus": "beliefs, expectations, and suppositions being made",
                "examples": "user behavior assumptions, market conditions, technology availability",
                "heuristic_tags": "high_risk, leverage_point, confidence_level"
            },
            "dependencies": {
                "focus": "prerequisites, sequences, and interdependencies between components",
                "examples": "approval processes, completion sequences, information dependencies",
                "heuristic_tags": "temporal_sequence, causal_chain, blocking_dependency"
            },
            "success_metrics": {
                "focus": "measurements, KPIs, and indicators of success",
                "examples": "performance metrics, user satisfaction, ROI, completion criteria",
                "heuristic_tags": "feedback_loop_trigger, reinforcement_potential"
            }
        }
    
    def _create_node_type_prompt(self, user_input: str, assessment: Dict, node_type: str, heuristics: list) -> str:
        """Create specialized prompt for specific node type expansion - EXACT from AdaptiveIntentRunner"""
        
        node_type_specs = {
            "constraints": {
                "focus": "limitations, restrictions, and boundaries that affect the project",
                "examples": "budget limits, timeline restrictions, regulatory requirements, resource constraints",
                "heuristic_tags": "cascade_potential, rigid_constraint, bottleneck_risk"
            },
            "technical_requirements": {
                "focus": "technical capabilities, systems, and infrastructure needed",
                "examples": "software systems, hardware requirements, integrations, APIs, databases",
                "heuristic_tags": "bottleneck_potential, dependency_critical, temporal_sequence"
            },
            "assumptions": {
                "focus": "beliefs, expectations, and suppositions being made",
                "examples": "user behavior assumptions, market conditions, technology availability",
                "heuristic_tags": "high_risk, leverage_point, confidence_level"
            },
            "dependencies": {
                "focus": "prerequisites, sequences, and interdependencies between components",
                "examples": "approval processes, completion sequences, information dependencies",
                "heuristic_tags": "temporal_sequence, causal_chain, blocking_dependency"
            },
            "success_metrics": {
                "focus": "measurements, KPIs, and indicators of success",
                "examples": "performance metrics, user satisfaction, ROI, completion criteria",
                "heuristic_tags": "feedback_loop_trigger, reinforcement_potential"
            }
        }
        
        spec = node_type_specs.get(node_type, node_type_specs["constraints"])
        
        if node_type == "constraints":
            return f"""
Analyze constraints for this restaurant POS implementation:

User Request: "{user_input}"
Focus: {spec['focus']}

Identify specific constraints that limit or restrict the project. For each constraint, provide:
- Unique ID
- Constraint type (budget, timeline, technical, resource, regulatory, accessibility)
- Detailed constraint description
- Numeric value if applicable (e.g., $25000 for budget)
- Flexibility level (rigid, somewhat_flexible, flexible)

ALSO provide AI-actionable insights for constraint management:
- Focus on what an AI agent can do with available tools (APIs, databases, monitoring)
- Include automation level: fully_automated, human_in_loop, or human_required
- Specify required tools: web_search, database_query, api_call, file_analysis, etc.
- Indicate if human oversight is needed for approval/validation

Examples of AI-actionable constraint insights:
- "Monitor budget spending through expense tracking APIs" (fully_automated)
- "Generate timeline alerts and progress reports" (human_in_loop)
- "Research POS vendor pricing through web search" (fully_automated)

Focus on constraints with these heuristic patterns: {heuristics}
"""
        elif node_type == "technical_requirements":
            return f"""
Analyze technical requirements for this restaurant POS implementation:

User Request: "{user_input}"
Focus: {spec['focus']}

Identify specific technical capabilities needed. For each requirement, provide:
- Unique ID  
- Requirement name
- Detailed description
- Complexity level (low, medium, high, very_high)
- Importance level (critical, high, medium, low)

Focus on requirements with these heuristic patterns: {heuristics}
"""
        elif node_type == "assumptions":
            return f"""
Analyze assumptions for this restaurant POS implementation:

User Request: "{user_input}"
Focus: {spec['focus']}

Identify beliefs or expectations being made. For each assumption, provide:
- Unique ID
- The actual assumption text (what is being assumed)
- Confidence level (0.0 to 1.0)
- Impact if wrong (critical, high, medium, low)
- Optional validation method

Focus on assumptions with these heuristic patterns: {heuristics}
"""
        else:
            return f"Analyze {node_type} for: {user_input}"
    
    def _create_stakeholder_objective_prompt(self, user_input: str, stakeholder: Dict, objective: Dict) -> str:
        """Create sophisticated stakeholder expansion prompt - EXACT from AdaptiveIntentRunner"""
        
        stakeholder_name = stakeholder.get("name", "") or stakeholder.get("role", "User")
        stakeholder_role = stakeholder.get("role", "participant")
        objective_name = objective.get("name", "Unnamed Objective")
        objective_priority = objective.get("priority", "medium")
        
        # Domain detection for restaurant context
        domain = "restaurant" if "restaurant" in user_input.lower() or "pos" in user_input.lower() else "general"
        
        return f"""
{{
    "task": "stakeholder_objective_expansion",
    "input": {{
        "user_request": "{user_input}",
        "domain": "{domain}",
        "stakeholder": {{
            "name": "{stakeholder_name}",
            "role": "{stakeholder_role}"
        }},
        "objective": {{
            "name": "{objective_name}",
            "priority": "{objective_priority}"
        }},
        "applicable_heuristics": ["constraint_cascades", "risk_propagation", "leverage_points"],
        "heuristic_focus": {{"systems_thinking": "restaurant_operations", "business_analysis": "pos_implementation"}}
    }},
    "analysis_requirements": [
        "constraints", "technical_requirements", "assumptions", 
        "dependencies", "heuristic_annotations"
    ],
    "output_format": {{
        "constraints": [{{
            "id": "1",
            "type": "budget|timeline|technical|regulatory|resource",
            "description": "Detailed constraint description",
            "impact": "high|medium|low",
            "flexibility": "rigid|somewhat_flexible|flexible",
            "heuristic_tags": ["cascade_potential", "bottleneck_risk"]
        }}],
        "technical_requirements": [{{
            "id": "1", 
            "name": "Requirement name",
            "description": "Detailed description",
            "complexity": "low|medium|high",
            "importance": "critical|high|medium|low",
            "heuristic_tags": ["bottleneck_potential", "dependency_critical"]
        }}],
        "assumptions": [{{
            "id": "1",
            "assumption": "What is being assumed",
            "confidence": 0.0-1.0,
            "impact_if_wrong": "critical|high|medium|low",
            "heuristic_tags": ["high_risk", "leverage_point"]
        }}],
        "dependencies": [{{
            "depends_on": "What this depends on",
            "type": "information|approval|resource|completion",
            "criticality": "blocking|important|nice_to_have",
            "heuristic_tags": ["temporal_sequence", "causal_chain"]
        }}],
        "success_metrics": [{{
            "metric": "How success is measured",
            "target": "Target value",
            "timeframe": "When measured",
            "heuristic_tags": ["feedback_loop_trigger"]
        }}]
    }}
}}

Return only the analysis result JSON (not the instruction JSON).

IMPORTANT: Your response must be valid JSON that can be parsed. Start with {{ and end with }}. Do not include any text before or after the JSON object.
"""
    
    def _handle_stakeholder_objective_expansion(self, message):
        """Override base method to use structured output with Pydantic models"""
        print(f"🎯 INTENT CLASS: Using structured stakeholder handler")
        stakeholder = {}
        objective = {}
        
        try:
            user_input = message.get("user_input", "")
            stakeholder = message.get("stakeholder", {})
            objective = message.get("objective", {})
            
            stakeholder_name = stakeholder.get("name", "Unknown")
            objective_name = objective.get("name", "Unknown")
            
            # Use sophisticated expansion prompt from AdaptiveIntentRunner
            expansion_prompt = self._create_stakeholder_objective_prompt(user_input, stakeholder, objective)
            
            # Use structured LLM call - ENFORCES PYDANTIC SCHEMA
            result = self._call_llm_structured_sync(expansion_prompt, "stakeholder_objective_expansion", StakeholderObjectiveExpansion)
            
            # Store result - no validation needed, Pydantic already enforced schema
            expansion_key = message.get("expansion_key") or f"stakeholder_{stakeholder_name}_{objective_name}".replace(" ", "_").lower()
            self.shared_results[expansion_key] = {
                "stakeholder": stakeholder,
                "objective": objective,
                "expansion": result
            }
            
            return {"status": "completed", "expansion_key": expansion_key}
            
        except Exception as e:
            print(f"❌ Error in structured stakeholder-objective expansion: {str(e)}")
            # NO FALLBACKS - let it fail fast and fix the root cause
            expansion_key = message.get("expansion_key") or f"expansion_{len([k for k in self.shared_results.keys() if k.startswith('expansion_')])}"
            self.shared_results[expansion_key] = {
                "stakeholder": stakeholder,
                "objective": objective,
                "expansion": {},
                "error": str(e)
            }
            return {"status": "error", "expansion_key": expansion_key, "error": str(e)}
    
    def _handle_node_type_expansion(self, message):
        """Override base method to use structured output with Pydantic models"""
        node_type = message.get("node_type", "unknown")
        print(f"🎯 INTENT CLASS: Using structured node-type handler for {node_type}")
        
        try:
            user_input = message.get("user_input", "")
            assessment = message.get("assessment", {})
            node_type = message.get("node_type", "constraints")
            heuristics = message.get("heuristics", [])
            
            # Get the appropriate Pydantic model for this node type
            response_model = self._get_response_model_for_node_type(node_type)
            
            # Use sophisticated node-type-specific prompt from AdaptiveIntentRunner
            expansion_prompt = self._create_node_type_prompt(user_input, assessment, node_type, heuristics)
            
            # Use structured LLM call - ENFORCES PYDANTIC SCHEMA
            result = self._call_llm_structured_sync(expansion_prompt, f"node_type_{node_type}", response_model)
            
            # Store result - no validation needed, Pydantic already enforced schema
            expansion_key = message.get("expansion_key") or f"node_type_{node_type}"
            self.shared_results[expansion_key] = {
                "node_type": node_type,
                "heuristics": heuristics,
                "expansion": result
            }
            
            return {"status": "completed", "expansion_key": expansion_key}
            
        except Exception as e:
            print(f"❌ Error in structured node-type expansion ({node_type}): {str(e)}")
            # NO FALLBACKS - let it fail fast and fix the root cause
            expansion_key = message.get("expansion_key") or f"node_type_{node_type}"
            self.shared_results[expansion_key] = {
                "node_type": node_type,
                "heuristics": [],
                "expansion": {},
                "error": str(e)
            }
            return {"status": "error", "expansion_key": expansion_key, "error": str(e)}
    
    def _get_response_model_for_node_type(self, node_type: str):
        """Get the appropriate Pydantic response model for the node type"""
        model_map = {
            "constraints": AIConstraintsResponse,
            "technical_requirements": AITechnicalRequirementsResponse,
            "assumptions": AIAssumptionsResponse,
            "success_metrics": AISuccessMetricsResponse,
            "stakeholders": AIStakeholdersResponse,
            "dependencies": AIDependenciesResponse
        }
        
        if node_type in model_map:
            return model_map[node_type]
        else:
            print(f"⚠️ No specific model for node_type '{node_type}', using AIConstraintsResponse")
            return AIConstraintsResponse
    
    # ============================================================================
    # SOPHISTICATED HEURISTIC LINKING (from AdaptiveIntentRunner)
    # ============================================================================
    
    def apply_heuristic_linking(self, graph: KnowledgeGraph, analysis_data: Dict[str, Any]) -> KnowledgeGraph:
        """
        Apply sophisticated heuristic linking from AdaptiveIntentRunner
        Advanced systems thinking and graph theory analysis
        """
        print(f"🔗 Applying advanced intent-specific heuristic linking...")
        print(f"   📊 Processing analysis data with {len(graph.nodes())} nodes")
        
        # Convert to IntentKnowledgeGraph for advanced operations
        intent_graph = self._convert_to_intent_graph(graph)
        
        # Get expansion data from analysis
        expansions = self._extract_expansions_from_analysis(analysis_data)
        assessment = analysis_data.get("complexity_assessment", {})
        
        # PHASE 1: LLM Semantic Heuristics (from expansion data)
        print(f"   🧠 Phase 1: LLM-based semantic linking")
        self._link_risk_propagation(intent_graph, expansions)
        self._link_constraint_cascades(intent_graph, expansions)
        self._link_causal_dependencies(intent_graph, expansions)
        self._identify_feedback_loops(intent_graph, expansions)
        self._link_system_leverage_points(intent_graph, expansions)
        
        # PHASE 2: Mathematical graph analysis
        print(f"   📊 Phase 2: Mathematical graph analysis")
        resource_competition = self._analyze_resource_competition(intent_graph)
        temporal_prereqs = self._analyze_temporal_sequences(intent_graph)
        bottlenecks = self._analyze_graph_bottlenecks(intent_graph)
        authority = self._analyze_power_dynamics(intent_graph)
        
        print(f"      💰 Created {resource_competition} resource competition relationships")
        print(f"      ⏰ Created {temporal_prereqs} temporal prerequisite relationships")
        print(f"      🔧 Created {bottlenecks} bottleneck relationships")
        print(f"      👑 Created {authority} authority relationships")
        
        # PHASE 3: Domain-specific patterns
        print(f"   🏢 Phase 3: Domain-specific patterns")
        domain = self._extract_domain_from_assessment(assessment)
        self._apply_domain_systems_thinking(intent_graph, domain, expansions)
        
        print(f"   ✅ Applied advanced heuristic linking")
        
        # Convert back to standard KnowledgeGraph
        return self._convert_from_intent_graph(intent_graph)
    
    def _convert_to_intent_graph(self, graph: KnowledgeGraph) -> IntentKnowledgeGraph:
        """Convert standard graph to IntentKnowledgeGraph for advanced operations"""
        intent_graph = IntentKnowledgeGraph(
            graph_id=graph.graph_id,
            intent_name=f"Intent_{graph.graph_id}"
        )
        
        # Copy all nodes and edges
        for node_id in graph.nodes():
            attrs = graph.get_node_attributes(node_id)
            intent_graph.add_node(node_id, **attrs)
        
        for edge in graph.edges():
            source, target = edge
            attrs = graph.get_edge_attributes(source, target)
            intent_graph.add_edge(source, target, **attrs)
        
        return intent_graph
    
    def _convert_from_intent_graph(self, intent_graph: IntentKnowledgeGraph) -> KnowledgeGraph:
        """Convert IntentKnowledgeGraph back to standard KnowledgeGraph"""
        graph = KnowledgeGraph(
            graph_type=GraphType.INTENT,
            graph_id=intent_graph.graph_id
        )
        
        # Copy all nodes and edges
        for node_id in intent_graph.nodes():
            attrs = intent_graph.get_node_attributes(node_id)
            graph.add_node(node_id, **attrs)
        
        for edge in intent_graph.edges():
            source, target = edge
            attrs = intent_graph.get_edge_attributes(source, target)
            graph.add_edge(source, target, **attrs)
        
        return graph
    
    def _extract_expansions_from_analysis(self, analysis_data: Dict) -> Dict:
        """Extract expansion data from analysis results"""
        expansions = {}
        
        # Get parallel expansion results
        stakeholder_expansions = analysis_data.get("stakeholder_expansions", {})
        node_type_expansions = analysis_data.get("node_type_expansions", {})
        
        # Combine all expansions
        expansions.update(stakeholder_expansions)
        expansions.update(node_type_expansions)
        
        return expansions
    
    # ============================================================================
    # ADVANCED HEURISTIC METHODS (from AdaptiveIntentRunner)
    # ============================================================================
    
    def _link_risk_propagation(self, intent_graph: IntentKnowledgeGraph, expansions: Dict):
        """Model how risks propagate through the system using heuristic tags"""
        high_risk_count = 0
        total_assumptions = 0
        
        for expansion_key, expansion_data in expansions.items():
            expansion = expansion_data.get("expansion", {})
            assumptions = expansion.get("assumptions", [])
            total_assumptions += len(assumptions)
            
            for assumption in assumptions:
                confidence = assumption.get("confidence", 0.5)
                impact = assumption.get("impact_if_wrong", "medium")
                
                # High-risk assumptions (low confidence + high impact)
                if confidence < 0.7 and impact in ["high", "critical"]:
                    high_risk_count += 1
                    
                    # Find objectives that could be affected
                    obj_nodes = [n for n in intent_graph.nodes() 
                               if intent_graph.get_node_attributes(n).get('type') == 'primary_objective']
                    
                    # Create assumption node if not exists
                    assumption_id = f"risk_assumption_{high_risk_count}"
                    intent_graph.add_node(
                        assumption_id,
                        type="high_risk_assumption",
                        assumption=assumption.get("assumption", "Unknown assumption"),
                        confidence=confidence,
                        impact_if_wrong=impact
                    )
                    
                    # Link to objectives with risk propagation
                    for obj_node in obj_nodes[:2]:  # Limit connections
                        intent_graph.add_edge(assumption_id, obj_node, relationship="risk_propagates_to")
        
        print(f"   🚨 Found {high_risk_count}/{total_assumptions} high-risk assumptions")
    
    def _link_constraint_cascades(self, intent_graph: IntentKnowledgeGraph, expansions: Dict):
        """Model how constraint violations cascade through objectives"""
        rigid_constraints = []
        
        for expansion_key, expansion_data in expansions.items():
            expansion = expansion_data.get("expansion", {})
            constraints = expansion.get("constraints", [])
            
            for constraint in constraints:
                flexibility = constraint.get("flexibility", "moderate")
                constraint_type = constraint.get("type", "general")
                
                if flexibility == "rigid" and constraint_type in ["budget", "timeline"]:
                    rigid_constraints.append(constraint)
        
        # Create cascade relationships for rigid constraints
        for i, constraint in enumerate(rigid_constraints[:3]):  # Limit to 3
            constraint_id = f"rigid_constraint_{i+1}"
            intent_graph.add_node(
                constraint_id,
                type="rigid_constraint",
                constraint_type=constraint.get("type"),
                description=constraint.get("description", "Rigid constraint"),
                flexibility="rigid"
            )
            
            # Link to all objectives with cascade potential
            obj_nodes = [n for n in intent_graph.nodes() 
                        if intent_graph.get_node_attributes(n).get('type') == 'primary_objective']
            
            for obj_node in obj_nodes:
                intent_graph.add_edge(constraint_id, obj_node, relationship="cascade_failure_to")
        
        print(f"   ⛓️ Created {len(rigid_constraints)} constraint cascade relationships")
    
    def _link_causal_dependencies(self, intent_graph: IntentKnowledgeGraph, expansions: Dict):
        """Link based on cause-effect relationships from expansion analysis"""
        causal_links = 0
        
        for expansion_key, expansion_data in expansions.items():
            expansion = expansion_data.get("expansion", {})
            dependencies = expansion.get("dependencies", [])
            
            for dependency in dependencies:
                dep_type = dependency.get("type", "unknown")
                depends_on = dependency.get("depends_on", "")
                
                if dep_type == "causal" and depends_on:
                    # Find nodes that match the dependency description
                    for node_id in intent_graph.nodes():
                        node_attrs = intent_graph.get_node_attributes(node_id)
                        node_name = node_attrs.get('name', '').lower()
                        
                        if depends_on.lower() in node_name:
                            # Find objectives to link to
                            obj_nodes = [n for n in intent_graph.nodes() 
                                       if intent_graph.get_node_attributes(n).get('type') == 'primary_objective']
                            
                            for obj_node in obj_nodes[:2]:  # Limit connections
                                intent_graph.add_edge(node_id, obj_node, relationship="causal_prerequisite_for")
                                causal_links += 1
        
        print(f"   🔗 Created {causal_links} causal dependency relationships")
    
    def _identify_feedback_loops(self, intent_graph: IntentKnowledgeGraph, expansions: Dict):
        """Identify and model feedback loops in the system"""
        feedback_loops = 0
        
        # Look for success metrics that could create feedback
        for node_id in intent_graph.nodes():
            node_attrs = intent_graph.get_node_attributes(node_id)
            if node_attrs.get('type') == 'success_metric':
                # Success metrics can create feedback to objectives
                obj_nodes = [n for n in intent_graph.nodes() 
                           if intent_graph.get_node_attributes(n).get('type') == 'primary_objective']
                
                for obj_node in obj_nodes[:1]:  # One feedback loop per metric
                    intent_graph.add_edge(node_id, obj_node, relationship="feedback_reinforces")
                    feedback_loops += 1
        
        print(f"   🔄 Created {feedback_loops} feedback loop relationships")
    
    def _link_system_leverage_points(self, intent_graph: IntentKnowledgeGraph, expansions: Dict):
        """Identify high-leverage intervention points"""
        leverage_points = 0
        
        # High-degree nodes are potential leverage points
        for node_id in intent_graph.nodes():
            degree = intent_graph.degree(node_id)
            if degree > 3:  # High connectivity suggests leverage
                node_attrs = intent_graph.get_node_attributes(node_id)
                node_type = node_attrs.get('type')
                
                if node_type in ['stakeholder', 'constraint', 'technical_requirement']:
                    # Mark as leverage point
                    intent_graph.update_node_attributes(node_id, leverage_point=True)
                    leverage_points += 1
        
        print(f"   🎯 Identified {leverage_points} system leverage points")
    
    def _analyze_resource_competition(self, intent_graph: IntentKnowledgeGraph) -> int:
        """Analyze resource competition between objectives"""
        competition_links = 0
        
        # Find budget-related constraints
        budget_constraints = [n for n in intent_graph.nodes() 
                            if intent_graph.get_node_attributes(n).get('constraint_type') == 'budget']
        
        if budget_constraints:
            # Objectives competing for same budget
            obj_nodes = [n for n in intent_graph.nodes() 
                        if intent_graph.get_node_attributes(n).get('type') == 'primary_objective']
            
            for i, obj1 in enumerate(obj_nodes):
                for obj2 in obj_nodes[i+1:]:
                    intent_graph.add_edge(obj1, obj2, relationship="resource_competition")
                    competition_links += 1
        
        return competition_links
    
    def _analyze_temporal_sequences(self, intent_graph: IntentKnowledgeGraph) -> int:
        """Analyze temporal dependencies from technical requirements"""
        temporal_links = 0
        
        # Find tech requirements with temporal patterns
        for node_id in intent_graph.nodes():
            node_attrs = intent_graph.get_node_attributes(node_id)
            if node_attrs.get('type') == 'technical_requirement':
                tech_name = node_attrs.get('name', '').lower()
                
                # Look for prerequisite patterns
                for other_id in intent_graph.nodes():
                    other_attrs = intent_graph.get_node_attributes(other_id)
                    other_name = other_attrs.get('name', '').lower()
                    
                    # Basic temporal logic
                    if 'payment' in tech_name and 'order' in other_name:
                        intent_graph.add_edge(other_id, node_id, relationship="temporal_prerequisite_for")
                        temporal_links += 1
                    elif 'inventory' in tech_name and ('order' in other_name or 'payment' in other_name):
                        intent_graph.add_edge(other_id, node_id, relationship="temporal_prerequisite_for")
                        temporal_links += 1
        
        return temporal_links
    
    def _analyze_graph_bottlenecks(self, intent_graph: IntentKnowledgeGraph) -> int:
        """Identify bottlenecks based on node degree centrality"""
        bottlenecks_created = 0
        total_nodes = len(intent_graph.nodes())
        
        # Conservative threshold to avoid over-connection
        if total_nodes <= 10:
            threshold = 4
        elif total_nodes <= 30:
            threshold = 6
        else:
            threshold = 8
        
        bottleneck_nodes = []
        for node_id in intent_graph.nodes():
            degree = intent_graph.degree(node_id)
            if degree > threshold:
                node_attrs = intent_graph.get_node_attributes(node_id)
                bottleneck_nodes.append((node_id, degree))
        
        # Create bottleneck relationships for top nodes
        bottleneck_nodes.sort(key=lambda x: x[1], reverse=True)
        for node_id, degree in bottleneck_nodes[:3]:
            objectives = [n for n in intent_graph.nodes() 
                         if intent_graph.get_node_attributes(n).get('type') == 'primary_objective'][:3]
            
            for target_id in objectives:
                if target_id != node_id:
                    intent_graph.add_edge(node_id, target_id, relationship="potential_bottleneck_for")
                    bottlenecks_created += 1
        
        return bottlenecks_created
    
    def _analyze_power_dynamics(self, intent_graph: IntentKnowledgeGraph) -> int:
        """Analyze authority and influence relationships"""
        authority_links = 0
        
        # Find stakeholders with decision authority
        for node_id in intent_graph.nodes():
            node_attrs = intent_graph.get_node_attributes(node_id)
            if node_attrs.get('type') == 'stakeholder':
                role = node_attrs.get('role', '').lower()
                
                # REMOVED: Authority relationships - LLMs can't reliably determine organizational authority
                # Focus on simple ownership based on stakeholder-objective pairs from expansions
                pass
        
        return authority_links
    
    def _extract_domain_from_assessment(self, assessment: Dict) -> str:
        """Extract domain context from assessment"""
        stakeholders = assessment.get("stakeholders", [])
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
    
    def _apply_domain_systems_thinking(self, intent_graph: IntentKnowledgeGraph, domain: str, expansions: Dict):
        """Apply domain-specific systems thinking patterns"""
        if domain == "restaurant":
            # Restaurant-specific patterns: customer flow, kitchen operations, staff coordination
            self._apply_restaurant_patterns(intent_graph)
        elif domain == "mobile_app":
            # Mobile app patterns: user experience, performance, app store dynamics
            self._apply_mobile_app_patterns(intent_graph)
        else:
            # General business patterns
            self._apply_general_business_patterns(intent_graph)
    
    def _apply_restaurant_patterns(self, intent_graph: IntentKnowledgeGraph):
        """Apply restaurant-specific systems thinking"""
        # Look for POS-related nodes and create restaurant workflow links
        pos_nodes = [n for n in intent_graph.nodes() 
                    if 'pos' in intent_graph.get_node_attributes(n).get('name', '').lower()]
        
        order_nodes = [n for n in intent_graph.nodes() 
                      if 'order' in intent_graph.get_node_attributes(n).get('name', '').lower()]
        
        # POS systems affect order processing
        for pos_node in pos_nodes:
            for order_node in order_nodes:
                intent_graph.add_edge(pos_node, order_node, relationship="restaurant_workflow_enables")
    
    def _apply_mobile_app_patterns(self, intent_graph: IntentKnowledgeGraph):
        """Apply mobile app-specific systems thinking"""
        # Mobile apps have user experience dependencies
        pass
    
    def _apply_general_business_patterns(self, intent_graph: IntentKnowledgeGraph):
        """Apply general business systems thinking"""
        # General business workflow patterns
        pass
