#!/usr/bin/env python3
"""
Adaptive Intent Runner - Heuristic LLM Pattern for Intent Analysis
Uses intelligent two-stage approach: complexity assessment + stakeholder-objective expansion
"""

import asyncio
import json
import uuid
import asyncio
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI

# Add the Ethos-ZeroMQ path
import sys
sys.path.append(r'C:\Users\DamonWeiss\PycharmProjects\Ethos-ZeroMQ')

from ethos_zeromq.EthosZeroMQ_Engine import ZeroMQEngine
from ethos_zeromq.RouterDealerServer import RouterDealerServer, DealerClient
from intent_knowledge_graph_zmq import (
    IntentKnowledgeGraph, 
    IntentGraphSchema,
    IntentObjective,
    IntentStakeholder, 
    IntentConstraint,
    IntentAssumption,
    IntentTechnicalRequirement,
    IntentDerivationData
)

load_dotenv()

# Assessment-specific models (different from graph models)
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

# AI Agent Actionability Constraints
class AIActionableInsight(BaseModel):
    insight: str = Field(description="The insight or recommendation")
    actionable_by_ai: bool = Field(description="Can this be executed by AI agent with tools?")
    required_tools: List[str] = Field(description="Tools/APIs needed for AI execution")
    human_oversight_required: bool = Field(description="Requires human approval/input?")
    automation_level: str = Field(pattern="^(fully_automated|human_in_loop|human_required)$")

# Enhanced node-type response models with AI constraints
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

# Fallback models (for compatibility)
class ConstraintsResponse(BaseModel):
    constraints: List[IntentConstraint]

class TechnicalRequirementsResponse(BaseModel):
    technical_requirements: List[IntentTechnicalRequirement]

class AssumptionsResponse(BaseModel):
    assumptions: List[IntentAssumption]

# Task Decomposition Model
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

# QAQC Validation Models
class QAQCValidation(BaseModel):
    overall_quality_score: float = Field(ge=0.0, le=1.0, description="0.0-1.0 quality assessment of the analysis")
    identified_gaps: List[str] = Field(description="Missing elements or weak areas in the analysis")
    suggested_improvements: List[str] = Field(description="Specific enhancement recommendations")
    human_queries: List[str] = Field(description="Questions requiring human clarification before execution")
    validation_passed: bool = Field(description="Whether analysis meets quality standards for execution")
    critical_issues: List[str] = Field(description="Must-fix issues before proceeding with execution")
    confidence_assessment: str = Field(description="Overall confidence level in the analysis quality")

class AdaptiveIntentRunner:
    """
    Intelligent, heuristic intent analysis using adaptive LLM patterns
    
    Stage 1: Complexity Assessment (1 LLM call)
    - Simple: Direct answer or MCP tool + LLM one-shot
    - Complex: Identify stakeholders and their objectives
    
    Stage 2: Stakeholder-Objective Expansion (N parallel LLM calls)
    - One call per stakeholder-objective pair
    - Dynamic scaling based on actual complexity
    """
    
    def __init__(self, base_port: int = 5900):
        """Initialize Adaptive Intent Runner"""
        self.base_port = base_port
        self.zmq_engine = ZeroMQEngine()
        self.servers = {}
        self.shared_results = {}
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.llm_client = OpenAI(api_key=api_key)
    
    async def analyze_intent(self, user_input: str) -> Tuple[str, Optional[IntentKnowledgeGraph]]:
        """
        Adaptive intent analysis with heuristic LLM pattern
        
        Returns:
            Tuple[str, Optional[IntentKnowledgeGraph]]: 
            - "direct_answer" or "complex_analysis"
            - Direct answer string or IntentKnowledgeGraph
        """
        
        print(f"ADAPTIVE INTENT RUNNER")
        print("=" * 60)
        print(f"Input: {user_input}")
        
        start_time = time.time()
        
        try:
            # Stage 1: Complexity Assessment
            assessment_result = await self._stage1_complexity_assessment(user_input)
            
            # Debug: Show assessment details
            print(f"\n🔍 ASSESSMENT DEBUG:")
            print(f"   Complexity: {assessment_result.get('complexity')}")
            print(f"   Reasoning: {assessment_result.get('reasoning', 'No reasoning')}")
            print(f"   Stakeholders: {len(assessment_result.get('stakeholders', []))}")
            print(f"   Applicable Heuristics: {assessment_result.get('applicable_heuristics', [])}")
            print(f"   Heuristic Guidance Keys: {list(assessment_result.get('heuristic_guidance', {}).keys())}")
            
            # Handle both upper and lowercase complexity values
            complexity = assessment_result.get("complexity", "").lower()
            if complexity == "simple":
                print(f"   🎯 SIMPLE: {assessment_result.get('reasoning', 'No reasoning provided')}")
                print(f"\n✅ SIMPLE REQUEST - Direct answer provided")
                assessment_time = time.time() - start_time
                print(f"   Assessment time: {assessment_time:.2f}s")
                print(f"   LLM calls: 1")
                return "direct_answer", assessment_result.get("direct_answer", "Answer not provided")
            
            else:
                # Stage 2: Node-Type-Specific Expansion
                intent_graph = await self._stage2_node_type_expansion(
                    user_input, assessment_result
                )
                
                # Stage 3: Task Decomposition (for complex analysis)
                task_decomposition = await self._stage3_task_decomposition(
                    user_input, intent_graph, assessment_result, detail_level="high_level"
                )
                
                # Integrate tasks into the graph before QAQC
                intent_graph = self._integrate_tasks_into_graph(intent_graph, task_decomposition)
                
                # Stage 4: QAQC Validation (quality assurance and gap analysis)
                enhanced_graph, qaqc_validation = await self._stage4_qaqc_validation(
                    user_input, intent_graph, task_decomposition
                )
                
                total_time = time.time() - start_time
                llm_calls = assessment_result.get('total_calls', 6) + 2  # Assessment + node-type + task decomposition + qaqc
                
                print(f"\n✅ COMPLEX ANALYSIS COMPLETE")
                print(f"   Total time: {total_time:.2f}s")
                print(f"   LLM calls: {llm_calls} (1 assessment + {llm_calls-3} node-type + 1 task + 1 qaqc)")
                print(f"   Enhanced Graph: {len(enhanced_graph.nodes())} nodes, {len(enhanced_graph.edges())} edges")
                print(f"   Tasks: {len(task_decomposition.get('task_breakdown', []))} actionable tasks identified")
                print(f"   Quality Score: {qaqc_validation.get('overall_quality_score', 'N/A')}")
                
                return "complex_analysis", {
                    "graph": enhanced_graph, 
                    "tasks": task_decomposition,
                    "qaqc": qaqc_validation
                }
                
        finally:
            self._stop_servers()
    
    async def _stage1_complexity_assessment(self, user_input: str) -> Dict[str, Any]:
        """
        Stage 1: Assess complexity and determine processing path
        
        Returns:
            Dict with complexity assessment results
        """
        
        print(f"\n🔍 STAGE 1: COMPLEXITY ASSESSMENT")
        print("-" * 40)
        
        # Create assessment prompt
        assessment_prompt = self._create_assessment_prompt(user_input)
        
        # Single LLM call for complexity assessment
        print("📤 Sending complexity assessment request...")
        assessment_start = time.time()
        
        # First try without structured output to determine complexity
        result = self._call_llm_sync(assessment_prompt, "complexity_assessment")
        
        # If we got a result, try to parse it with the appropriate model
        if result and result.get("complexity") == "simple":
            # For simple requests, we already have what we need
            pass
        elif result and result.get("complexity") == "complex":
            # For complex requests, try to get structured output
            try:
                structured_result = self._call_llm_sync(assessment_prompt, "complexity_assessment", ComplexAssessment)
                if structured_result:
                    result = structured_result
            except Exception as e:
                print(f"⚠️ Structured complex assessment failed, using basic result: {str(e)}")
                # Use the basic result but add fallback values
                result["required_node_types"] = ["constraints", "technical_requirements", "assumptions"]
                result["node_type_heuristics"] = {
                    "constraints": ["constraint_cascades"],
                    "technical_requirements": ["bottlenecks", "temporal_sequences"], 
                    "assumptions": ["risk_propagation", "leverage_points"]
                }
        
        assessment_time = time.time() - assessment_start
        print(f"📥 Assessment completed in {assessment_time:.2f}s")
        
        # Parse assessment result (handle both upper and lowercase)
        complexity = result.get("complexity", "").lower()
        if complexity == "simple":
            print(f"   🎯 SIMPLE: {result.get('reasoning', 'Direct answer possible')}")
            return result
        
        else:
            print(f"   🔄 COMPLEX: {result.get('reasoning', 'Multiple stakeholders detected')}")
            
            # Calculate stakeholder-objective pairs
            stakeholders = result.get("stakeholders", [])
            total_pairs = 0
            
            for stakeholder in stakeholders:
                objectives = stakeholder.get("objectives", [])
                stakeholder_name = stakeholder.get('name', '') or stakeholder.get('role', 'User')
                print(f"   - {stakeholder_name}: {len(objectives)} objectives")
            
            result["total_pairs"] = total_pairs
            return result
    
    async def _stage2_node_type_expansion(self, user_input: str, assessment: Dict) -> IntentKnowledgeGraph:
        """
        Stage 2: Node-type-specific parallel expansion for consistent, specialized results
        
        Returns:
            IntentKnowledgeGraph: Complete graph with all node types and relationships
        """
        
        print(f"\n🚀 STAGE 2: NODE-TYPE-SPECIFIC EXPANSION")
        print("-" * 50)
        
        # Get required node types and their heuristics from assessment with fallbacks
        required_node_types = assessment.get("required_node_types", ["constraints", "technical_requirements", "assumptions"])
        node_type_heuristics = assessment.get("node_type_heuristics", {
            "constraints": ["constraint_cascades"],
            "technical_requirements": ["bottlenecks", "temporal_sequences"], 
            "assumptions": ["risk_propagation", "leverage_points"],
            "dependencies": ["temporal_sequences", "causal_chains"],
            "success_metrics": ["feedback_loops"]
        })
        stakeholders = assessment.get("stakeholders", [])
        
        print(f"📋 Required Node Types: {required_node_types}")
        print(f"🧠 Node-Type Heuristics: {list(node_type_heuristics.keys())}")
        
        # Start ZMQ servers for parallel processing
        num_calls = len(required_node_types)
        print(f"🚀 Starting {num_calls} ZMQ servers for parallel node-type expansion...")
        
        self._start_expansion_servers(num_calls)
        
        # Reset shared results
        self.shared_results = {}
        
        # Create node-type-specific expansion tasks
        clients = []
        print(f"📤 Sending {num_calls} parallel node-type requests...")
        
        for i, node_type in enumerate(required_node_types):
            content_type = f"node_type_{i}"
            heuristics = node_type_heuristics.get(node_type, [])
            
            # Create DealerClient for this node type
            client = DealerClient(connect=f"tcp://localhost:{self.base_port + i}")
            clients.append(client)
            
            # Send request with node type context
            request_data = {
                "user_input": user_input,
                "assessment": assessment,
                "node_type": node_type,
                "heuristics": heuristics
            }
            
            client.send_message(json.dumps(request_data), content_type)
            print(f"   📤 Node Type {i+1}: {node_type} (heuristics: {heuristics})")
        
        # Wait for all responses
        print(f"📥 Waiting for {num_calls} node-type expansion responses...")
        for i, client in enumerate(clients):
            response = client.receive_message()
            print(f"   ✅ Node Type {i+1} completed")
            client.close()
        
        # Build graph from node-type-specific data
        intent_graph = self._build_graph_from_node_types(user_input, assessment, stakeholders, self.shared_results)
        
        # Update assessment with actual call count
        assessment['total_calls'] = 1 + num_calls
        
        return intent_graph
    
    async def _stage3_task_decomposition(self, user_input: str, intent_graph: IntentKnowledgeGraph, assessment: Dict, detail_level: str = "high_level") -> Dict:
        """
        Stage 3: Task Decomposition for complex analysis
        
        Returns:
            Dict with task breakdown suitable for AI agent delegation
        """
        
        print(f"\n🎯 STAGE 3: TASK DECOMPOSITION")
        print("-" * 50)
        
        # Create task decomposition prompt
        task_prompt = self._create_task_decomposition_prompt(user_input, intent_graph, assessment, detail_level)
        
        print(f"📤 Generating {detail_level} actionable task breakdown...")
        task_start = time.time()
        
        # Call LLM with structured output for task decomposition
        try:
            result = self._call_llm_sync(task_prompt, "task_decomposition", TaskDecompositionResponse)
        except Exception as e:
            print(f"⚠️ Task decomposition failed: {str(e)}")
            # Fallback to basic task structure
            result = {
                "task_breakdown": [],
                "execution_sequence": [],
                "critical_path": [],
                "coverage_assessment": "Fallback structure - limited coverage"
            }
        
        task_time = time.time() - task_start
        print(f"📥 Task decomposition completed in {task_time:.2f}s")
        
        if result.get("task_breakdown"):
            print(f"   📋 {len(result['task_breakdown'])} tasks identified")
            print(f"   🎯 {len(result['critical_path'])} critical path tasks")
            
            # Display ALL tasks
            print(f"\n📋 COMPLETE TASK BREAKDOWN:")
            for i, task in enumerate(result['task_breakdown'], 1):
                task_name = task.get('task_name', 'Unknown task')
                priority = task.get('priority', 'medium')
                duration = task.get('estimated_duration', 'TBD')
                agent_type = task.get('assigned_agent_type', 'general')
                print(f"   {i}. {task_name}")
                print(f"      Priority: {priority} | Duration: {duration} | Agent: {agent_type}")
                
                tools = task.get('required_tools', [])
                if tools:
                    print(f"      Tools: {', '.join(tools[:3])}{'...' if len(tools) > 3 else ''}")
                
                dependencies = task.get('dependencies', [])
                if dependencies:
                    print(f"      Dependencies: {', '.join(dependencies)}")
                print()
            
            # Show coverage assessment
            coverage = result.get('coverage_assessment', '')
            if coverage:
                print(f"📊 Coverage Assessment: {coverage}")
        
        return result
    
    def _integrate_tasks_into_graph(self, intent_graph: IntentKnowledgeGraph, task_decomposition: Dict) -> IntentKnowledgeGraph:
        """Integrate task breakdown into the knowledge graph as task nodes"""
        
        print(f"🔗 Integrating {len(task_decomposition.get('task_breakdown', []))} tasks into graph...")
        
        task_breakdown = task_decomposition.get('task_breakdown', [])
        
        for task in task_breakdown:
            task_id = task.get('task_id', f"task_{len(intent_graph.nodes())}")
            task_name = task.get('task_name', 'Unknown Task')
            
            # Add task node to graph
            intent_graph.add_node(
                task_id,
                type="task",
                name=task_name,
                description=task.get('description', ''),
                priority=task.get('priority', 'medium'),
                estimated_duration=task.get('estimated_duration', 'TBD'),
                assigned_agent_type=task.get('assigned_agent_type', 'general'),
                required_tools=task.get('required_tools', [])
            )
            
            # Link tasks to relevant objectives they fulfill
            for obj_id in intent_graph.nodes():
                obj_attrs = intent_graph.get_node_attributes(obj_id)
                if obj_attrs.get('type') == 'primary_objective':
                    obj_name = obj_attrs.get('name', '').lower()
                    task_name_lower = task_name.lower()
                    
                    # Smart linking based on semantic similarity
                    if any(word in obj_name for word in ['budget', 'cost', 'financial']) and 'budget' in task_name_lower:
                        intent_graph.add_edge(task_id, obj_id, relationship="fulfills")
                    elif any(word in obj_name for word in ['vendor', 'procurement', 'selection']) and 'vendor' in task_name_lower:
                        intent_graph.add_edge(task_id, obj_id, relationship="fulfills")
                    elif any(word in obj_name for word in ['technical', 'system', 'implementation']) and 'technical' in task_name_lower:
                        intent_graph.add_edge(task_id, obj_id, relationship="fulfills")
                    elif any(word in obj_name for word in ['staff', 'training', 'stakeholder']) and any(word in task_name_lower for word in ['stakeholder', 'coordination']):
                        intent_graph.add_edge(task_id, obj_id, relationship="fulfills")
            
            # Link task dependencies
            dependencies = task.get('dependencies', [])
            for dep_task_id in dependencies:
                if dep_task_id in [t.get('task_id') for t in task_breakdown]:
                    intent_graph.add_edge(dep_task_id, task_id, relationship="prerequisite_for")
        
        print(f"   Added {len(task_breakdown)} task nodes with relationships")
        return intent_graph
    
    async def _stage4_qaqc_validation(self, user_input: str, intent_graph: IntentKnowledgeGraph, tasks: Dict) -> tuple[IntentKnowledgeGraph, Dict]:
        """
        Stage 4: Quality Assurance, Quality Control, and Gap Analysis
        
        Returns:
            Dict with validation results, gaps, improvements, and human queries
        """
        
        print(f"\n🔍 STAGE 4: QAQC VALIDATION & GAP ANALYSIS")
        print("-" * 50)
        
        # Convert graph and tasks to JSON for LLM analysis
        graph_json = self._serialize_graph_for_qaqc(intent_graph)
        tasks_json = json.dumps(tasks, indent=2)
        
        # Create QAQC validation prompt
        qaqc_prompt = self._create_qaqc_prompt(user_input, graph_json, tasks_json)
        
        print("📤 Performing quality assurance validation...")
        qaqc_start = time.time()
        
        # Call LLM with structured output for QAQC validation
        try:
            result = self._call_llm_sync(qaqc_prompt, "qaqc_validation", QAQCValidation)
        except Exception as e:
            print(f"⚠️ QAQC validation failed: {str(e)}")
            # Fallback to basic validation structure
            result = {
                "overall_quality_score": 0.7,
                "identified_gaps": ["QAQC validation failed - manual review recommended"],
                "suggested_improvements": ["Re-run QAQC validation when system is stable"],
                "human_queries": ["Please manually review the analysis quality"],
                "validation_passed": False,
                "critical_issues": ["QAQC system error"],
                "confidence_assessment": "Low due to validation system failure"
            }
        
        qaqc_time = time.time() - qaqc_start
        print(f"📥 QAQC validation completed in {qaqc_time:.2f}s")
        
        # Display QAQC results
        if result:
            quality_score = result.get("overall_quality_score", 0.0)
            validation_passed = result.get("validation_passed", False)
            
            print(f"   📊 Quality Score: {quality_score:.2f}/1.0")
            print(f"   ✅ Validation: {'PASSED' if validation_passed else 'NEEDS REVIEW'}")
            
            gaps = result.get("identified_gaps", [])
            if gaps:
                print(f"   🔍 Identified Gaps ({len(gaps)}):")
                for i, gap in enumerate(gaps[:3], 1):
                    print(f"      {i}. {gap}")
                if len(gaps) > 3:
                    print(f"      ... and {len(gaps) - 3} more")
            
            human_queries = result.get("human_queries", [])
            if human_queries:
                print(f"   ❓ Human Queries ({len(human_queries)}):")
                for i, query in enumerate(human_queries[:2], 1):
                    print(f"      {i}. {query}")
                if len(human_queries) > 2:
                    print(f"      ... and {len(human_queries) - 2} more")
            
            critical_issues = result.get("critical_issues", [])
            if critical_issues:
                print(f"   🚨 Critical Issues ({len(critical_issues)}):")
                for i, issue in enumerate(critical_issues[:2], 1):
                    print(f"      {i}. {issue}")
        
        # Enhance the graph based on QAQC results
        enhanced_graph = self._enhance_graph_with_qaqc(intent_graph, result)
        
        return enhanced_graph, result
    
    def _enhance_graph_with_qaqc(self, intent_graph: IntentKnowledgeGraph, qaqc_result: Dict) -> IntentKnowledgeGraph:
        """Enhance the graph based on QAQC validation results"""
        
        print(f"🔧 Enhancing graph based on QAQC validation...")
        
        # Add human query nodes
        human_queries = qaqc_result.get("human_queries", [])
        for i, query in enumerate(human_queries, 1):
            query_id = f"human_query_{i}"
            intent_graph.add_node(
                query_id,
                type="human_query",
                name=f"Human Query {i}",
                query=query,
                status="pending",
                priority="high" if "critical" in query.lower() or "budget" in query.lower() else "medium"
            )
            
            # Link queries to relevant stakeholders
            for node_id in intent_graph.nodes():
                node_attrs = intent_graph.get_node_attributes(node_id)
                if node_attrs.get('type') == 'stakeholder':
                    intent_graph.add_edge(query_id, node_id, relationship="requires_input_from")
        
        # Add risk nodes for critical issues
        critical_issues = qaqc_result.get("critical_issues", [])
        if critical_issues:
            # Create risk category if it doesn't exist
            risk_nodes = [n for n in intent_graph.nodes() if intent_graph.get_node_attributes(n).get('type') == 'risk']
            if not risk_nodes:
                for i, issue in enumerate(critical_issues, 1):
                    risk_id = f"risk_{i}"
                    intent_graph.add_node(
                        risk_id,
                        type="risk",
                        name=f"Critical Risk {i}",
                        description=issue,
                        severity="high",
                        status="identified"
                    )
                    
                    # Link risks to relevant constraints or assumptions
                    for node_id in intent_graph.nodes():
                        node_attrs = intent_graph.get_node_attributes(node_id)
                        node_type = node_attrs.get('type')
                        node_name = node_attrs.get('name', '').lower()
                        
                        if node_type in ['constraint', 'assumption']:
                            if "budget" in issue.lower() and "budget" in node_name:
                                intent_graph.add_edge(risk_id, node_id, relationship="threatens")
                            elif "timeline" in issue.lower() and any(word in node_name for word in ["time", "week", "deadline"]):
                                intent_graph.add_edge(risk_id, node_id, relationship="threatens")
        
        # Prune non-actionable items based on validation score
        quality_score = qaqc_result.get("overall_quality_score", 1.0)
        if quality_score < 0.6:  # Low quality - prune weak connections
            print(f"   🧹 Pruning weak connections due to low quality score ({quality_score:.2f})")
            
            # Remove edges with generic relationships that don't add value
            edges_to_remove = []
            for edge in intent_graph.edges():
                source_id, target_id = edge
                edge_attrs = intent_graph.get_edge_attributes(source_id, target_id)
                relationship = edge_attrs.get('relationship', '')
                
                # Remove very generic or weak relationships
                if relationship in ['unknown', 'related_to', 'connected_to']:
                    edges_to_remove.append(edge)
            
            for edge in edges_to_remove:
                intent_graph.remove_edge(edge[0], edge[1])
            
            if edges_to_remove:
                print(f"   Removed {len(edges_to_remove)} weak relationship(s)")
        
        # Add improvement suggestions as enhancement nodes
        improvements = qaqc_result.get("suggested_improvements", [])
        for i, improvement in enumerate(improvements[:3], 1):  # Limit to top 3
            improvement_id = f"improvement_{i}"
            intent_graph.add_node(
                improvement_id,
                type="improvement",
                name=f"Suggested Improvement {i}",
                description=improvement,
                status="suggested",
                priority="medium"
            )
        
        nodes_added = len(human_queries) + len(critical_issues) + min(len(improvements), 3)
        print(f"   Added {nodes_added} enhancement nodes (queries, risks, improvements)")
        
        return intent_graph
    
    def _serialize_graph_for_qaqc(self, intent_graph: IntentKnowledgeGraph) -> str:
        """Convert intent graph to JSON format for QAQC analysis"""
        
        graph_data = {
            "nodes": {},
            "relationships": []
        }
        
        # Serialize nodes by type
        for node_id in intent_graph.nodes():
            node_attrs = intent_graph.get_node_attributes(node_id)
            node_type = node_attrs.get('type', 'unknown')
            
            if node_type not in graph_data["nodes"]:
                graph_data["nodes"][node_type] = []
            
            graph_data["nodes"][node_type].append({
                "id": node_id,
                "name": node_attrs.get('name', 'Unknown'),
                "attributes": {k: v for k, v in node_attrs.items() if k not in ['type', 'name']}
            })
        
        # Serialize relationships
        for edge in intent_graph.edges():
            source_id, target_id = edge
            edge_attrs = intent_graph.get_edge_attributes(source_id, target_id)
            relationship = edge_attrs.get('relationship', 'unknown')
            
            source_attrs = intent_graph.get_node_attributes(source_id)
            target_attrs = intent_graph.get_node_attributes(target_id)
            
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
    
    def _create_qaqc_prompt(self, user_input: str, graph_json: str, tasks_json: str) -> str:
        """Create comprehensive QAQC validation prompt"""
        
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
   - 0.0-0.4: Poor, major problems requiring significant rework

2. IDENTIFIED GAPS: What's missing or weak?
   - Missing stakeholders or objectives
   - Overlooked constraints or requirements
   - Incomplete task coverage
   - Logical inconsistencies in relationships

3. SUGGESTED IMPROVEMENTS: Specific enhancements
   - Additional nodes or relationships needed
   - Better task definitions or sequencing
   - More realistic constraints or assumptions

4. HUMAN QUERIES: Questions needing clarification
   - Ambiguous requirements in original request
   - Missing technical specifications
   - Budget/timeline clarifications needed
   - Stakeholder contact information

5. VALIDATION PASSED: Can this proceed to execution?
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
    
    def _create_task_decomposition_prompt(self, user_input: str, intent_graph: IntentKnowledgeGraph, assessment: Dict, detail_level: str) -> str:
        """Create prompt for task decomposition with no gaps"""
        
        # Extract key context from the graph
        objectives_count = len([n for n in intent_graph.nodes() if intent_graph.get_node_attributes(n).get('type') == 'primary_objective'])
        constraints_count = len([n for n in intent_graph.nodes() if intent_graph.get_node_attributes(n).get('type') == 'constraint'])
        tech_reqs_count = len([n for n in intent_graph.nodes() if intent_graph.get_node_attributes(n).get('type') == 'technical_requirement'])
        
        detail_guidance = {
            "high_level": {
                "task_count": "3-7 strategic tasks",
                "scope": "broad, strategic phases that cover major project areas",
                "example": "Budget Management, Technical Implementation, Stakeholder Coordination"
            },
            "detailed": {
                "task_count": "7-10 specific tasks", 
                "scope": "specific, actionable tasks with clear deliverables",
                "example": "Configure payment gateway API, Train staff on order entry, Generate vendor comparison report"
            }
        }
        
        guidance = detail_guidance.get(detail_level, detail_guidance["high_level"])
        
        return f"""
Break down this complex project into {detail_level} actionable tasks for AI agent delegation:

Original Request: "{user_input}"
Analysis Summary: {objectives_count} objectives, {constraints_count} constraints, {tech_reqs_count} technical requirements

CRITICAL REQUIREMENTS:
1. Create {guidance['task_count']} - FEWER tasks are better if they provide COMPLETE coverage
2. Tasks must be {guidance['scope']}
3. Tasks must be COMPLETE with no gaps - assume each task will be delegated and completed independently
4. Each task must be actionable by an AI agent with specified tools
5. Include clear dependencies and execution sequence
6. Specify which type of AI agent is best suited for each task

QUALITY OVER QUANTITY: Prefer fewer, well-scoped tasks that fully cover the problem rather than many granular tasks.

Example {detail_level} tasks: {guidance['example']}

Task Types to Consider:
- Budget monitoring and financial analysis
- Technical integration and API testing  
- Stakeholder communication and training coordination
- Vendor evaluation and procurement
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
    
    async def _stage2_stakeholder_expansion(self, user_input: str, domain: str, assessment: Dict) -> IntentKnowledgeGraph:
        """
        Stage 2: Stakeholder-objective expansion for consistent, specialized results
        
        Returns:
            Completed IntentKnowledgeGraph
        """
        
        print(f"\n🚀 STAGE 2: STAKEHOLDER-OBJECTIVE EXPANSION")
        print("-" * 50)
        
        stakeholders = assessment.get("stakeholders", [])
        total_pairs = assessment.get("total_pairs", 0)
        
        if total_pairs == 0:
            print("⚠️ No stakeholder-objective pairs found, creating minimal graph")
            return self._create_minimal_graph(user_input, domain, assessment)
        
        # Start ZMQ servers for parallel processing
        self._start_expansion_servers(total_pairs)
        
        # Reset shared results
        self.shared_results = {}
        
        # Create clients and send parallel requests
        clients = []
        pair_index = 0
        
        print(f"📤 Sending {total_pairs} parallel expansion requests...")
        
        for stakeholder in stakeholders:
            stakeholder_name = stakeholder.get("name", "") or stakeholder.get("role", "User")
            stakeholder_role = stakeholder.get("role", "participant")
            
            for objective in stakeholder.get("objectives", []):
                objective_name = objective.get("name", "Unnamed Objective")
                
                # Create expansion prompt for this stakeholder-objective pair
                expansion_prompt = self._create_expansion_prompt(
                    user_input, domain, assessment, stakeholder, objective
                )
                
                # Send request to dedicated server
                port = self.base_port + pair_index
                client = DealerClient(connect=f"tcp://localhost:{port}")
                clients.append(client)
                
                message = {
                    "user_input": user_input,
                    "domain": domain,
                    "stakeholder": stakeholder,
                    "objective": objective,
                    "assessment": assessment
                }
                
                client.send_message(message, f"expansion_{pair_index}")
                
                print(f"   📤 Pair {pair_index + 1}: {stakeholder_name} → {objective_name}")
                pair_index += 1
        
        # Wait for all expansions to complete
        print(f"📥 Waiting for {total_pairs} expansion responses...")
        
        completed_pairs = set()
        while len(completed_pairs) < total_pairs:
            for i in range(total_pairs):
                if i not in completed_pairs and f"expansion_{i}" in self.shared_results:
                    completed_pairs.add(i)
                    print(f"   ✅ Pair {i + 1} completed")
            
            if len(completed_pairs) < total_pairs:
                await asyncio.sleep(0.1)
        
        # Clean up clients
        for client in clients:
            try:
                client.receive_message()
            except:
                pass
            client.close()
        
        # Build intent graph from expansion results
        try:
            intent_graph = self._build_graph_from_expansions(user_input, domain, assessment, self.shared_results)
            return intent_graph
        except Exception as e:
            print(f"❌ Error building graph: {str(e)}")
            print(f"   Expansion results: {len(self.shared_results)} items")
            # Return minimal graph as fallback
            return self._create_minimal_graph(user_input, domain, assessment)
    
    def _start_expansion_servers(self, num_pairs: int):
        """Start ZMQ servers for stakeholder-objective expansion"""
        
        print(f"🚀 Starting {num_pairs} ZMQ servers for parallel expansion...")
        
        for i in range(num_pairs):
            port = self.base_port + i
            
            server = RouterDealerServer(
                engine=self.zmq_engine,
                bind=f"tcp://*:{port}",
                connect=f"tcp://localhost:{port}"
            )
            
            # Register handler - check if it's node-type or stakeholder expansion
            handler_name = f"expansion_{i}" if hasattr(self, '_using_stakeholder_expansion') else f"node_type_{i}"
            handler_method = self._handle_stakeholder_expansion if hasattr(self, '_using_stakeholder_expansion') else self._handle_node_type_expansion
            server.register_handler(handler_name, handler_method)
            server.start()
            
            self.servers[f"expansion_{i}"] = server
            time.sleep(0.02)  # Brief delay
    
    def _stop_servers(self):
        """Stop all ZMQ servers"""
        if self.servers:
            for server in self.servers.values():
                server.stop()
            self.servers.clear()
    
    def _handle_stakeholder_expansion(self, message):
        """Handle stakeholder-objective expansion request"""
        
        try:
            user_input = message.get("user_input", "")
            domain = message.get("domain", "unknown")
            stakeholder = message.get("stakeholder", {})
            objective = message.get("objective", {})
            assessment = message.get("assessment", {})
            
            # Create expansion prompt
            expansion_prompt = self._create_expansion_prompt(user_input, domain, assessment, stakeholder, objective)
            
            # Call LLM
            result = self._call_llm_sync(expansion_prompt, "stakeholder_expansion")
            
            # Validate result
            if not isinstance(result, dict):
                print(f"⚠️ Invalid expansion result type: {type(result)}")
                result = {}
            
            # Store result with unique key
            expansion_key = f"expansion_{len([k for k in self.shared_results.keys() if k.startswith('expansion_')])}"
            self.shared_results[expansion_key] = {
                "stakeholder": stakeholder,
                "objective": objective,
                "expansion": result
            }
            
            return {"status": "completed", "expansion_key": expansion_key}
            
        except Exception as e:
            print(f"❌ Error in stakeholder expansion: {str(e)}")
            # Store empty result to prevent hanging
            expansion_key = f"expansion_{len([k for k in self.shared_results.keys() if k.startswith('expansion_')])}"
            self.shared_results[expansion_key] = {
                "stakeholder": stakeholder,
                "objective": objective,
                "expansion": {}
            }
            return {"status": "error", "expansion_key": expansion_key, "error": str(e)}
    
    def _handle_node_type_expansion(self, message):
        """Handle node-type-specific expansion request"""
        
        node_type = "unknown"  # Initialize for error handling
        try:
            # Parse JSON message
            if isinstance(message, str):
                message = json.loads(message)
            
            user_input = message.get("user_input", "")
            assessment = message.get("assessment", {})
            node_type = message.get("node_type", "constraints")
            heuristics = message.get("heuristics", [])
            
            # Create node-type-specific prompt
            expansion_prompt = self._create_node_type_prompt(user_input, assessment, node_type, heuristics)
            
            # Call LLM with AI-enhanced structured output based on node type
            response_model = None
            if node_type == "constraints":
                response_model = AIConstraintsResponse
            elif node_type == "technical_requirements":
                response_model = AITechnicalRequirementsResponse
            elif node_type == "assumptions":
                response_model = AIAssumptionsResponse
            
            result = self._call_llm_sync(expansion_prompt, f"node_type_{node_type}", response_model)
            
            # Validate result
            if not isinstance(result, dict):
                print(f"⚠️ Invalid node-type result type: {type(result)}")
                result = {node_type: []}
            
            # Store result with unique key
            expansion_key = f"node_type_{node_type}"
            self.shared_results[expansion_key] = {
                "node_type": node_type,
                "heuristics": heuristics,
                "expansion": result
            }
            
            return {"status": "completed", "expansion_key": expansion_key}
        
        except Exception as e:
            print(f"❌ Error in node-type expansion ({node_type}): {str(e)}")
            # Store empty result to prevent hanging
            expansion_key = f"node_type_{node_type}"
            self.shared_results[expansion_key] = {
                "node_type": node_type,
                "heuristics": [],
                "expansion": {node_type: []}
            }
            return {"status": "error", "expansion_key": expansion_key, "error": str(e)}
    
    def _create_assessment_prompt(self, user_input: str) -> str:
        """Create assessment prompt for Pydantic structured output"""
        
        return f"""
Analyze this user request for complexity and stakeholder involvement:

User Request: "{user_input}"

Determine if this is:
- SIMPLE: Direct factual question, basic lookup, simple how-to instruction
- COMPLEX: Multi-stakeholder business planning, resource allocation, project management

If SIMPLE: Provide a direct answer.

If COMPLEX: 
1. Identify all stakeholders involved (people, roles, groups)
2. For each stakeholder, identify their specific objectives
3. The system will automatically include appropriate node types and heuristics for analysis

Focus on identifying the key people/roles involved and what each wants to achieve.
"""
    
    def _create_node_type_prompt(self, user_input: str, assessment: Dict, node_type: str, heuristics: list) -> str:
        """Create specialized prompt for specific node type expansion"""
        
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
    
    def _create_expansion_prompt(self, user_input: str, domain: str, assessment: Dict, stakeholder: Dict, objective: Dict) -> str:
        """Create heuristic-guided JSON expansion prompt"""
        
        stakeholder_name = stakeholder.get("name", "") or stakeholder.get("role", "User")
        stakeholder_role = stakeholder.get("role", "participant")
        objective_name = objective.get("name", "Unnamed Objective")
        objective_priority = objective.get("priority", "medium")
        
        # Get heuristic guidance from assessment
        applicable_heuristics = assessment.get('applicable_heuristics', [])
        heuristic_guidance = assessment.get('heuristic_guidance', {})
        
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
        "applicable_heuristics": {applicable_heuristics},
        "heuristic_focus": {heuristic_guidance}
    }},
    "analysis_requirements": [
        "constraints", "technical_requirements", "assumptions", 
        "dependencies", "success_metrics", "heuristic_annotations"
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
"""
    
    def _create_minimal_graph(self, user_input: str, domain: str, assessment: Dict) -> IntentKnowledgeGraph:
        """Create minimal graph for cases with no stakeholder-objective pairs"""
        
        intent_graph = IntentKnowledgeGraph("minimal", "Minimal Graph")
        
        # Add basic objective from user input
        intent_graph.add_objective(
            "obj_1",
            name="User Request",
            description=user_input,
            priority="high"
        )
        
        # Add user as stakeholder
        intent_graph.add_stakeholder(
            "stake_1",
            name="User",
            role="requester"
        )
        
        # Link user to objective
        intent_graph.add_edge("stake_1", "obj_1", relationship="requests")
        
        return intent_graph
    
    def _build_graph_from_expansions(self, user_input: str, domain: str, assessment: Dict, expansions: Dict) -> IntentKnowledgeGraph:
        """Build complete intent graph from expansion results"""
        
        print(f"🏗️ Building intent graph from {len(expansions)} expansions...")
        
        # Debug: Show expansion data
        print(f"\n📋 EXPANSION DEBUG:")
        for key, data in expansions.items():
            expansion = data.get("expansion", {})
            print(f"   {key}:")
            print(f"     • Constraints: {len(expansion.get('constraints', []))}")
            print(f"     • Tech Requirements: {len(expansion.get('technical_requirements', []))}")
            print(f"     • Assumptions: {len(expansion.get('assumptions', []))}")
            print(f"     • Dependencies: {len(expansion.get('dependencies', []))}")
            print(f"     • Success Metrics: {len(expansion.get('success_metrics', []))}")
            
            # Show sample data
            if expansion.get('constraints'):
                sample_constraint = expansion['constraints'][0]
                print(f"     • Sample Constraint: {sample_constraint.get('description', 'No description')[:50]}...")
                print(f"     • Heuristic Tags: {sample_constraint.get('heuristic_tags', [])}")
        
        intent_graph = IntentKnowledgeGraph("adaptive", "Adaptive Analysis Graph")
        
        # Track IDs to avoid duplicates
        constraint_id = 1
        tech_req_id = 1
        assumption_id = 1
        dependency_id = 1
        
        # Process each expansion
        for expansion_key, expansion_data in expansions.items():
            stakeholder = expansion_data["stakeholder"]
            objective = expansion_data["objective"]
            expansion = expansion_data["expansion"]
            
            stakeholder_name = stakeholder.get("name", "Unknown")
            objective_name = objective.get("name", "Unknown")
            
            # Add stakeholder (if not already added)
            stakeholder_id = f"stake_{stakeholder_name.lower().replace(' ', '_')}"
            if stakeholder_id not in [n for n in intent_graph.nodes()]:
                intent_graph.add_stakeholder(
                    stakeholder_id,
                    name=stakeholder_name,
                    role=stakeholder.get("role", "participant")
                )
            
            # Add objective (ensure unique ID)
            objective_base = objective_name.lower().replace(' ', '_').replace('-', '_')[:20]
            objective_id = f"obj_{objective_base}_{hash(stakeholder_name + objective_name) % 1000}"
            
            try:
                intent_graph.add_objective(
                    objective_id,
                    name=objective_name,
                    description=f"{stakeholder_name}'s objective: {objective_name}",
                    priority=objective.get("priority", "medium")
                )
            except Exception as e:
                print(f"⚠️ Error adding objective {objective_id}: {str(e)}")
                continue
            
            # Link stakeholder to objective
            intent_graph.add_edge(stakeholder_id, objective_id, relationship="pursues")
            
            # Add constraints
            for constraint in expansion.get("constraints", []):
                constraint_node_id = f"con_{constraint_id}"
                intent_graph.add_constraint(
                    constraint_node_id,
                    constraint_type=constraint.get("type", "other"),
                    constraint=constraint.get("description", ""),
                    name=f"{constraint.get('type', 'Other').title()} Constraint"
                )
                
                # Link constraint to objective
                intent_graph.add_edge(constraint_node_id, objective_id, relationship="constrains")
                constraint_id += 1
            
            # Add technical requirements
            for tech_req in expansion.get("technical_requirements", []):
                tech_req_node_id = f"tech_{tech_req_id}"
                intent_graph.add_node(
                    tech_req_node_id,
                    type="technical_requirement",
                    name=tech_req.get("name", "Unknown Requirement"),
                    description=tech_req.get("description", ""),
                    complexity=tech_req.get("complexity", "medium"),
                    importance=tech_req.get("importance", "medium")
                )
                
                # Link tech requirement to objective
                intent_graph.add_edge(tech_req_node_id, objective_id, relationship="enables")
                tech_req_id += 1
            
            # Add assumptions
            for assumption in expansion.get("assumptions", []):
                assumption_text = assumption.get("assumption", "")
                assumption_name = self._extract_meaningful_name(assumption_text, "assumption")
                
                assumption_node_id = f"assumption_{assumption_id}"
                intent_graph.add_node(
                    assumption_node_id,
                    type="assumption",
                    name=assumption_name,
                    assumption=assumption_text,
                    confidence=assumption.get("confidence", 0.5),
                    impact_if_wrong=assumption.get("impact_if_wrong", "medium")
                )
                
                # Link assumption to stakeholder
                intent_graph.add_edge(stakeholder_id, assumption_node_id, relationship="assumes")
                assumption_id += 1
        
        # Apply heuristic-informed intelligent linking
        self._apply_heuristic_linking(intent_graph, expansions, assessment)
        
        print(f"   Created {len(intent_graph.nodes())} nodes, {len(intent_graph.edges())} edges")
        
        # Generate detailed heuristic analysis report
        self._generate_heuristic_report(intent_graph, expansions, assessment)
        
        # Demonstrate meaningful insights with graph queries
        self._demonstrate_graph_insights(intent_graph, assessment)
        
        return intent_graph
    
    def _build_graph_from_node_types(self, user_input: str, assessment: Dict, stakeholders: list, node_type_data: Dict) -> IntentKnowledgeGraph:
        """Build complete intent graph from node-type-specific expansion results"""
        
        print(f"🏗️ Building intent graph from {len(node_type_data)} node types...")
        
        # Debug: Show node type data (concise)
        print(f"\n📋 NODE TYPE DEBUG:")
        for node_type, data in node_type_data.items():
            if isinstance(data, dict):
                actual_node_type = data.get('node_type', node_type.replace('node_type_', ''))
                expansion_data = data.get('expansion', {})
                items = expansion_data.get(actual_node_type, [])
                print(f"   {actual_node_type}: {len(items)} items")
            else:
                print(f"   {node_type}: Invalid data type")
        
        intent_graph = IntentKnowledgeGraph("adaptive", "Node-Type Analysis Graph")
        
        # Add stakeholders and objectives first
        for stakeholder in stakeholders:
            print(f"   Processing stakeholder: {stakeholder}")
            stakeholder_name = stakeholder.get("name", "User")
            stakeholder_role = stakeholder.get("role", "participant")
            stakeholder_id = f"stake_{stakeholder_name.lower().replace(' ', '_')}"
            
            intent_graph.add_stakeholder(
                stakeholder_id,
                name=stakeholder_name,
                role=stakeholder_role
            )
            
            # Add objectives for this stakeholder
            objectives = stakeholder.get("objectives", [])
            print(f"   Objectives for {stakeholder_name}: {objectives}")
            for objective in objectives:
                print(f"   Processing objective: {objective} (type: {type(objective)})")
                if isinstance(objective, dict):
                    objective_name = objective.get("name", "Unknown")
                elif isinstance(objective, str):
                    objective_name = objective
                else:
                    objective_name = str(objective)
                objective_id = f"obj_{objective_name.lower().replace(' ', '_')}_{hash(objective_name) % 1000}"
                
                # Get priority safely
                if isinstance(objective, dict):
                    priority = objective.get("priority", "medium")
                else:
                    priority = "medium"
                
                intent_graph.add_objective(
                    objective_id,
                    name=objective_name,
                    description=f"{stakeholder_name}'s objective: {objective_name}",
                    priority=priority
                )
                
                # Link stakeholder to objective
                intent_graph.add_edge(stakeholder_id, objective_id, relationship="pursues")
        
        # Process each node type
        constraint_id = 1
        tech_req_id = 1
        assumption_id = 1
        
        for node_type, data in node_type_data.items():
            if isinstance(data, dict):
                # Extract the actual node type name and get items from expansion
                actual_node_type = data.get('node_type', node_type.replace('node_type_', ''))
                expansion_data = data.get('expansion', {})
                items = expansion_data.get(actual_node_type, [])
            else:
                items = []
            
            for item in items:
                if actual_node_type == "constraints":
                    constraint_node_id = f"con_{constraint_id}"
                    
                    # Create meaningful constraint name from type and description
                    constraint_type = item.get("constraint_type", "resource")
                    constraint_desc = item.get("constraint", "")
                    provided_name = item.get("name", "")
                    
                    # Generate meaningful name based on available data
                    if provided_name and provided_name != "":
                        constraint_name = provided_name
                    elif constraint_type and constraint_desc:
                        # Create name from type and key details from description
                        if constraint_type == "budget" and "$" in constraint_desc:
                            constraint_name = f"Budget Constraint ({constraint_desc.split('$')[1].split()[0] if '$' in constraint_desc else 'Limited'})"
                        elif constraint_type == "timeline" and ("week" in constraint_desc.lower() or "day" in constraint_desc.lower()):
                            time_ref = "6 weeks" if "6" in constraint_desc else "Limited timeframe"
                            constraint_name = f"Timeline Constraint ({time_ref})"
                        elif constraint_type == "technical":
                            constraint_name = f"Technical Constraint (Integration)"
                        elif constraint_type == "resource":
                            constraint_name = f"Resource Constraint (Staff/Equipment)"
                        else:
                            constraint_name = f"{constraint_type.title()} Constraint"
                    else:
                        constraint_name = f"{constraint_type.title()} Constraint"
                    
                    intent_graph.add_constraint(
                        constraint_node_id,
                        constraint_type=constraint_type,
                        constraint=constraint_desc,
                        name=constraint_name,
                        flexibility=item.get("flexibility", "rigid")
                    )
                    
                    # Link to objectives selectively based on constraint type and relevance
                    heuristic_tags = item.get("heuristic_tags", [])
                    constraint_type = item.get("constraint_type", "resource")
                    
                    # Only link to relevant objectives, not all objectives
                    objective_nodes = [n for n in intent_graph.nodes() if intent_graph.get_node_attributes(n).get('type') == 'primary_objective']
                    
                    # Selective linking based on constraint type
                    linked_count = 0
                    for obj_id in objective_nodes:
                        obj_attrs = intent_graph.get_node_attributes(obj_id)
                        obj_name = obj_attrs.get('name', '').lower()
                        
                        # Only link if there's clear relevance
                        should_link = False
                        if constraint_type == "budget" and any(word in obj_name for word in ["budget", "cost", "financial", "money"]):
                            should_link = True
                        elif constraint_type == "timeline" and any(word in obj_name for word in ["time", "deadline", "schedule", "deliver", "implement"]):
                            should_link = True
                        elif constraint_type == "technical" and any(word in obj_name for word in ["system", "technical", "integration", "compatibility"]):
                            should_link = True
                        elif constraint_type == "resource" and any(word in obj_name for word in ["staff", "training", "resource", "capacity"]):
                            should_link = True
                        elif linked_count < 3:  # Fallback: link to first 3 objectives if no specific matches
                            should_link = True
                        
                        if should_link:
                            if "cascade_potential" in heuristic_tags:
                                intent_graph.add_edge(constraint_node_id, obj_id, relationship="cascade_failure_to")
                            else:
                                intent_graph.add_edge(constraint_node_id, obj_id, relationship="constrains")
                            linked_count += 1
                            
                            # Limit to max 5 links per constraint to prevent over-connection
                            if linked_count >= 5:
                                break
                    
                    constraint_id += 1
                
                elif actual_node_type == "technical_requirements":
                    tech_req_node_id = f"tech_{tech_req_id}"
                    intent_graph.add_node(
                        tech_req_node_id,
                        type="technical_requirement",
                        name=item.get("name", f"Tech Requirement {tech_req_id}"),
                        description=item.get("description", ""),
                        complexity=item.get("complexity", "medium"),
                        importance=item.get("importance", "medium")
                    )
                    
                    # Link to relevant objectives only
                    tech_req_name = item.get("name", "").lower()
                    objective_nodes = [n for n in intent_graph.nodes() if intent_graph.get_node_attributes(n).get('type') == 'primary_objective']
                    
                    linked_count = 0
                    for obj_id in objective_nodes:
                        obj_attrs = intent_graph.get_node_attributes(obj_id)
                        obj_name = obj_attrs.get('name', '').lower()
                        
                        # Only link if there's clear technical relevance
                        should_link = False
                        if "order" in tech_req_name and any(word in obj_name for word in ["order", "handling", "management"]):
                            should_link = True
                        elif "payment" in tech_req_name and any(word in obj_name for word in ["payment", "financial", "processing"]):
                            should_link = True
                        elif "inventory" in tech_req_name and any(word in obj_name for word in ["inventory", "stock", "tracking"]):
                            should_link = True
                        elif "user" in tech_req_name and any(word in obj_name for word in ["staff", "training", "user", "interface"]):
                            should_link = True
                        elif "report" in tech_req_name and any(word in obj_name for word in ["report", "analytics", "monitor"]):
                            should_link = True
                        elif linked_count < 3:  # Fallback: link to first 3 objectives if no specific matches
                            should_link = True
                        
                        if should_link:
                            intent_graph.add_edge(tech_req_node_id, obj_id, relationship="enables")
                            linked_count += 1
                            
                            # Limit to max 4 links per tech requirement
                            if linked_count >= 4:
                                break
                    
                    tech_req_id += 1
                
                elif actual_node_type == "assumptions":
                    assumption_node_id = f"assumption_{assumption_id}"
                    # Use the correct field name from IntentAssumption schema
                    assumption_text = item.get("assumption", "")
                    # Use full assumption text as name instead of truncated version
                    assumption_name = assumption_text if assumption_text else f"Assumption {assumption_id}"
                    
                    intent_graph.add_node(
                        assumption_node_id,
                        type="assumption",
                        name=assumption_name,
                        assumption=assumption_text,
                        confidence=item.get("confidence", 0.5),
                        impact_if_wrong=item.get("impact_if_wrong", "medium")
                    )
                    
                    # Link to stakeholder
                    for stakeholder_id in [n for n in intent_graph.nodes() if intent_graph.get_node_attributes(n).get('type') == 'stakeholder']:
                        intent_graph.add_edge(stakeholder_id, assumption_node_id, relationship="assumes")
                    
                    # Link to objectives based on heuristic tags
                    heuristic_tags = item.get("heuristic_tags", [])
                    if "high_risk" in heuristic_tags:
                        for obj_id in [n for n in intent_graph.nodes() if intent_graph.get_node_attributes(n).get('type') == 'primary_objective']:
                            intent_graph.add_edge(assumption_node_id, obj_id, relationship="high_risk_for")
                    
                    assumption_id += 1
        
        # Apply heuristic-informed intelligent linking
        self._apply_heuristic_linking(intent_graph, node_type_data, assessment)
        
        print(f"   Created {len(intent_graph.nodes())} nodes, {len(intent_graph.edges())} edges")
        
        # Generate detailed heuristic analysis report
        self._generate_heuristic_report(intent_graph, node_type_data, assessment)
        
        # Demonstrate meaningful insights with graph queries
        self._demonstrate_graph_insights(intent_graph, assessment)
        
        return intent_graph
    
    def _extract_meaningful_name(self, text: str, node_type: str) -> str:
        """Extract meaningful name from text content"""
        
        if not text:
            return f"Unknown {node_type.title()}"
        
        # Remove common filler words
        filler_words = {'the', 'user', 'assumes', 'that', 'a', 'an', 'is', 'can', 'be', 'will', 'has', 'have', 'this', 'it'}
        
        # Split and clean words
        words = text.lower().split()
        meaningful_words = [w for w in words if w not in filler_words and len(w) > 2]
        
        # Take first 4-5 meaningful words
        if meaningful_words:
            name = ' '.join(meaningful_words[:4]).title()
            # Limit length
            if len(name) > 40:
                name = name[:37] + "..."
            return name
        else:
            return f"Unknown {node_type.title()}"
    
    def _generate_heuristic_report(self, intent_graph: IntentKnowledgeGraph, expansions: Dict, assessment: Dict):
        """Generate detailed report of heuristic linking results"""
        
        print(f"\n📊 HEURISTIC LINKING REPORT")
        print("=" * 80)
        
        # 1. Node Analysis
        self._report_nodes_by_type(intent_graph)
        
        # 2. Relationship Analysis
        self._report_relationships_by_type(intent_graph)
        
        # 3. Advanced Heuristic Relationships
        self._report_advanced_heuristics(intent_graph)
        
        # 4. System Dynamics Analysis
        self._report_system_dynamics(intent_graph)
        
        print("=" * 80)
    
    def _report_nodes_by_type(self, intent_graph: IntentKnowledgeGraph):
        """Report all nodes organized by type"""
        
        print(f"\n🏗️ NODE INVENTORY")
        print("-" * 40)
        
        # Group nodes by type
        nodes_by_type = {}
        for node_id in intent_graph.nodes():
            node_attrs = intent_graph.get_node_attributes(node_id)
            node_type = node_attrs.get('type', 'unknown')
            node_name = node_attrs.get('name', 'Unknown')
            
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append((node_id, node_name))
        
        # Display by type
        for node_type, nodes in sorted(nodes_by_type.items()):
            print(f"\n📋 {node_type.upper().replace('_', ' ')} ({len(nodes)} nodes):")
            for node_id, node_name in nodes:
                print(f"   • {node_id}: '{node_name}'")
    
    def _report_relationships_by_type(self, intent_graph: IntentKnowledgeGraph):
        """Report all relationships organized by type"""
        
        print(f"\n🔗 RELATIONSHIP INVENTORY")
        print("-" * 40)
        
        # Group relationships by type
        relationships_by_type = {}
        for edge in intent_graph.edges():
            source_id, target_id = edge
            edge_attrs = intent_graph.get_edge_attributes(source_id, target_id)
            relationship = edge_attrs.get('relationship', 'unknown')
            
            source_attrs = intent_graph.get_node_attributes(source_id)
            target_attrs = intent_graph.get_node_attributes(target_id)
            source_name = source_attrs.get('name', source_id)
            target_name = target_attrs.get('name', target_id)
            
            if relationship not in relationships_by_type:
                relationships_by_type[relationship] = []
            relationships_by_type[relationship].append((source_name, target_name))
        
        # Display by relationship type
        for relationship, edges in sorted(relationships_by_type.items()):
            print(f"\n🔗 {relationship.upper().replace('_', ' ')} ({len(edges)} links):")
            for source_name, target_name in edges[:5]:  # Show first 5
                print(f"   • '{source_name}' --{relationship}--> '{target_name}'")
            if len(edges) > 5:
                print(f"   ... and {len(edges) - 5} more")
    
    def _report_advanced_heuristics(self, intent_graph: IntentKnowledgeGraph):
        """Report advanced heuristic relationship patterns"""
        
        print(f"\n🧠 ADVANCED HEURISTIC PATTERNS")
        print("-" * 40)
        
        # Count advanced relationship types
        advanced_patterns = {
            'bottleneck': 0,
            'causal': 0,
            'temporal': 0,
            'risk': 0,
            'cascade': 0,
            'competition': 0,
            'power': 0,
            'feedback': 0,
            'leverage': 0
        }
        
        for edge in intent_graph.edges():
            source_id, target_id = edge
            edge_attrs = intent_graph.get_edge_attributes(source_id, target_id)
            relationship = edge_attrs.get('relationship', '')
            
            # Graph math heuristics
            if 'potential_bottleneck_for' in relationship or 'bottleneck' in relationship:
                advanced_patterns['bottleneck'] += 1
            elif 'temporal_prerequisite_for' in relationship or 'temporal_prerequisite' in relationship:
                advanced_patterns['temporal'] += 1
            elif 'competes_for_budget' in relationship or 'competes_for_timeline' in relationship or 'competes_for' in relationship:
                advanced_patterns['competition'] += 1
            elif 'has_authority_over' in relationship or 'veto_power' in relationship:
                advanced_patterns['power'] += 1
            # LLM semantic heuristics
            elif 'causes_failure' in relationship:
                advanced_patterns['causal'] += 1
            elif 'high_risk' in relationship:
                advanced_patterns['risk'] += 1
            elif 'cascade_failure' in relationship:
                advanced_patterns['cascade'] += 1
            elif 'reinforcing_feedback' in relationship:
                advanced_patterns['feedback'] += 1
            elif 'leverage_point' in relationship:
                advanced_patterns['leverage'] += 1
        
        print(f"🔧 System Bottlenecks: {advanced_patterns['bottleneck']} identified")
        print(f"⚡ Causal Failures: {advanced_patterns['causal']} cause-effect chains")
        print(f"⏰ Temporal Dependencies: {advanced_patterns['temporal']} prerequisite links")
        print(f"🚨 Risk Propagation: {advanced_patterns['risk']} high-risk paths")
        print(f"💥 Cascade Failures: {advanced_patterns['cascade']} cascade effects")
        print(f"💰 Resource Competition: {advanced_patterns['competition']} competing objectives")
        print(f"👑 Power Dynamics: {advanced_patterns['power']} authority relationships")
        print(f"🔄 Feedback Loops: {advanced_patterns['feedback']} reinforcing cycles")
        print(f"🎯 Leverage Points: {advanced_patterns['leverage']} high-impact interventions")
    
    def _report_system_dynamics(self, intent_graph: IntentKnowledgeGraph):
        """Report system dynamics and complexity metrics"""
        
        print(f"\n📈 SYSTEM DYNAMICS ANALYSIS")
        print("-" * 40)
        
        # Calculate system metrics
        total_nodes = len(intent_graph.nodes())
        total_edges = len(intent_graph.edges())
        
        # Node connectivity analysis
        high_connectivity_nodes = []
        for node_id in intent_graph.nodes():
            degree = intent_graph.degree(node_id)
            if degree > 4:  # Highly connected
                node_attrs = intent_graph.get_node_attributes(node_id)
                node_name = node_attrs.get('name', node_id)
                node_type = node_attrs.get('type', 'unknown')
                high_connectivity_nodes.append((node_name, node_type, degree))
        
        # System complexity metrics
        avg_connectivity = (total_edges * 2) / total_nodes if total_nodes > 0 else 0
        density = total_edges / (total_nodes * (total_nodes - 1) / 2) if total_nodes > 1 else 0
        
        print(f"📊 System Complexity:")
        print(f"   • Total Nodes: {total_nodes}")
        print(f"   • Total Relationships: {total_edges}")
        print(f"   • Average Connectivity: {avg_connectivity:.1f} connections per node")
        print(f"   • Network Density: {density:.1%}")
        
        if high_connectivity_nodes:
            print(f"\n🔗 System Hubs (>4 connections):")
            for name, node_type, degree in sorted(high_connectivity_nodes, key=lambda x: x[2], reverse=True)[:5]:
                print(f"   • '{name}' ({node_type}): {degree} connections")
        
        # Identify potential system vulnerabilities
        stakeholder_nodes = [n for n in intent_graph.nodes() 
                           if intent_graph.get_node_attributes(n).get('type') == 'stakeholder']
        constraint_nodes = [n for n in intent_graph.nodes() 
                          if intent_graph.get_node_attributes(n).get('type') == 'constraint']
        
        print(f"\n⚠️ System Vulnerabilities:")
        print(f"   • Single Points of Failure: {len([n for n, _, d in high_connectivity_nodes if d > 6])}")
        print(f"   • Critical Stakeholders: {len(stakeholder_nodes)}")
        print(f"   • Rigid Constraints: {len(constraint_nodes)}")
        
        # System health assessment
        if density > 0.3:
            health = "🔴 OVER-CONNECTED (may be brittle)"
        elif density > 0.1:
            health = "🟡 WELL-CONNECTED (balanced complexity)"
        else:
            health = "🟢 LOOSELY-CONNECTED (resilient but may lack coordination)"
        
        print(f"\n🏥 System Health: {health}")
    
    def _demonstrate_graph_insights(self, intent_graph: IntentKnowledgeGraph, assessment: Dict):
        """Query the graph to demonstrate meaningful insights from heuristic linking"""
        
        print(f"\n🔍 GRAPH INSIGHTS ANALYSIS")
        print("-" * 50)
        
        # 1. Authority Analysis
        authority_insights = self._query_authority_patterns(intent_graph)
        if authority_insights:
            print(f"👑 AUTHORITY PATTERNS:")
            for insight in authority_insights:
                print(f"   {insight}")
        
        # 2. Critical Path Analysis
        critical_paths = self._query_critical_dependencies(intent_graph)
        if critical_paths:
            print(f"\n🚨 CRITICAL DEPENDENCIES:")
            for path in critical_paths:
                print(f"   {path}")
        
        # 3. Constraint Impact Analysis
        constraint_impacts = self._query_constraint_severity(intent_graph)
        if constraint_impacts:
            print(f"\n⚖️ CONSTRAINT SEVERITY:")
            for impact in constraint_impacts:
                print(f"   {impact}")
        
        # 4. Technical Enablement Analysis
        tech_enablement = self._query_technical_criticality(intent_graph)
        if tech_enablement:
            print(f"\n⚙️ TECHNICAL CRITICALITY:")
            for tech in tech_enablement:
                print(f"   {tech}")
        
        # 5. Stakeholder Influence Analysis
        influence_patterns = self._query_stakeholder_influence(intent_graph)
        if influence_patterns:
            print(f"\n👥 STAKEHOLDER INFLUENCE:")
            for pattern in influence_patterns:
                print(f"   {pattern}")
        
        # 6. Domain-Specific Insights
        domain_insights = self._query_domain_patterns(intent_graph)
        if domain_insights:
            print(f"\n🏢 DOMAIN-SPECIFIC PATTERNS:")
            for insight in domain_insights:
                print(f"   {insight}")
    
    def _query_authority_patterns(self, intent_graph: IntentKnowledgeGraph) -> List[str]:
        """Query for authority and decision-making patterns"""
        insights = []
        
        for node_id in intent_graph.nodes():
            node_attrs = intent_graph.get_node_attributes(node_id)
            if node_attrs.get('type') == 'stakeholder':
                stakeholder_name = node_attrs.get('name', 'Unknown')
                
                # Find what this stakeholder has authority over
                authority_targets = []
                for target_id in intent_graph.nodes():
                    if intent_graph.has_edge(node_id, target_id):
                        edge_attrs = intent_graph.get_edge_attributes(node_id, target_id)
                        relationship = edge_attrs.get('relationship', '')
                        
                        if 'authority' in relationship:
                            target_attrs = intent_graph.get_node_attributes(target_id)
                            target_name = target_attrs.get('name', 'Unknown')
                            authority_targets.append(target_name)
                
                if authority_targets:
                    insights.append(f"{stakeholder_name} has authority over: {', '.join(authority_targets)}")
        
        return insights
    
    def _query_critical_dependencies(self, intent_graph: IntentKnowledgeGraph) -> List[str]:
        """Query for critical dependency chains"""
        critical_paths = []
        
        for node_id in intent_graph.nodes():
            node_attrs = intent_graph.get_node_attributes(node_id)
            if node_attrs.get('type') == 'primary_objective':
                objective_name = node_attrs.get('name', 'Unknown')
                
                # Find critical dependencies
                blocking_deps = []
                for source_id in intent_graph.nodes():
                    if intent_graph.has_edge(node_id, source_id):
                        edge_attrs = intent_graph.get_edge_attributes(node_id, source_id)
                        relationship = edge_attrs.get('relationship', '')
                        
                        if 'depends_on' in relationship:
                            source_attrs = intent_graph.get_node_attributes(source_id)
                            source_name = source_attrs.get('name', 'Unknown')
                            blocking_deps.append(f"{source_name} ({relationship})")
                
                if blocking_deps:
                    critical_paths.append(f"'{objective_name}' blocked by: {', '.join(blocking_deps)}")
        
        return critical_paths
    
    def _query_constraint_severity(self, intent_graph: IntentKnowledgeGraph) -> List[str]:
        """Query for constraint severity patterns"""
        severity_analysis = []
        
        for node_id in intent_graph.nodes():
            node_attrs = intent_graph.get_node_attributes(node_id)
            if node_attrs.get('type') == 'constraint':
                constraint_name = node_attrs.get('name', 'Unknown')
                
                # Analyze constraint relationships
                strict_targets = []
                moderate_targets = []
                loose_targets = []
                
                for target_id in intent_graph.nodes():
                    if intent_graph.has_edge(node_id, target_id):
                        edge_attrs = intent_graph.get_edge_attributes(node_id, target_id)
                        relationship = edge_attrs.get('relationship', '')
                        target_attrs = intent_graph.get_node_attributes(target_id)
                        target_name = target_attrs.get('name', 'Unknown')
                        
                        if 'strictly_constrains' in relationship:
                            strict_targets.append(target_name)
                        elif 'moderately_constrains' in relationship:
                            moderate_targets.append(target_name)
                        elif 'loosely_constrains' in relationship:
                            loose_targets.append(target_name)
                
                if strict_targets:
                    severity_analysis.append(f"'{constraint_name}' STRICTLY constrains: {', '.join(strict_targets)}")
                if moderate_targets:
                    severity_analysis.append(f"'{constraint_name}' moderately constrains: {', '.join(moderate_targets)}")
        
        return severity_analysis
    
    def _query_technical_criticality(self, intent_graph: IntentKnowledgeGraph) -> List[str]:
        """Query for technical requirement criticality"""
        tech_analysis = []
        
        for node_id in intent_graph.nodes():
            node_attrs = intent_graph.get_node_attributes(node_id)
            if node_attrs.get('type') == 'technical_requirement':
                tech_name = node_attrs.get('name', 'Unknown')
                
                # Find what this tech requirement enables
                critical_enablement = []
                complex_enablement = []
                
                for target_id in intent_graph.nodes():
                    if intent_graph.has_edge(node_id, target_id):
                        edge_attrs = intent_graph.get_edge_attributes(node_id, target_id)
                        relationship = edge_attrs.get('relationship', '')
                        target_attrs = intent_graph.get_node_attributes(target_id)
                        target_name = target_attrs.get('name', 'Unknown')
                        
                        if 'critically_enables' in relationship:
                            critical_enablement.append(target_name)
                        elif 'complex_enables' in relationship:
                            complex_enablement.append(target_name)
                
                if critical_enablement:
                    tech_analysis.append(f"'{tech_name}' is CRITICAL for: {', '.join(critical_enablement)}")
                if complex_enablement:
                    tech_analysis.append(f"'{tech_name}' has complex enablement for: {', '.join(complex_enablement)}")
        
        return tech_analysis
    
    def _query_stakeholder_influence(self, intent_graph: IntentKnowledgeGraph) -> List[str]:
        """Query for stakeholder influence patterns"""
        influence_patterns = []
        
        for node_id in intent_graph.nodes():
            node_attrs = intent_graph.get_node_attributes(node_id)
            if node_attrs.get('type') == 'stakeholder':
                stakeholder_name = node_attrs.get('name', 'Unknown')
                
                # Count different types of influence
                monitors_closely = []
                coordinates = []
                tracks_progress = []
                
                for target_id in intent_graph.nodes():
                    if intent_graph.has_edge(node_id, target_id):
                        edge_attrs = intent_graph.get_edge_attributes(node_id, target_id)
                        relationship = edge_attrs.get('relationship', '')
                        target_attrs = intent_graph.get_node_attributes(target_id)
                        target_name = target_attrs.get('name', 'Unknown')
                        
                        if 'monitors_closely' in relationship:
                            monitors_closely.append(target_name)
                        elif 'coordinates' in relationship:
                            coordinates.append(target_name)
                        elif 'tracks_progress' in relationship:
                            tracks_progress.append(target_name)
                
                if monitors_closely:
                    influence_patterns.append(f"{stakeholder_name} closely monitors: {', '.join(monitors_closely)}")
                if coordinates:
                    influence_patterns.append(f"{stakeholder_name} coordinates: {', '.join(coordinates)}")
        
        return influence_patterns
    
    def _query_domain_patterns(self, intent_graph: IntentKnowledgeGraph) -> List[str]:
        """Query for domain-specific integration patterns"""
        domain_insights = []
        
        # Look for integration patterns
        for node_id in intent_graph.nodes():
            for target_id in intent_graph.nodes():
                if intent_graph.has_edge(node_id, target_id):
                    edge_attrs = intent_graph.get_edge_attributes(node_id, target_id)
                    relationship = edge_attrs.get('relationship', '')
                    
                    if 'integrates_with' in relationship:
                        source_attrs = intent_graph.get_node_attributes(node_id)
                        target_attrs = intent_graph.get_node_attributes(target_id)
                        source_name = source_attrs.get('name', 'Unknown')
                        target_name = target_attrs.get('name', 'Unknown')
                        domain_insights.append(f"System integration: {source_name} ↔ {target_name}")
                    
                    elif 'enhances_experience' in relationship:
                        source_attrs = intent_graph.get_node_attributes(node_id)
                        target_attrs = intent_graph.get_node_attributes(target_id)
                        source_name = source_attrs.get('name', 'Unknown')
                        target_name = target_attrs.get('name', 'Unknown')
                        domain_insights.append(f"UX enhancement: {source_name} → {target_name}")
        
        return domain_insights
    
    def _apply_heuristic_linking(self, intent_graph: IntentKnowledgeGraph, expansions: Dict, assessment: Dict):
        """Apply advanced heuristic linking based on systems thinking and graph theory"""
        
        print(f"🔗 Applying advanced heuristic linking...")
        print(f"   📊 Processing {len(expansions)} expansion results")
        
        # Debug: Show what expansion data we have
        for key, data in expansions.items():
            expansion = data.get("expansion", {})
            print(f"   📋 {key}: {len(expansion.get('constraints', []))} constraints, {len(expansion.get('assumptions', []))} assumptions, {len(expansion.get('dependencies', []))} dependencies")
        
        # PHASE 1: LLM Semantic Heuristics (from expansion data)
        print(f"   🤖 Phase 1: LLM Semantic Heuristics")
        self._link_risk_propagation(intent_graph, expansions)
        self._link_constraint_cascades(intent_graph, expansions)
        self._link_causal_dependencies(intent_graph, expansions)
        self._identify_feedback_loops(intent_graph, expansions)
        self._link_system_leverage_points(intent_graph, expansions)
        
        # PHASE 2: Graph Mathematical Heuristics (structural analysis)
        print(f"   📊 Phase 2: Graph Mathematical Heuristics")
        self._analyze_graph_bottlenecks(intent_graph)
        self._analyze_temporal_sequences(intent_graph)
        self._analyze_resource_competition(intent_graph)
        self._analyze_power_dynamics(intent_graph)
        
        # PHASE 3: Domain Intelligence
        print(f"   🏢 Phase 3: Domain Intelligence")
        domain = self._extract_domain_from_assessment(assessment)
        self._apply_domain_systems_thinking(intent_graph, domain, expansions)
        
        print(f"   ✅ Applied advanced systems-thinking heuristics")
    
    def _analyze_graph_bottlenecks(self, intent_graph: IntentKnowledgeGraph):
        """Graph math: Identify bottlenecks based on node degree centrality (selective)"""
        
        total_nodes = len(intent_graph.nodes())
        bottlenecks_created = 0
        
        # More conservative threshold to avoid over-connection
        if total_nodes <= 10:
            threshold = 4  # Small graphs: >4 connections
        elif total_nodes <= 30:
            threshold = 6  # Medium graphs: >6 connections  
        else:
            threshold = 8  # Large graphs: >8 connections
        
        print(f"      🔧 Bottleneck threshold: >{threshold} connections for {total_nodes} nodes")
        
        bottleneck_nodes = []
        for node_id in intent_graph.nodes():
            degree = intent_graph.degree(node_id)
            if degree > threshold:
                node_attrs = intent_graph.get_node_attributes(node_id)
                node_type = node_attrs.get('type', 'unknown')
                node_name = node_attrs.get('name', node_id)
                bottleneck_nodes.append((node_id, node_name, node_type, degree))
        
        # Only create relationships for the top 3 bottlenecks to avoid over-connection
        bottleneck_nodes.sort(key=lambda x: x[3], reverse=True)
        for node_id, node_name, node_type, degree in bottleneck_nodes[:3]:
            print(f"      🔧 Top Bottleneck: '{node_name}' ({node_type}) with {degree} connections")
            
            # Only link to 2-3 most relevant objectives, not all
            objectives = [n for n in intent_graph.nodes() 
                         if intent_graph.get_node_attributes(n).get('type') == 'primary_objective'][:3]
            
            for target_id in objectives:
                if target_id != node_id:
                    intent_graph.add_edge(node_id, target_id, relationship="potential_bottleneck_for")
                    bottlenecks_created += 1
        
        print(f"      🔧 Created {bottlenecks_created} selective bottleneck relationships")
    
    def _analyze_temporal_sequences(self, intent_graph: IntentKnowledgeGraph):
        """Graph math: Identify temporal dependencies from dependency relationships"""
        
        temporal_links = 0
        
        # Find nodes with dependency information
        for node_id in intent_graph.nodes():
            node_attrs = intent_graph.get_node_attributes(node_id)
            node_type = node_attrs.get('type')
            
            if node_type == 'technical_requirement':
                # Tech requirements often have temporal dependencies
                tech_name = node_attrs.get('name', '').lower()
                
                # Look for prerequisite patterns
                for other_id in intent_graph.nodes():
                    other_attrs = intent_graph.get_node_attributes(other_id)
                    other_name = other_attrs.get('name', '').lower()
                    
                    # Basic temporal logic: payment processing depends on order handling
                    if 'payment' in tech_name and 'order' in other_name:
                        intent_graph.add_edge(other_id, node_id, relationship="temporal_prerequisite_for")
                        temporal_links += 1
                    elif 'inventory' in tech_name and ('order' in other_name or 'payment' in other_name):
                        intent_graph.add_edge(other_id, node_id, relationship="temporal_prerequisite_for")
                        temporal_links += 1
        
        print(f"      ⏰ Created {temporal_links} temporal prerequisite relationships")
    
    def _analyze_resource_competition(self, intent_graph: IntentKnowledgeGraph):
        """Graph math: Find objectives that share the same constraints (compete for resources)"""
        
        competition_links = 0
        
        # Group objectives by their constraints
        constraint_to_objectives = {}
        
        for edge in intent_graph.edges():
            source_id, target_id = edge
            edge_attrs = intent_graph.get_edge_attributes(source_id, target_id)
            relationship = edge_attrs.get('relationship', '')
            
            if relationship == 'constrains':
                # source_id is constraint, target_id is objective
                source_attrs = intent_graph.get_node_attributes(source_id)
                target_attrs = intent_graph.get_node_attributes(target_id)
                
                if source_attrs.get('type') == 'constraint' and target_attrs.get('type') == 'primary_objective':
                    constraint_type = source_attrs.get('constraint_type', 'unknown')
                    
                    if constraint_type not in constraint_to_objectives:
                        constraint_to_objectives[constraint_type] = []
                    constraint_to_objectives[constraint_type].append(target_id)
        
        # Create selective competition relationships (only for critical constraints)
        for constraint_type, objectives in constraint_to_objectives.items():
            if len(objectives) > 1:
                print(f"      💰 {constraint_type} constraint shared by {len(objectives)} objectives")
                
                # Only create competition links for budget/timeline constraints (most critical)
                if constraint_type in ['budget', 'timeline'] and len(objectives) <= 6:
                    # Limit to max 6 objectives to avoid explosion
                    limited_objectives = objectives[:6]
                    for i, obj1 in enumerate(limited_objectives):
                        for obj2 in limited_objectives[i+1:]:
                            intent_graph.add_edge(obj1, obj2, relationship=f"competes_for_{constraint_type}")
                            competition_links += 1
                elif constraint_type in ['budget', 'timeline']:
                    # For larger sets, only link the first 3 objectives
                    for i in range(min(3, len(objectives))):
                        for j in range(i+1, min(3, len(objectives))):
                            intent_graph.add_edge(objectives[i], objectives[j], relationship=f"competes_for_{constraint_type}")
                            competition_links += 1
        
        print(f"      💰 Created {competition_links} resource competition relationships")
    
    def _analyze_power_dynamics(self, intent_graph: IntentKnowledgeGraph):
        """Graph math: Identify authority relationships based on stakeholder-objective patterns"""
        
        power_links = 0
        
        # Find stakeholders and their objectives
        for node_id in intent_graph.nodes():
            node_attrs = intent_graph.get_node_attributes(node_id)
            if node_attrs.get('type') == 'stakeholder':
                stakeholder_role = node_attrs.get('role', '').lower()
                stakeholder_name = node_attrs.get('name', '')
                
                # Authority patterns based on role
                has_authority = any(word in stakeholder_role for word in ['owner', 'manager', 'director', 'ceo', 'decision'])
                
                if has_authority:
                    print(f"      👑 Authority detected: '{stakeholder_name}' ({stakeholder_role})")
                    
                    # Create authority relationships to all objectives
                    for target_id in intent_graph.nodes():
                        target_attrs = intent_graph.get_node_attributes(target_id)
                        if target_attrs.get('type') == 'primary_objective':
                            intent_graph.add_edge(node_id, target_id, relationship="has_authority_over")
                            power_links += 1
        
        print(f"      👑 Created {power_links} authority relationships")
    
    def _analyze_graph_structure(self, intent_graph: IntentKnowledgeGraph, expansions: Dict):
        """Analyze graph structure to identify critical nodes and bottlenecks"""
        
        # Calculate node importance based on connections
        node_importance = {}
        for node_id in intent_graph.nodes():
            # Simple centrality: count of connections
            connections = intent_graph.degree(node_id)
            node_importance[node_id] = connections
        
        # Identify highly connected nodes (potential bottlenecks)
        high_importance = [n for n, importance in node_importance.items() if importance > 3]
        print(f"   🔧 Found {len(high_importance)} potential bottlenecks (>3 connections)")
        
        # Create bottleneck relationships
        bottlenecks_created = 0
        for bottleneck_node in high_importance:
            bottleneck_attrs = intent_graph.get_node_attributes(bottleneck_node)
            if bottleneck_attrs.get('type') == 'technical_requirement':
                # High-connection tech requirements are system bottlenecks
                for obj_node in intent_graph.nodes():
                    obj_attrs = intent_graph.get_node_attributes(obj_node)
                    if obj_attrs.get('type') == 'primary_objective':
                        intent_graph.add_edge(bottleneck_node, obj_node, relationship="potential_bottleneck_for")
                        bottlenecks_created += 1
        
        print(f"   🔧 Created {bottlenecks_created} bottleneck relationships")
    
    def _link_causal_dependencies(self, intent_graph: IntentKnowledgeGraph, expansions: Dict):
        """Link based on cause-effect relationships from expansion analysis"""
        
        for expansion_key, expansion_data in expansions.items():
            expansion = expansion_data.get("expansion", {})
            objective = expansion_data.get("objective", {})
            
            # Look for causal language in constraints and assumptions
            for constraint in expansion.get("constraints", []):
                constraint_desc = constraint.get("description", "").lower()
                if any(word in constraint_desc for word in ["if", "then", "causes", "leads to", "results in"]):
                    # This constraint has causal implications
                    constraint_nodes = [n for n in intent_graph.nodes() 
                                      if intent_graph.get_node_attributes(n).get('type') == 'constraint']
                    
                    for con_node in constraint_nodes:
                        con_attrs = intent_graph.get_node_attributes(con_node)
                        if constraint.get("type") in con_attrs.get("constraint_type", ""):
                            # Create causal relationship
                            obj_nodes = [n for n in intent_graph.nodes() 
                                       if objective.get("name", "").lower() in intent_graph.get_node_attributes(n).get('name', '').lower()]
                            for obj_node in obj_nodes:
                                intent_graph.add_edge(con_node, obj_node, relationship="causes_failure_of")
    
    def _link_temporal_sequences(self, intent_graph: IntentKnowledgeGraph, expansions: Dict):
        """Link based on temporal dependencies and sequences"""
        
        temporal_keywords = ["before", "after", "first", "then", "prerequisite", "depends on completion"]
        
        for expansion_key, expansion_data in expansions.items():
            expansion = expansion_data.get("expansion", {})
            
            for dependency in expansion.get("dependencies", []):
                depends_on = dependency.get("depends_on", "").lower()
                dep_type = dependency.get("type", "")
                
                if any(keyword in depends_on for keyword in temporal_keywords):
                    # This is a temporal dependency
                    objective = expansion_data.get("objective", {})
                    obj_nodes = [n for n in intent_graph.nodes() 
                               if objective.get("name", "").lower() in intent_graph.get_node_attributes(n).get('name', '').lower()]
                    
                    # Find prerequisite nodes
                    for node_id in intent_graph.nodes():
                        node_attrs = intent_graph.get_node_attributes(node_id)
                        node_name = node_attrs.get('name', '').lower()
                        
                        if any(word in node_name for word in depends_on.split()[:3]):
                            for obj_node in obj_nodes:
                                intent_graph.add_edge(node_id, obj_node, relationship="temporal_prerequisite_for")
    
    def _link_risk_propagation(self, intent_graph: IntentKnowledgeGraph, expansions: Dict):
        """Model how risks propagate through the system using heuristic tags"""
        
        high_risk_count = 0
        total_assumptions = 0
        
        for expansion_key, expansion_data in expansions.items():
            expansion = expansion_data.get("expansion", {})
            
            for assumption in expansion.get("assumptions", []):
                total_assumptions += 1
                confidence = assumption.get("confidence", 1.0)
                impact = assumption.get("impact_if_wrong", "low")
                heuristic_tags = assumption.get("heuristic_tags", [])
                
                print(f"   🚨 Assumption: confidence={confidence}, impact={impact}, tags={heuristic_tags}")
                
                # Use heuristic tags OR fallback to confidence/impact
                is_high_risk = "high_risk" in heuristic_tags or (confidence < 0.7 and impact in ["critical", "high"])
                
                if is_high_risk:
                    high_risk_count += 1
                    # Find the specific assumption node
                    assumption_text = assumption.get("assumption", "")
                    assumption_nodes = [n for n in intent_graph.nodes() 
                                      if intent_graph.get_node_attributes(n).get('type') == 'assumption'
                                      and assumption_text[:20] in intent_graph.get_node_attributes(n).get('assumption', '')]
                    
                    for assumption_node in assumption_nodes:
                        # Risk propagates to connected objectives
                        for obj_node in intent_graph.nodes():
                            obj_attrs = intent_graph.get_node_attributes(obj_node)
                            if obj_attrs.get('type') == 'primary_objective':
                                intent_graph.add_edge(assumption_node, obj_node, relationship="high_risk_for")
        
        print(f"   🚨 Found {high_risk_count}/{total_assumptions} high-risk assumptions")
    
    def _link_constraint_cascades(self, intent_graph: IntentKnowledgeGraph, expansions: Dict):
        """Model how constraint violations cascade through objectives"""
        
        rigid_constraints = []
        
        for expansion_key, expansion_data in expansions.items():
            expansion = expansion_data.get("expansion", {})
            
            for constraint in expansion.get("constraints", []):
                flexibility = constraint.get("flexibility", "flexible")
                impact = constraint.get("impact", "medium")
                
                if flexibility == "rigid" and impact == "high":
                    # Rigid high-impact constraints create cascades
                    constraint_nodes = [n for n in intent_graph.nodes() 
                                      if intent_graph.get_node_attributes(n).get('type') == 'constraint']
                    
                    for con_node in constraint_nodes:
                        # Cascade to multiple objectives
                        objective_nodes = [n for n in intent_graph.nodes() 
                                         if intent_graph.get_node_attributes(n).get('type') == 'primary_objective']
                        
                        for obj_node in objective_nodes[:2]:  # Limit cascade
                            intent_graph.add_edge(con_node, obj_node, relationship="cascade_failure_to")
    
    def _link_resource_competition(self, intent_graph: IntentKnowledgeGraph, expansions: Dict):
        """Model resource competition between objectives"""
        
        budget_objectives = []
        time_objectives = []
        
        for expansion_key, expansion_data in expansions.items():
            expansion = expansion_data.get("expansion", {})
            objective = expansion_data.get("objective", {})
            
            # Check if objective competes for budget or time
            for constraint in expansion.get("constraints", []):
                if constraint.get("type") == "budget":
                    budget_objectives.append(objective)
                elif constraint.get("type") == "timeline":
                    time_objectives.append(objective)
        
        # Create competition relationships
        for i, obj1 in enumerate(budget_objectives):
            for obj2 in budget_objectives[i+1:]:
                obj1_nodes = [n for n in intent_graph.nodes() 
                            if obj1.get("name", "").lower() in intent_graph.get_node_attributes(n).get('name', '').lower()]
                obj2_nodes = [n for n in intent_graph.nodes() 
                            if obj2.get("name", "").lower() in intent_graph.get_node_attributes(n).get('name', '').lower()]
                
                for obj1_node in obj1_nodes:
                    for obj2_node in obj2_nodes:
                        intent_graph.add_edge(obj1_node, obj2_node, relationship="competes_for_budget")
    
    def _link_power_networks(self, intent_graph: IntentKnowledgeGraph, expansions: Dict, assessment: Dict):
        """Model organizational power networks and influence"""
        
        stakeholders = assessment.get("stakeholders", [])
        
        # Identify power relationships
        decision_makers = [s for s in stakeholders if "decision" in s.get("role", "").lower() or "owner" in s.get("name", "").lower()]
        managers = [s for s in stakeholders if "manager" in s.get("role", "").lower()]
        
        # Decision makers have veto power over managers' objectives
        for decision_maker in decision_makers:
            for manager in managers:
                dm_nodes = [n for n in intent_graph.nodes() 
                          if decision_maker.get("name", "").lower() in intent_graph.get_node_attributes(n).get('name', '').lower()]
                mgr_nodes = [n for n in intent_graph.nodes() 
                           if manager.get("name", "").lower() in intent_graph.get_node_attributes(n).get('name', '').lower()]
                
                for dm_node in dm_nodes:
                    for mgr_node in mgr_nodes:
                        intent_graph.add_edge(dm_node, mgr_node, relationship="has_veto_power_over")
    
    def _identify_feedback_loops(self, intent_graph: IntentKnowledgeGraph, expansions: Dict):
        """Identify reinforcing and balancing feedback loops"""
        
        # Look for success metrics that could create reinforcing loops
        for expansion_key, expansion_data in expansions.items():
            expansion = expansion_data.get("expansion", {})
            objective = expansion_data.get("objective", {})
            
            for metric in expansion.get("success_metrics", []):
                metric_text = metric.get("metric", "").lower()
                
                if any(word in metric_text for word in ["revenue", "profit", "growth", "efficiency"]):
                    # Success metrics that could reinforce more investment
                    obj_nodes = [n for n in intent_graph.nodes() 
                               if objective.get("name", "").lower() in intent_graph.get_node_attributes(n).get('name', '').lower()]
                    
                    # Find budget/resource constraints
                    resource_nodes = [n for n in intent_graph.nodes() 
                                    if intent_graph.get_node_attributes(n).get('type') == 'constraint' 
                                    and 'budget' in intent_graph.get_node_attributes(n).get('name', '').lower()]
                    
                    for obj_node in obj_nodes:
                        for resource_node in resource_nodes:
                            intent_graph.add_edge(obj_node, resource_node, relationship="reinforcing_feedback_to")
    
    def _link_system_leverage_points(self, intent_graph: IntentKnowledgeGraph, expansions: Dict):
        """Identify high-leverage intervention points in the system"""
        
        # Assumptions with high confidence and high impact are leverage points
        for expansion_key, expansion_data in expansions.items():
            expansion = expansion_data.get("expansion", {})
            
            for assumption in expansion.get("assumptions", []):
                confidence = assumption.get("confidence", 0.5)
                impact = assumption.get("impact_if_wrong", "medium")
                
                if confidence > 0.8 and impact in ["critical", "high"]:
                    # High-confidence, high-impact assumptions are leverage points
                    assumption_nodes = [n for n in intent_graph.nodes() 
                                      if intent_graph.get_node_attributes(n).get('type') == 'assumption']
                    
                    for assumption_node in assumption_nodes:
                        # This assumption is a system leverage point
                        objective_nodes = [n for n in intent_graph.nodes() 
                                         if intent_graph.get_node_attributes(n).get('type') == 'primary_objective']
                        
                        for obj_node in objective_nodes[:2]:  # Limit connections
                            intent_graph.add_edge(assumption_node, obj_node, relationship="leverage_point_for")
    
    def _extract_domain_from_assessment(self, assessment: Dict) -> str:
        """Extract domain from assessment data"""
        stakeholders = assessment.get("stakeholders", [])
        if stakeholders:
            # Look for domain clues in stakeholder roles or context
            first_stakeholder = stakeholders[0]
            role = first_stakeholder.get("role", "").lower()
            
            if "restaurant" in role or "food" in role:
                return "restaurant"
            elif "mobile" in role or "app" in role or "software" in role:
                return "mobile_app"
            elif "corporate" in role or "company" in role:
                return "corporate"
        
        return "general"
    
    def _apply_domain_systems_thinking(self, intent_graph: IntentKnowledgeGraph, domain: str, expansions: Dict):
        """Apply domain-specific systems thinking patterns"""
        
        if domain == "restaurant":
            # Restaurant systems: customer flow → revenue → reinvestment
            self._link_restaurant_value_chain(intent_graph)
        elif domain == "mobile_app":
            # Mobile app systems: user experience → retention → growth
            self._link_mobile_app_growth_loops(intent_graph)
        elif domain == "corporate":
            # Corporate systems: policy → behavior → culture → performance
            self._link_corporate_culture_loops(intent_graph)
    
    def _link_restaurant_value_chain(self, intent_graph: IntentKnowledgeGraph):
        """Restaurant-specific value chain relationships"""
        
        # POS → Order efficiency → Customer satisfaction → Revenue
        pos_nodes = [n for n in intent_graph.nodes() 
                    if "pos" in intent_graph.get_node_attributes(n).get('name', '').lower()]
        order_nodes = [n for n in intent_graph.nodes() 
                      if "order" in intent_graph.get_node_attributes(n).get('name', '').lower()]
        customer_nodes = [n for n in intent_graph.nodes() 
                         if "customer" in intent_graph.get_node_attributes(n).get('name', '').lower()]
        
        for pos_node in pos_nodes:
            for order_node in order_nodes:
                intent_graph.add_edge(pos_node, order_node, relationship="improves_efficiency_of")
            for customer_node in customer_nodes:
                intent_graph.add_edge(pos_node, customer_node, relationship="enhances_experience_for")
    
    def _link_mobile_app_growth_loops(self, intent_graph: IntentKnowledgeGraph):
        """Mobile app growth loop relationships"""
        
        tracking_nodes = [n for n in intent_graph.nodes() 
                         if "tracking" in intent_graph.get_node_attributes(n).get('name', '').lower()]
        user_nodes = [n for n in intent_graph.nodes() 
                     if "user" in intent_graph.get_node_attributes(n).get('name', '').lower()]
        
        for tracking_node in tracking_nodes:
            for user_node in user_nodes:
                intent_graph.add_edge(tracking_node, user_node, relationship="drives_engagement_for")
    
    def _link_corporate_culture_loops(self, intent_graph: IntentKnowledgeGraph):
        """Corporate culture and policy feedback loops"""
        
        policy_nodes = [n for n in intent_graph.nodes() 
                       if "policy" in intent_graph.get_node_attributes(n).get('name', '').lower()]
        performance_nodes = [n for n in intent_graph.nodes() 
                           if "performance" in intent_graph.get_node_attributes(n).get('name', '').lower()]
        
        for policy_node in policy_nodes:
            for perf_node in performance_nodes:
                intent_graph.add_edge(policy_node, perf_node, relationship="shapes_culture_affecting")
    
    def _link_dependencies(self, intent_graph: IntentKnowledgeGraph, expansions: Dict):
        """Link nodes based on dependencies identified in expansion analysis"""
        
        for expansion_key, expansion_data in expansions.items():
            expansion = expansion_data.get("expansion", {})
            stakeholder = expansion_data.get("stakeholder", {})
            objective = expansion_data.get("objective", {})
            
            stakeholder_name = stakeholder.get("name", "Unknown")
            objective_name = objective.get("name", "Unknown")
            
            # Find the objective node for this expansion
            objective_nodes = [n for n in intent_graph.nodes() 
                             if intent_graph.get_node_attributes(n).get('name', '').lower() in objective_name.lower()]
            
            if not objective_nodes:
                continue
                
            objective_node = objective_nodes[0]
            
            # Process dependencies from expansion
            for dependency in expansion.get("dependencies", []):
                depends_on = dependency.get("depends_on", "")
                dep_type = dependency.get("type", "information")
                criticality = dependency.get("criticality", "important")
                
                # Find nodes that match the dependency
                matching_nodes = []
                for node_id in intent_graph.nodes():
                    node_attrs = intent_graph.get_node_attributes(node_id)
                    node_name = node_attrs.get('name', '').lower()
                    
                    if any(word in node_name for word in depends_on.lower().split()[:3]):
                        matching_nodes.append(node_id)
                
                # Create dependency links
                for dep_node in matching_nodes:
                    relationship = f"depends_on_{dep_type}" if criticality == "blocking" else f"influenced_by_{dep_type}"
                    intent_graph.add_edge(objective_node, dep_node, relationship=relationship)
    
    def _link_constraint_impacts(self, intent_graph: IntentKnowledgeGraph, expansions: Dict):
        """Link constraints to objectives based on impact analysis"""
        
        constraint_nodes = [n for n in intent_graph.nodes() 
                           if intent_graph.get_node_attributes(n).get('type') == 'constraint']
        
        for expansion_key, expansion_data in expansions.items():
            expansion = expansion_data.get("expansion", {})
            objective = expansion_data.get("objective", {})
            objective_name = objective.get("name", "Unknown")
            
            # Find objective node
            objective_nodes = [n for n in intent_graph.nodes() 
                             if intent_graph.get_node_attributes(n).get('name', '').lower() in objective_name.lower()]
            
            if not objective_nodes:
                continue
                
            objective_node = objective_nodes[0]
            
            # Link constraints based on impact level
            for constraint in expansion.get("constraints", []):
                constraint_type = constraint.get("type", "other")
                impact = constraint.get("impact", "medium")
                flexibility = constraint.get("flexibility", "flexible")
                
                # Find matching constraint nodes
                matching_constraints = [n for n in constraint_nodes 
                                      if constraint_type in intent_graph.get_node_attributes(n).get('constraint_type', '')]
                
                for con_node in matching_constraints:
                    if impact == "high" or flexibility == "rigid":
                        relationship = "strictly_constrains"
                    elif impact == "medium":
                        relationship = "moderately_constrains"
                    else:
                        relationship = "loosely_constrains"
                    
                    intent_graph.add_edge(con_node, objective_node, relationship=relationship)
    
    def _link_technical_enablement(self, intent_graph: IntentKnowledgeGraph, expansions: Dict):
        """Link technical requirements to objectives based on enablement analysis"""
        
        tech_nodes = [n for n in intent_graph.nodes() 
                     if intent_graph.get_node_attributes(n).get('type') == 'technical_requirement']
        
        for expansion_key, expansion_data in expansions.items():
            expansion = expansion_data.get("expansion", {})
            objective = expansion_data.get("objective", {})
            objective_name = objective.get("name", "Unknown")
            
            # Find objective node
            objective_nodes = [n for n in intent_graph.nodes() 
                             if intent_graph.get_node_attributes(n).get('name', '').lower() in objective_name.lower()]
            
            if not objective_nodes:
                continue
                
            objective_node = objective_nodes[0]
            
            # Link tech requirements based on importance and complexity
            for tech_req in expansion.get("technical_requirements", []):
                importance = tech_req.get("importance", "medium")
                complexity = tech_req.get("complexity", "medium")
                req_name = tech_req.get("name", "").lower()
                
                # Find matching tech requirement nodes
                matching_tech = [n for n in tech_nodes 
                               if any(word in intent_graph.get_node_attributes(n).get('name', '').lower() 
                                     for word in req_name.split()[:2])]
                
                for tech_node in matching_tech:
                    if importance == "critical":
                        relationship = "critically_enables"
                    elif complexity == "high":
                        relationship = "complex_enables"
                    else:
                        relationship = "enables"
                    
                    intent_graph.add_edge(tech_node, objective_node, relationship=relationship)
    
    def _link_stakeholder_influence(self, intent_graph: IntentKnowledgeGraph, expansions: Dict, assessment: Dict):
        """Link stakeholders based on influence patterns from assessment"""
        
        stakeholder_nodes = [n for n in intent_graph.nodes() 
                           if intent_graph.get_node_attributes(n).get('type') == 'stakeholder']
        
        # Get stakeholder info from assessment
        assessment_stakeholders = assessment.get("stakeholders", [])
        
        for stakeholder_info in assessment_stakeholders:
            stakeholder_name = stakeholder_info.get("name", "")
            stakeholder_role = stakeholder_info.get("role", "")
            
            # Find stakeholder node
            stakeholder_nodes_match = [n for n in stakeholder_nodes 
                                     if stakeholder_name.lower() in intent_graph.get_node_attributes(n).get('name', '').lower()]
            
            if not stakeholder_nodes_match:
                continue
                
            stakeholder_node = stakeholder_nodes_match[0]
            
            # Create role-based influence links
            if "owner" in stakeholder_role.lower() or "decision" in stakeholder_role.lower():
                # Owners have authority over constraints and high-priority objectives
                constraint_nodes = [n for n in intent_graph.nodes() 
                                  if intent_graph.get_node_attributes(n).get('type') == 'constraint']
                for con_node in constraint_nodes:
                    intent_graph.add_edge(stakeholder_node, con_node, relationship="has_authority_over")
                    
            elif "manager" in stakeholder_role.lower():
                # Managers coordinate between objectives and technical requirements
                tech_nodes = [n for n in intent_graph.nodes() 
                            if intent_graph.get_node_attributes(n).get('type') == 'technical_requirement']
                for tech_node in tech_nodes[:3]:  # Link to first few tech requirements
                    intent_graph.add_edge(stakeholder_node, tech_node, relationship="coordinates")
    
    def _link_success_metrics(self, intent_graph: IntentKnowledgeGraph, expansions: Dict):
        """Link success metrics to objectives and stakeholders"""
        
        for expansion_key, expansion_data in expansions.items():
            expansion = expansion_data.get("expansion", {})
            stakeholder = expansion_data.get("stakeholder", {})
            objective = expansion_data.get("objective", {})
            
            stakeholder_name = stakeholder.get("name", "Unknown")
            objective_name = objective.get("name", "Unknown")
            
            # Find nodes
            stakeholder_nodes = [n for n in intent_graph.nodes() 
                               if stakeholder_name.lower() in intent_graph.get_node_attributes(n).get('name', '').lower()]
            objective_nodes = [n for n in intent_graph.nodes() 
                             if objective_name.lower() in intent_graph.get_node_attributes(n).get('name', '').lower()]
            
            if not (stakeholder_nodes and objective_nodes):
                continue
                
            stakeholder_node = stakeholder_nodes[0]
            objective_node = objective_nodes[0]
            
            # Process success metrics
            for metric in expansion.get("success_metrics", []):
                timeframe = metric.get("timeframe", "")
                
                if "immediate" in timeframe.lower() or "short" in timeframe.lower():
                    intent_graph.add_edge(stakeholder_node, objective_node, relationship="monitors_closely")
                else:
                    intent_graph.add_edge(stakeholder_node, objective_node, relationship="tracks_progress")
    
    def _link_domain_patterns(self, intent_graph: IntentKnowledgeGraph, domain: str):
        """Apply domain-specific linking patterns"""
        
        if "restaurant" in domain.lower() or "food" in domain.lower():
            # Restaurant domain: link payment systems to inventory systems
            payment_nodes = [n for n in intent_graph.nodes() 
                           if "payment" in intent_graph.get_node_attributes(n).get('name', '').lower()]
            inventory_nodes = [n for n in intent_graph.nodes() 
                             if "inventory" in intent_graph.get_node_attributes(n).get('name', '').lower()]
            
            for payment_node in payment_nodes:
                for inventory_node in inventory_nodes:
                    intent_graph.add_edge(payment_node, inventory_node, relationship="integrates_with")
                    
        elif "mobile" in domain.lower() or "app" in domain.lower():
            # Mobile domain: link tracking to user experience
            tracking_nodes = [n for n in intent_graph.nodes() 
                            if "tracking" in intent_graph.get_node_attributes(n).get('name', '').lower()]
            user_nodes = [n for n in intent_graph.nodes() 
                        if "user" in intent_graph.get_node_attributes(n).get('name', '').lower() or 
                           "customer" in intent_graph.get_node_attributes(n).get('name', '').lower()]
            
            for tracking_node in tracking_nodes:
                for user_node in user_nodes:
                    intent_graph.add_edge(tracking_node, user_node, relationship="enhances_experience")
    
    def _call_llm_sync(self, prompt: str, content_type: str, response_model=None) -> Dict:
        """Synchronous LLM call for use in ZMQ handlers"""
        
        try:
            if response_model:
                # Use structured output for both assessment and node-type expansions
                if content_type == "complexity_assessment":
                    system_message = "You are an expert business analyst. Analyze complexity and return structured assessment."
                else:
                    system_message = "You are an expert business analyst. Analyze the specific node type and return structured data."
                
                response = self.llm_client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=response_model,
                    temperature=0.3,
                    max_tokens=2000
                )
                
                return response.choices[0].message.parsed.model_dump()
            else:
                # Regular JSON response for node-type expansions
                response = self.llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert business analyst. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                
                content = response.choices[0].message.content.strip()
                
                # Parse JSON response
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error for {content_type}: {str(e)}")
                    print(f"Raw content: {content[:200]}...")
                    return {}
                
        except Exception as e:
            print(f"❌ LLM call failed for {content_type}: {str(e)}")
            return {}


# ========================================
# Usage Example
# ========================================

async def example_adaptive_analysis():
    """Example of adaptive intent analysis"""
    
    runner = AdaptiveIntentRunner(base_port=6000)
    
    # Test cases
    test_cases = [
        {
            "name": "COMPLEX REQUEST", 
            "input": "I need a new POS system for my restaurant that handles orders, payments, and inventory with a $25k budget in 6 weeks",
            "expected": "complex"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}: {test_case['expected'].upper()} REQUEST")
        
        result_type, result = await runner.analyze_intent(
            test_case["input"]
        )
        
        print(f"\n🎯 RESULT: {result_type}")
        
        if result_type == "direct_answer":
            print(f"   Answer: {result}")
        elif result_type == "complex_analysis":
            # Handle new dict format with graph, tasks, and qaqc
            if isinstance(result, dict):
                graph = result.get("graph")
                tasks = result.get("tasks", {})
                qaqc = result.get("qaqc", {})
                
                if graph:
                    print(f"   Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
                if tasks.get("task_breakdown"):
                    print(f"   Tasks: {len(tasks['task_breakdown'])} actionable tasks")
                if qaqc:
                    quality_score = qaqc.get("overall_quality_score", "N/A")
                    validation_passed = qaqc.get("validation_passed", False)
                    print(f"   QAQC: {quality_score:.2f}/1.0 quality, {'PASSED' if validation_passed else 'NEEDS REVIEW'}")
            else:
                # Fallback for old format
                print(f"   Graph: {len(result.nodes())} nodes, {len(result.edges())} edges")

async def main():
    """Run the original test cases"""
    
    print("Adaptive Intent Runner - Heuristic LLM Pattern")
    print("Initializing ZeroMQ Engine..")
    
    await example_adaptive_analysis()

async def test_restaurant_pos_detailed():
    """Test with detailed restaurant POS scenario to see heuristic analysis"""
    
    print("=" * 60)
    print("TEST CASE: DETAILED RESTAURANT POS ANALYSIS")
    print("=" * 60)
    
    runner = AdaptiveIntentRunner()
    
    # More detailed POS request to trigger rich analysis
    detailed_request = """I need to implement a comprehensive POS system for my restaurant. 
    The system must handle order management, payment processing, inventory tracking, and staff management. 
    I have a $25,000 budget and need it operational within 6 weeks. 
    The restaurant serves 200+ customers daily and has 15 staff members. 
    We need integration with our existing accounting system and supplier networks."""
    
    result_type, result_data = await runner.analyze_intent(detailed_request)
    
    print(f"\n🎯 RESULT: {result_type}")
    if result_type == "complex_analysis":
        print(f"   Graph: {len(result_data.nodes())} nodes, {len(result_data.edges())} edges")
    
    return result_type, result_data

if __name__ == "__main__":
    # Run main test only (detailed test commented out to avoid duplication)
    asyncio.run(main())
    # print("\n" + "="*80)
    # asyncio.run(test_restaurant_pos_detailed())
