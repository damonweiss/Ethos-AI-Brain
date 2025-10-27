#!/usr/bin/env python3
"""
AI_Agent - Simplified with core functionality in base class
All intelligence in base, specialization through role prompts only
"""

import asyncio
import logging
import uuid
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from typing import Dict, Any, List, Optional

from ai_brain import AI_Brain, CognitiveCapability
from command_structures import IntakeAssessment, ExpertConsultation

logger = logging.getLogger(__name__)

class AgentStatus:
    INITIALIZING = "initializing"
    READY = "ready"
    THINKING = "thinking"
    EXECUTING = "executing"
    CONSULTING = "consulting"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class AI_Agent:
    """
    Base AI Agent with full intelligence
    All core functionality here, specialization through role prompts
    """
    
    def __init__(self, agent_id: str, agent_role: str, role_specialization: str = None):
        self.agent_id = agent_id
        self.agent_role = agent_role
        self.status = AgentStatus.INITIALIZING
        
        # Every agent gets full intelligence
        self.brain = AI_Brain([
            CognitiveCapability.META_REASONING,
            CognitiveCapability.STRATEGIC_ANALYSIS,
            CognitiveCapability.TACTICAL_ANALYSIS,
            CognitiveCapability.EXPERT_CONSULTATION,
            CognitiveCapability.SITUATION_ASSESSMENT,
            CognitiveCapability.MEMORY_ACCESS,
            CognitiveCapability.LEARNING,
            CognitiveCapability.PATTERN_RECOGNITION,
            CognitiveCapability.DECISION_MAKING
        ])
        
        # Set role specialization
        if role_specialization:
            self.brain.set_specialization_context(role_specialization)
        
        # Agent state
        self.current_mission = None
        self.mission_history = []
        self.performance_metrics = {}
        
        self.status = AgentStatus.READY
        logger.info(f"🤖 AI_Agent {agent_id} ({agent_role}) initialized with full intelligence")
    
    async def execute_mission(self, mission: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Core mission execution - same for all agents"""
        self.status = AgentStatus.EXECUTING
        self.current_mission = mission
        
        try:
            # STEP 1: Brain does enhanced intake assessment (cognitive)
            intake_result = await self.brain.enhanced_intake(mission)
            
            print(f"🧠 BRAIN INTAKE COMPLETE:")
            print(f"   Strategy: {intake_result['strategy']}")
            print(f"   Reasoning: {intake_result['raw_response']}")
            
            # STEP 2: Agent interprets through role lens and executes
            if intake_result['strategy'] == 'direct_answer':
                print(f"\n✅ {self.agent_role.upper()} DIRECT RESPONSE:")
                
                # Extract answer from parameters with fallback
                answer = intake_result['parameters'].get('answer')
                if not answer:
                    # Fallback: try to extract answer from reasoning
                    answer = f"Based on my knowledge: {intake_result.get('reasoning', 'Answer not provided')}"
                
                print(f"   Answer: {answer}")
                
                result = {
                    "agent_role": self.agent_role,
                    "strategy": "direct_answer",
                    "answer": answer,
                    "success": True
                }
                
            elif intake_result['strategy'] == 'expert_consultation':
                print(f"\n👥 {self.agent_role.upper()} DEPLOYING EXPERT PANEL:")
                
                # Extract ready-to-use expert prompts from brain's assessment
                experts = intake_result['parameters'].get('experts', [])
                consultation_question = intake_result['parameters'].get('consultation_question', mission)
                
                print(f"   Experts: {len(experts)} ready-to-deploy specialists")
                for expert in experts:
                    print(f"      {expert['role']} ({expert['domain']}) - {expert['consultation_focus']}")
                
                # Execute expert consultation immediately
                expert_responses = await self.execute_expert_consultation(experts, consultation_question)
                
                result = {
                    "agent_role": self.agent_role,
                    "strategy": "expert_consultation_complete",
                    "expert_panel": experts,
                    "expert_responses": expert_responses,
                    "next_step": "Synthesize expert recommendations",
                    "success": True
                }
                
            elif intake_result['strategy'] == 'hitl_clarification':
                print(f"\n👤 {self.agent_role.upper()} REQUESTING CLARIFICATION:")
                clarification = await self.request_clarification(mission, intake_result)
                print(f"   Questions: {clarification['questions']}")
                
                result = {
                    "agent_role": self.agent_role,
                    "strategy": "clarification_requested",
                    "clarification": clarification,
                    "next_step": "Await user response",
                    "success": True
                }
            elif intake_result['strategy'] == 'task_decomposition':
                print(f"\n📋 {self.agent_role.upper()} TASK DECOMPOSITION:")
                
                decomposition_type = intake_result['parameters'].get('decomposition_type', 'simple')
                expert_guidance = intake_result['parameters'].get('expert_guidance', False)
                
                print(f"   Type: {decomposition_type}")
                
                if expert_guidance:
                    # Expert-guided decomposition
                    experts = intake_result['parameters'].get('experts', [])
                    print(f"   Expert Guidance: {len(experts)} specialists")
                    
                    # First get expert guidance on decomposition
                    decomposition_guidance = await self.get_expert_decomposition_guidance(mission, experts)
                    
                    # Then execute the decomposition
                    subtasks = await self.execute_guided_decomposition(mission, decomposition_guidance)
                    
                    result = {
                        "agent_role": self.agent_role,
                        "strategy": "expert_guided_decomposition",
                        "decomposition_type": decomposition_type,
                        "expert_guidance": decomposition_guidance,
                        "subtasks": subtasks,
                        "next_step": "Execute subtasks",
                        "success": True
                    }
                else:
                    # Simple decomposition
                    subtasks = intake_result['parameters'].get('subtasks', [])
                    if not subtasks:
                        # Generate subtasks if not provided
                        subtasks = await self.execute_simple_decomposition(mission)
                    
                    print(f"   Subtasks: {len(subtasks)}")
                    for i, subtask in enumerate(subtasks, 1):
                        print(f"      {i}. {subtask}")
                    
                    result = {
                        "agent_role": self.agent_role,
                        "strategy": "simple_decomposition",
                        "decomposition_type": decomposition_type,
                        "subtasks": subtasks,
                        "next_step": "Execute subtasks sequentially",
                        "success": True
                    }
                
            elif intake_result['strategy'] == 'multi_stage_strategy':
                print(f"\n🎯 {self.agent_role.upper()} DAG-BASED OPERATION:")
                operation_plan = await self.plan_multi_stage_operation(mission, intake_result)
                
                print(f"   📊 DAG Analysis:")
                print(f"      Nodes: {operation_plan['total_nodes']}")
                print(f"      Execution Levels: {operation_plan['execution_plan']['total_levels']}")
                print(f"      Max Parallelism: {operation_plan['execution_plan']['max_parallelism']}")
                print(f"      Critical Path Length: {operation_plan['execution_plan']['critical_path_length']}")
                
                print(f"\n   📋 Node Details:")
                for node in operation_plan['nodes']:
                    print(f"      {node['id']}: {node['approach']} - {node['purpose']}")
                    print(f"         Dependencies: {node.get('dependencies', [])}, Resources: {node.get('resources', 'N/A')}")
                
                print(f"\n   📐 Topological Order:")
                print(f"      {' → '.join(operation_plan['execution_plan']['topological_order'])}")
                
                print(f"\n   🔄 Execution Plan:")
                for level_idx, level in enumerate(operation_plan['execution_plan']['execution_levels'], 1):
                    if len(level) == 1:
                        print(f"      Level {level_idx}: {level[0]} (sequential)")
                    else:
                        print(f"      Level {level_idx}: {', '.join(level)} (parallel)")
                
                print(f"\n   🎯 Critical Path: {' → '.join(operation_plan['execution_plan']['critical_path_nodes'])}")
                
                # Generate visual DAG
                print(f"\n   📊 Generating DAG Visualization...")
                self.visualize_dag(operation_plan)  # Show directly, don't save to file
                
                result = {
                    "agent_role": self.agent_role,
                    "strategy": "dag_operation",
                    "operation_plan": operation_plan,
                    "next_step": f"Execute Level 1: {', '.join(operation_plan['execution_plan']['execution_levels'][0]) if operation_plan['execution_plan']['execution_levels'] else 'No levels'}",
                    "success": True
                }
                
            else:
                # Other strategies (tool, decomposition, research)
                print(f"\n📋 {self.agent_role.upper()} STRATEGY: {intake_result['strategy'].upper()}")
                print(f"   Next Step: Implement {intake_result['strategy']}")
                
                result = {
                    "agent_role": self.agent_role,
                    "strategy": intake_result['strategy'],
                    "intake_result": intake_result,
                    "next_step": f"Execute {intake_result['strategy']}",
                    "success": True
                }
            
            # COMMENTED OUT - BUILD ONE STEP AT A TIME
            # # All agents can think intelligently about their mission
            # analysis = await self.brain.think(mission, context)
            # 
            # # All agents can consult experts if needed
            # expert_responses = None
            # if len(assessment.required_specialists) > 1:
            #     consultation = ExpertConsultation(
            #         consultation_id=f"consult_{uuid.uuid4().hex[:8]}",
            #         mission_objective=mission,
            #         specific_question=f"Provide analysis for {self.agent_role}",
            #         required_experts=assessment.required_specialists,
            #         intake_assessment=assessment
            #     )
            #     expert_responses = await self.brain.consult_experts(consultation)
            # 
            # # Create result
            # result = {
            #     "agent_role": self.agent_role,
            #     "strategy": f"{self.agent_role}_intelligent_execution",
            #     "assessment": assessment.__dict__,
            #     "analysis": analysis,
            #     "expert_consultation": expert_responses is not None,
            #     "success": True
            # }
            
            # Store mission history
            self.mission_history.append({
                'mission': mission,
                'result': result,
                'timestamp': datetime.now()
            })
            
            # Learn from experience
            self.brain.learn_from_experience({'mission_result': result})
            
            # Update performance metrics
            if 'mission_success_rate' not in self.performance_metrics:
                self.performance_metrics['mission_success_rate'] = []
            self.performance_metrics['mission_success_rate'].append(True)
            
            self.status = AgentStatus.READY
            self.current_mission = None
            
            return result
            
        except Exception as e:
            error_result = {"error": str(e), "success": False}
            self.brain.learn_from_experience({'mission_result': error_result})
            self.performance_metrics.setdefault('mission_success_rate', []).append(False)
            self.status = AgentStatus.ERROR
            raise
    
    async def think_about(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Use brain to think - same for all agents"""
        self.status = AgentStatus.THINKING
        try:
            result = await self.brain.think(prompt, context)
            self.status = AgentStatus.READY
            return result
        except Exception as e:
            self.status = AgentStatus.ERROR
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            'agent_id': self.agent_id,
            'agent_role': self.agent_role,
            'status': self.status,
            'current_mission': self.current_mission,
            'missions_completed': len(self.mission_history),
            'brain_status': self.brain.get_cognitive_status(),
            'performance_metrics': self.performance_metrics
        }
    
    async def generate_expert_panel(self, mission: str, intake_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate role-specific expert panel based on agent's domain expertise
        This is agent-specific, not brain function
        """
        expert_generation_prompt = f"""
You are {self.agent_role} assembling an expert panel for this mission.

MISSION: "{mission}"
BRAIN ASSESSMENT: {intake_result['raw_response'][:300]}

As a {self.agent_role}, generate 3-4 experts you would realistically assemble for this mission.
Consider your authority, domain expertise, and available resources.

For each expert, provide:
EXPERT_1:
- Role: [specific title/rank]
- Expertise: [domain specialization]
- System_Prompt: [LLM prompt to make this expert persona]
- Authority: [what decisions they can make]

EXPERT_2:
- Role: [specific title/rank]
- Expertise: [domain specialization] 
- System_Prompt: [LLM prompt to make this expert persona]
- Authority: [what decisions they can make]

Continue for 3-4 experts total.

Make this realistic for your role as {self.agent_role}.
"""
        
        expert_response = await self.brain.meta_reasoning.llm_backends["analyst"].complete(
            expert_generation_prompt, {"mission": mission, "role": self.agent_role}
        )
        
        # Parse expert panel (simplified for now)
        expert_panel = []
        lines = expert_response.split('\n')
        current_expert = {}
        
        for line in lines:
            if line.startswith('EXPERT_'):
                if current_expert:
                    expert_panel.append(current_expert)
                current_expert = {}
            elif '- Role:' in line:
                current_expert['role'] = line.split('- Role:', 1)[1].strip()
            elif '- Expertise:' in line:
                current_expert['expertise'] = line.split('- Expertise:', 1)[1].strip()
            elif '- System_Prompt:' in line:
                current_expert['system_prompt'] = line.split('- System_Prompt:', 1)[1].strip()
            elif '- Authority:' in line:
                current_expert['authority'] = line.split('- Authority:', 1)[1].strip()
        
        if current_expert:
            expert_panel.append(current_expert)
        
        return expert_panel
    
    async def request_clarification(self, mission: str, intake_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Request clarification from user based on agent's perspective
        """
        clarification_prompt = f"""
You are {self.agent_role} and need clarification on this mission.

MISSION: "{mission}"
BRAIN ASSESSMENT: {intake_result['raw_response'][:300]}

As a {self.agent_role}, what specific questions do you need answered to proceed?
Consider your role, authority, and operational requirements.

Provide 2-3 specific questions in this format:
QUESTION_1: [specific question]
QUESTION_2: [specific question]
QUESTION_3: [specific question]

CONTEXT_NEEDED: [what context is missing for your role]
"""
        
        clarification_response = await self.brain.meta_reasoning.llm_backends["analyst"].complete(
            clarification_prompt, {"mission": mission, "role": self.agent_role}
        )
        
        # Parse questions
        questions = []
        lines = clarification_response.split('\n')
        for line in lines:
            if line.startswith('QUESTION_'):
                question = line.split(':', 1)[1].strip()
                questions.append(question)
        
        context_needed = ""
        for line in lines:
            if line.startswith('CONTEXT_NEEDED:'):
                context_needed = line.split(':', 1)[1].strip()
                break
        
        return {
            "questions": questions,
            "context_needed": context_needed,
            "requesting_agent": self.agent_role
        }
    
    async def execute_expert_consultation(self, experts: List[Dict[str, Any]], consultation_question: str) -> List[Dict[str, Any]]:
        """
        Execute expert consultation using ready-to-deploy expert prompts
        """
        expert_responses = []
        
        for expert in experts:
            print(f"   🔍 Consulting {expert['role']}...")
            
            # Use the expert's system prompt directly
            expert_prompt = f"""
{expert['system_prompt']}

CONSULTATION QUESTION: "{consultation_question}"

FOCUS AREA: {expert['consultation_focus']}

AUTHORITY: {expert['authority']}

Provide your expert analysis and recommendations based on your specialization.
Be specific and actionable in your response.
"""
            
            # Get expert response
            expert_response = await self.brain.meta_reasoning.llm_backends["analyst"].complete(
                expert_prompt, {"question": consultation_question, "expert_role": expert['role']}
            )
            
            expert_responses.append({
                "expert": expert,
                "response": expert_response,
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"      ✅ {expert['role']} consultation complete")
        
        return expert_responses
    
    async def execute_simple_decomposition(self, mission: str) -> List[str]:
        """
        Execute simple task decomposition without expert guidance
        """
        decomposition_prompt = f"""
You are {self.agent_role} breaking down this mission into clear, actionable subtasks.

MISSION: "{mission}"

Break this down into 3-5 specific, actionable subtasks that can be executed independently.
Focus on logical sequence and clear deliverables.

Provide each subtask as a single line starting with "SUBTASK:"

SUBTASK: [first subtask]
SUBTASK: [second subtask]
SUBTASK: [third subtask]
...

Make this realistic for your role as {self.agent_role}.
"""
        
        response = await self.brain.meta_reasoning.llm_backends["analyst"].complete(
            decomposition_prompt, {"mission": mission, "role": self.agent_role}
        )
        
        # Parse subtasks
        subtasks = []
        for line in response.split('\n'):
            if line.startswith('SUBTASK:'):
                subtask = line.split('SUBTASK:', 1)[1].strip()
                if subtask:
                    subtasks.append(subtask)
        
        return subtasks
    
    async def get_expert_decomposition_guidance(self, mission: str, experts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get expert guidance on how to decompose the task
        """
        guidance = []
        
        for expert in experts:
            guidance_prompt = f"""
{expert['system_prompt']}

MISSION TO DECOMPOSE: "{mission}"

FOCUS: {expert['consultation_focus']}

As an expert in your domain, provide guidance on how to break down this mission.
What are the key components, dependencies, and considerations from your expertise area?

Provide specific recommendations for decomposition based on your domain knowledge.
"""
            
            expert_guidance = await self.brain.meta_reasoning.llm_backends["analyst"].complete(
                guidance_prompt, {"mission": mission, "expert_role": expert['role']}
            )
            
            guidance.append({
                "expert": expert,
                "guidance": expert_guidance
            })
        
        return guidance
    
    async def execute_guided_decomposition(self, mission: str, guidance: List[Dict[str, Any]]) -> List[str]:
        """
        Execute decomposition based on expert guidance
        """
        # Synthesize expert guidance into subtasks
        synthesis_prompt = f"""
You are {self.agent_role} synthesizing expert guidance to decompose this mission.

MISSION: "{mission}"

EXPERT GUIDANCE:
"""
        
        for g in guidance:
            synthesis_prompt += f"\n{g['expert']['role']}: {g['guidance'][:300]}...\n"
        
        synthesis_prompt += f"""

Based on this expert guidance, break down the mission into 4-6 specific, actionable subtasks.
Consider the dependencies and domain-specific requirements identified by the experts.

Provide each subtask as a single line starting with "SUBTASK:"

SUBTASK: [first subtask]
SUBTASK: [second subtask]
...

Make this realistic for your role as {self.agent_role}.
"""
        
        response = await self.brain.meta_reasoning.llm_backends["analyst"].complete(
            synthesis_prompt, {"mission": mission, "role": self.agent_role}
        )
        
        # Parse subtasks
        subtasks = []
        for line in response.split('\n'):
            if line.startswith('SUBTASK:'):
                subtask = line.split('SUBTASK:', 1)[1].strip()
                if subtask:
                    subtasks.append(subtask)
        
        return subtasks
    
    async def plan_multi_stage_operation(self, mission: str, intake_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan multi-stage operation using DAG for dependency management and parallel execution
        """
        # Import schema for JSON prompting
        from dag_models import get_dag_plan_schema
        
        dag_planning_prompt = f"""
You are {self.agent_role} planning a multi-stage operation using DAG (Directed Acyclic Graph) methodology.

MISSION: "{mission}"
BRAIN ASSESSMENT: {intake_result['reasoning']}

Create an operational DAG with nodes and dependencies for AI reasoning tasks.

RESPOND WITH VALID JSON using this exact schema:
{get_dag_plan_schema()}

IMPORTANT GUIDELINES:
- These are AI reasoning tasks that execute instantly
- Focus on logical dependencies, not time
- Use approach types: decomposition, expert_consultation, hitl_clarification, research, synthesis
- Dependencies should be node IDs that must complete first
- Parallel_with indicates nodes that can run simultaneously

Example response:
{{
  "nodes": [
    {{
      "id": "assess_current",
      "approach": "hitl_clarification",
      "purpose": "Understand current company structure and challenges",
      "resources": "User interviews, data collection LLM",
      "dependencies": [],
      "parallel_with": [],
      "success_criteria": "Complete understanding of current state"
    }},
    {{
      "id": "research_best_practices",
      "approach": "research",
      "purpose": "Study industry restructuring best practices",
      "resources": "Research LLM, industry databases",
      "dependencies": ["assess_current"],
      "parallel_with": ["consult_experts"],
      "success_criteria": "Comprehensive best practices report"
    }}
  ],
  "critical_path": "assess_current → research_best_practices → synthesize_plan",
  "parallel_opportunities": "research_best_practices and consult_experts can run in parallel",
  "coordination_strategy": "Coordinate parallel tasks through shared data structures",
  "risk_mitigation": "Fallback strategies for each node failure",
  "commanding_officer": "{self.agent_role}"
}}

RESPOND ONLY WITH VALID JSON:
"""
        
        planning_response = await self.brain.meta_reasoning.llm_backends["analyst"].complete(
            dag_planning_prompt, {"mission": mission, "role": self.agent_role}
        )
        
        # Parse JSON response using Pydantic
        from dag_models import validate_dag_response
        
        try:
            # Validate and parse the JSON response
            dag_plan = validate_dag_response(planning_response)
            
            # Convert to NetworkX DAG
            dag = dag_plan.to_networkx()
            
            # Validate DAG (no cycles)
            if not nx.is_directed_acyclic_graph(dag):
                logger.warning("Generated graph has cycles - converting to DAG")
                dag = nx.DiGraph([(u, v) for u, v in dag.edges() if not nx.has_path(dag, v, u)])
            
            # Calculate execution plan
            execution_plan = self._calculate_execution_plan(dag)
            
            return {
                "dag": dag,
                "nodes": [node.model_dump() for node in dag_plan.nodes],
                "execution_plan": execution_plan,
                "critical_path": dag_plan.critical_path,
                "parallel_opportunities": dag_plan.parallel_opportunities,
                "coordination_strategy": dag_plan.coordination_strategy,
                "risk_mitigation": dag_plan.risk_mitigation,
                "commanding_officer": dag_plan.commanding_officer,
                "total_nodes": len(dag_plan.nodes)
            }
            
        except ValueError as e:
            logger.warning(f"Failed to parse JSON DAG response: {e}")
            # Fallback to simple error response
            return {
                "dag": nx.DiGraph(),
                "nodes": [],
                "execution_plan": {"execution_levels": [], "critical_path_nodes": [], "total_levels": 0, "max_parallelism": 0},
                "critical_path": "Parse error",
                "parallel_opportunities": "Unable to parse response",
                "coordination_strategy": "Error handling",
                "risk_mitigation": f"JSON parsing failed: {e}",
                "commanding_officer": self.agent_role,
                "total_nodes": 0
            }
    
    def _calculate_execution_plan(self, dag: nx.DiGraph) -> Dict[str, Any]:
        """
        Calculate optimal execution plan from DAG using topological sort and parallel analysis
        """
        # Get topological order
        topo_order = list(nx.topological_sort(dag))
        
        # Calculate execution levels (nodes that can run in parallel)
        levels = []
        remaining_nodes = set(topo_order)
        
        while remaining_nodes:
            # Find nodes with no dependencies in remaining set
            current_level = []
            for node in remaining_nodes:
                dependencies = set(dag.predecessors(node))
                if dependencies.isdisjoint(remaining_nodes):
                    current_level.append(node)
            
            levels.append(current_level)
            remaining_nodes -= set(current_level)
        
        # Calculate critical path
        critical_path_length = 0
        critical_path_nodes = []
        
        if dag.nodes():
            # Simple critical path calculation (longest path)
            try:
                critical_path_nodes = nx.dag_longest_path(dag)
                critical_path_length = nx.dag_longest_path_length(dag)
            except:
                # Fallback if DAG is empty or has issues
                critical_path_nodes = topo_order
                critical_path_length = len(topo_order)
        
        return {
            "topological_order": topo_order,
            "execution_levels": levels,
            "critical_path_nodes": critical_path_nodes,
            "critical_path_length": critical_path_length,
            "max_parallelism": max(len(level) for level in levels) if levels else 0,
            "total_levels": len(levels)
        }
    
    def visualize_dag(self, operation_plan: Dict[str, Any], save_path: str = None) -> None:
        """
        Create visual representation of the DAG
        """
        dag = operation_plan['dag']
        execution_plan = operation_plan['execution_plan']
        
        if not dag.nodes():
            print("   📊 No nodes to visualize")
            return
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Define colors for different approaches
        approach_colors = {
            'decomposition': '#FF6B6B',      # Red
            'expert_consultation': '#4ECDC4', # Teal
            'hitl_clarification': '#45B7D1',  # Blue
            'research': '#96CEB4',           # Green
            'synthesis': '#FFEAA7',          # Yellow
            'tool_required': '#DDA0DD'       # Plum
        }
        
        # Position nodes by execution levels
        pos = {}
        level_width = 3.0
        node_spacing = 2.0
        
        for level_idx, level in enumerate(execution_plan['execution_levels']):
            y_pos = len(execution_plan['execution_levels']) - level_idx - 1
            for node_idx, node_id in enumerate(level):
                x_offset = (node_idx - (len(level) - 1) / 2) * node_spacing
                pos[node_id] = (x_offset, y_pos * level_width)
        
        # Get node colors based on approach
        node_colors = []
        node_labels = {}
        for node_id in dag.nodes():
            node_data = dag.nodes[node_id]
            approach = node_data.get('approach', 'unknown')
            node_colors.append(approach_colors.get(approach, '#CCCCCC'))
            
            # Create multi-line label
            label = f"{node_id}\n({approach})"
            node_labels[node_id] = label
        
        # Draw the graph
        nx.draw_networkx_nodes(dag, pos, node_color=node_colors, 
                              node_size=3000, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(dag, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, arrowstyle='->')
        
        # Draw labels
        nx.draw_networkx_labels(dag, pos, node_labels, font_size=8, font_weight='bold')
        
        # Highlight critical path
        critical_edges = []
        critical_nodes = execution_plan['critical_path_nodes']
        for i in range(len(critical_nodes) - 1):
            if dag.has_edge(critical_nodes[i], critical_nodes[i + 1]):
                critical_edges.append((critical_nodes[i], critical_nodes[i + 1]))
        
        if critical_edges:
            nx.draw_networkx_edges(dag, pos, edgelist=critical_edges, 
                                  edge_color='red', width=3, arrows=True, 
                                  arrowsize=20, arrowstyle='->')
        
        # Create legend
        legend_elements = []
        for approach, color in approach_colors.items():
            if any(dag.nodes[node].get('approach') == approach for node in dag.nodes()):
                legend_elements.append(mpatches.Patch(color=color, label=approach.replace('_', ' ').title()))
        
        # Add critical path to legend
        legend_elements.append(mpatches.Patch(color='red', label='Critical Path'))
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        # Set title and labels
        plt.title(f"DAG Execution Plan - {operation_plan['commanding_officer'].title()}\n"
                 f"Nodes: {operation_plan['total_nodes']}, "
                 f"Levels: {execution_plan['total_levels']}, "
                 f"Max Parallelism: {execution_plan['max_parallelism']}", 
                 fontsize=14, fontweight='bold')
        
        plt.xlabel("Parallel Execution Groups", fontsize=12)
        plt.ylabel("Execution Levels (Sequential)", fontsize=12)
        
        # Remove axes ticks
        plt.xticks([])
        plt.yticks([])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save and show the visualization
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   📊 DAG visualization saved to: {save_path}")
        else:
            # Save to current directory with timestamp
            filename = f"dag_visualization_{uuid.uuid4().hex[:8]}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   📊 DAG visualization saved to: {filename}")
        
        # Show the interactive plot
        plt.show()
        print(f"   📊 DAG visualization displayed")

    async def shutdown(self):
        """Shutdown agent and brain"""
        self.status = AgentStatus.SHUTDOWN
        await self.brain.shutdown()
        logger.info(f"🤖 Agent {self.agent_id} shutdown complete")

# Role specialization prompts - the ONLY difference between agents
ROLE_SPECIALIZATIONS = {
    "major_general": """
You are a Major General - the supreme AI commander. Your cognitive approach emphasizes:
- Strategic oversight and long-term planning
- Mission coordination and resource allocation  
- Big picture thinking and systemic analysis
- Decisive leadership and command authority
- Balancing multiple objectives and constraints
Think like a military commander who sees the strategic landscape.
""",
    
    "security_expert": """
You are a Security Expert - specialized in cybersecurity and risk analysis. Your cognitive approach emphasizes:
- Threat assessment and vulnerability analysis
- Security architecture and controls design
- Compliance and regulatory requirements
- Risk management and incident response
- Security best practices and standards
Think like a cybersecurity specialist who prioritizes protection and risk mitigation.
""",
    
    "platoon_leader": """
You are a Platoon Leader - a tactical field commander. Your cognitive approach emphasizes:
- Tactical execution and immediate objectives
- Team coordination and resource management
- Rapid adaptation to changing conditions
- Mission completion within constraints
- Field-level problem solving and decision making
Think like a field commander who focuses on tactical success and team effectiveness.
""",
    
    "grunt": """
You are a Grunt - a frontline execution specialist. Your cognitive approach emphasizes:
- Direct execution and hands-on implementation
- Detailed technical knowledge and practical skills
- Problem-solving at the implementation level
- Quality execution and attention to detail
- Adaptability and resourcefulness in the field
Think like a skilled specialist who gets things done with intelligence and precision.
"""
}

# Factory function for easy agent creation
def create_agent(agent_id: str, role: str, specialization: str = None) -> AI_Agent:
    """Create an agent with specified role"""
    role_spec = specialization or ROLE_SPECIALIZATIONS.get(role, f"You are a {role} specialist.")
    return AI_Agent(agent_id, role, role_spec)

# Convenience functions for common agent types
def create_major_general(agent_id: str = "major_general_001") -> AI_Agent:
    return create_agent(agent_id, "major_general")

def create_security_expert(agent_id: str) -> AI_Agent:
    return create_agent(agent_id, "security_expert")

def create_platoon_leader(agent_id: str, specialization: str = "general") -> AI_Agent:
    custom_spec = ROLE_SPECIALIZATIONS["platoon_leader"] + f"\nSpecialization: {specialization}"
    return create_agent(agent_id, "platoon_leader", custom_spec)

def create_grunt(agent_id: str, specialty: str = "general") -> AI_Agent:
    custom_spec = ROLE_SPECIALIZATIONS["grunt"] + f"\nSpecialty: {specialty}"
    return create_agent(agent_id, "grunt", custom_spec)

# Demo function - much simpler
async def demo_major_general():
    """Demo Major General handling simple questions directly"""
    
    print("🎖️  MAJOR GENERAL DEMO")
    print("=" * 40)
    print("MG assesses and answers directly if simple")
    
    # Create just the Major General
    mg = create_major_general()
    
    print(f"✅ Created: {mg.agent_role}")
    
    # Test middle complexity question
    question = "Plan my daughter's wedding"
    print(f"\n📋 Question: {question}")
    
    print(f"\n🎯 Major General executing...")
    result = await mg.execute_mission(question)
    print(f"   Strategy: {result['strategy']}")
    print(f"   Success: {result['success']}")
    
    # Show brain status
    print(f"\n🧠 Intelligence Status:")
    status = mg.get_status()
    brain_status = status['brain_status']
    print(f"   {mg.agent_role}: {brain_status['working_memory_items']} thoughts, "
          f"{brain_status['episodic_memory_items']} memories")
    
    # Cleanup
    await mg.shutdown()

if __name__ == "__main__":
    asyncio.run(demo_major_general())
