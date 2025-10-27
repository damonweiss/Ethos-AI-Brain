#!/usr/bin/env python3
"""
Mission Graph Service - Knowledge service for mission structure and dependency analysis
Part of Sprint B: Knowledge Graph Foundation
"""

import asyncio
import json
import logging
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import zmq

from dag_models import DAGNode, DAGPlan, ApproachType, IntakeStrategy
from intake_planner import IntakePlanner

logger = logging.getLogger(__name__)

class MissionGraphService:
    """
    ZMQ REP service for mission structure queries and dependency analysis
    Integrates with intake_planner.py for mission graph creation
    """
    
    def __init__(self, port: int = 5580, meta_reasoning_engine=None):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.is_running = False
        
        # Initialize intake planner for mission analysis
        self.intake_planner = IntakePlanner(meta_reasoning_engine) if meta_reasoning_engine else None
        
        # Mission graph storage and templates
        self.mission_graphs: Dict[str, nx.DiGraph] = {}
        self.mission_templates: Dict[str, Dict[str, Any]] = {}
        self.dependency_patterns: Dict[str, List[str]] = {}
        
        # Initialize with basic templates
        self._initialize_mission_templates()
        self._initialize_dependency_patterns()
        
        logger.info(f"MissionGraphService initialized on port {port}")
    
    def _initialize_mission_templates(self):
        """Initialize basic mission templates for common patterns"""
        
        self.mission_templates = {
            "software_development": {
                "phases": ["analysis", "design", "implementation", "testing", "deployment"],
                "dependencies": {
                    "design": ["analysis"],
                    "implementation": ["design"],
                    "testing": ["implementation"],
                    "deployment": ["testing"]
                },
                "parallel_opportunities": [
                    ["documentation", "implementation"],
                    ["unit_testing", "integration_testing"]
                ]
            },
            
            "system_architecture": {
                "phases": ["requirements", "architecture_design", "component_design", "integration_planning", "validation"],
                "dependencies": {
                    "architecture_design": ["requirements"],
                    "component_design": ["architecture_design"],
                    "integration_planning": ["component_design"],
                    "validation": ["integration_planning"]
                },
                "parallel_opportunities": [
                    ["security_review", "performance_analysis"],
                    ["documentation", "component_design"]
                ]
            },
            
            "analysis_project": {
                "phases": ["data_gathering", "analysis", "synthesis", "recommendations", "presentation"],
                "dependencies": {
                    "analysis": ["data_gathering"],
                    "synthesis": ["analysis"],
                    "recommendations": ["synthesis"],
                    "presentation": ["recommendations"]
                },
                "parallel_opportunities": [
                    ["data_validation", "preliminary_analysis"]
                ]
            }
        }
    
    def _initialize_dependency_patterns(self):
        """Initialize common dependency patterns"""
        
        self.dependency_patterns = {
            "security_review": ["architecture_design", "component_design"],
            "performance_testing": ["implementation", "integration"],
            "documentation": ["design", "implementation"],
            "deployment": ["testing", "security_review", "performance_testing"],
            "user_acceptance": ["testing", "documentation"],
            "compliance_check": ["security_review", "documentation"]
        }
    
    async def start(self):
        """Start the ZMQ service"""
        try:
            self.socket.bind(f"tcp://*:{self.port}")
            self.is_running = True
            logger.info(f"MissionGraphService started on port {self.port}")
            
            while self.is_running:
                try:
                    # Wait for request with timeout
                    if self.socket.poll(1000):  # 1 second timeout
                        message = self.socket.recv_json()
                        response = await self._handle_request(message)
                        self.socket.send_json(response)
                    
                except zmq.Again:
                    continue  # Timeout, continue loop
                except Exception as e:
                    logger.error(f"Error handling request: {e}")
                    error_response = {
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    self.socket.send_json(error_response)
                    
        except Exception as e:
            logger.error(f"Failed to start MissionGraphService: {e}")
            raise
    
    async def _handle_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming ZMQ requests"""
        
        request_type = message.get("type")
        
        if request_type == "create_mission_graph":
            return await self._create_mission_graph(message)
        elif request_type == "analyze_dependencies":
            return await self._analyze_dependencies(message)
        elif request_type == "get_critical_path":
            return await self._get_critical_path(message)
        elif request_type == "identify_parallel_opportunities":
            return await self._identify_parallel_opportunities(message)
        elif request_type == "get_mission_templates":
            return await self._get_mission_templates(message)
        elif request_type == "validate_mission_graph":
            return await self._validate_mission_graph(message)
        else:
            return {
                "status": "error",
                "error": f"Unknown request type: {request_type}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _create_mission_graph(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Create mission graph from user objective using intake planner"""
        
        try:
            objective = message.get("objective", "")
            context = message.get("context", {})
            mission_id = message.get("mission_id", f"mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            if not objective:
                return {
                    "status": "error",
                    "error": "No objective provided",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Use intake planner to assess the mission
            if self.intake_planner:
                assessment = await self.intake_planner.assess_request(objective)
                
                # Create DAG based on assessment
                dag_plan = await self._create_dag_from_assessment(assessment, objective, context)
            else:
                # Fallback: create basic DAG without intake planner
                dag_plan = await self._create_basic_dag(objective, context)
            
            # Convert DAG to NetworkX graph
            nx_graph = self._dag_to_networkx(dag_plan)
            
            # Store the graph
            self.mission_graphs[mission_id] = nx_graph
            
            # Analyze the graph
            analysis = self._analyze_graph_structure(nx_graph)
            
            return {
                "status": "success",
                "mission_id": mission_id,
                "dag_plan": dag_plan.model_dump() if hasattr(dag_plan, 'model_dump') else dag_plan,
                "graph_analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating mission graph: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _create_dag_from_assessment(self, assessment, objective: str, context: Dict[str, Any]) -> DAGPlan:
        """Create DAG plan from intake assessment"""
        
        # This is a simplified version - will be enhanced with more sophisticated logic
        nodes = []
        
        # Always start with mission analysis
        nodes.append(DAGNode(
            id="mission_analysis",
            approach=ApproachType.DECOMPOSITION,
            purpose="Analyze mission requirements and constraints",
            resources="analysis, planning",
            dependencies=[],
            success_criteria="Mission requirements clearly understood and documented"
        ))
        
        # Add nodes based on assessment strategy
        if assessment.execution_strategy.value == "panel_of_experts":
            # Create expert consultation nodes
            for specialist in assessment.required_specialists:
                nodes.append(DAGNode(
                    id=f"expert_{specialist}",
                    approach=ApproachType.EXPERT_CONSULTATION,
                    purpose=f"Consult {specialist} for specialized input",
                    resources=f"{specialist}, domain_expertise",
                    dependencies=["mission_analysis"],
                    success_criteria=f"{specialist} recommendations provided"
                ))
            
            # Add synthesis node
            nodes.append(DAGNode(
                id="expert_synthesis",
                approach=ApproachType.SYNTHESIS,
                purpose="Synthesize expert recommendations into coherent plan",
                resources="synthesis, integration",
                dependencies=[f"expert_{s}" for s in assessment.required_specialists],
                success_criteria="Integrated recommendations ready for implementation"
            ))
        
        elif assessment.execution_strategy.value == "full_decomposition":
            # Create decomposition-based DAG
            template = self._select_mission_template(objective)
            if template:
                nodes.extend(self._create_nodes_from_template(template))
        
        # Add final validation node
        nodes.append(DAGNode(
            id="final_validation",
            approach=ApproachType.SYNTHESIS,
            purpose="Final validation and quality assurance",
            resources="validation, quality_assurance",
            dependencies=[node.id for node in nodes if node.id != "final_validation"][-1:],  # Depend on last node
            success_criteria="Mission objectives achieved and validated"
        ))
        
        return DAGPlan(
            strategy=IntakeStrategy(assessment.execution_strategy.value),
            nodes=nodes,
            estimated_time=assessment.estimated_time,
            confidence=assessment.confidence
        )
    
    async def _create_basic_dag(self, objective: str, context: Dict[str, Any]) -> DAGPlan:
        """Create basic DAG without intake planner (fallback)"""
        
        nodes = [
            DAGNode(
                id="analysis",
                approach=ApproachType.DECOMPOSITION,
                purpose="Analyze the objective and requirements",
                resources="analysis, planning",
                dependencies=[],
                success_criteria="Requirements clearly understood"
            ),
            DAGNode(
                id="execution",
                approach=ApproachType.SYNTHESIS,
                purpose="Execute the main objective",
                resources="implementation, execution",
                dependencies=["analysis"],
                success_criteria="Objective completed successfully"
            ),
            DAGNode(
                id="validation",
                approach=ApproachType.SYNTHESIS,
                purpose="Validate results and ensure quality",
                resources="validation, quality_assurance",
                dependencies=["execution"],
                success_criteria="Results validated and approved"
            )
        ]
        
        return DAGPlan(
            strategy=IntakeStrategy.TASK_DECOMPOSITION,
            nodes=nodes,
            estimated_time=5.0,
            confidence=0.7
        )
    
    def _select_mission_template(self, objective: str) -> Optional[Dict[str, Any]]:
        """Select appropriate mission template based on objective"""
        
        objective_lower = objective.lower()
        
        if any(word in objective_lower for word in ["build", "develop", "create", "implement", "code"]):
            return self.mission_templates.get("software_development")
        elif any(word in objective_lower for word in ["architecture", "design", "system", "structure"]):
            return self.mission_templates.get("system_architecture")
        elif any(word in objective_lower for word in ["analyze", "analysis", "research", "study"]):
            return self.mission_templates.get("analysis_project")
        
        return None
    
    def _create_nodes_from_template(self, template: Dict[str, Any]) -> List[DAGNode]:
        """Create DAG nodes from mission template"""
        
        nodes = []
        phases = template.get("phases", [])
        dependencies = template.get("dependencies", {})
        
        for phase in phases:
            phase_deps = dependencies.get(phase, [])
            
            nodes.append(DAGNode(
                id=phase,
                approach=ApproachType.DECOMPOSITION,
                purpose=f"Execute {phase} phase",
                resources=f"{phase}, specialized_knowledge",
                dependencies=phase_deps,
                success_criteria=f"{phase} phase completed successfully"
            ))
        
        return nodes
    
    def _dag_to_networkx(self, dag_plan: DAGPlan) -> nx.DiGraph:
        """Convert DAG plan to NetworkX graph"""
        
        graph = nx.DiGraph()
        
        # Add nodes
        for node in dag_plan.nodes:
            graph.add_node(
                node.id,
                approach=node.approach.value,
                purpose=node.purpose,
                resources=node.resources,
                success_criteria=node.success_criteria
            )
        
        # Add edges based on dependencies
        for node in dag_plan.nodes:
            for dep in node.dependencies:
                graph.add_edge(dep, node.id)
        
        return graph
    
    def _analyze_graph_structure(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze NetworkX graph structure"""
        
        analysis = {
            "node_count": len(graph.nodes()),
            "edge_count": len(graph.edges()),
            "is_dag": nx.is_directed_acyclic_graph(graph),
            "has_cycles": False,
            "critical_path": [],
            "parallel_opportunities": [],
            "complexity_score": 0.0
        }
        
        if analysis["is_dag"]:
            # Calculate critical path
            try:
                critical_path = nx.dag_longest_path(graph)
                analysis["critical_path"] = critical_path
                analysis["critical_path_length"] = len(critical_path)
            except:
                analysis["critical_path"] = []
                analysis["critical_path_length"] = 0
            
            # Identify parallel opportunities
            analysis["parallel_opportunities"] = self._find_parallel_groups(graph)
        else:
            analysis["has_cycles"] = True
            analysis["cycles"] = list(nx.simple_cycles(graph))
        
        # Calculate complexity score
        analysis["complexity_score"] = self._calculate_complexity_score(graph)
        
        return analysis
    
    def _find_parallel_groups(self, graph: nx.DiGraph) -> List[List[str]]:
        """Find groups of nodes that can execute in parallel"""
        
        parallel_groups = []
        
        # Group nodes by their level in the DAG
        levels = {}
        for node in nx.topological_sort(graph):
            predecessors = list(graph.predecessors(node))
            if not predecessors:
                levels[node] = 0
            else:
                levels[node] = max(levels[pred] for pred in predecessors) + 1
        
        # Find levels with multiple nodes
        level_groups = {}
        for node, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node)
        
        for level, nodes in level_groups.items():
            if len(nodes) > 1:
                parallel_groups.append(nodes)
        
        return parallel_groups
    
    def _calculate_complexity_score(self, graph: nx.DiGraph) -> float:
        """Calculate mission complexity score"""
        
        node_count = len(graph.nodes())
        edge_count = len(graph.edges())
        
        # Basic complexity calculation
        complexity = (node_count * 0.5) + (edge_count * 0.3)
        
        # Adjust for parallel opportunities (reduces complexity)
        parallel_groups = self._find_parallel_groups(graph)
        parallel_reduction = len(parallel_groups) * 0.2
        
        return max(0.1, complexity - parallel_reduction)
    
    async def _analyze_dependencies(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dependencies for a mission graph"""
        
        mission_id = message.get("mission_id")
        if not mission_id or mission_id not in self.mission_graphs:
            return {
                "status": "error",
                "error": "Mission graph not found",
                "timestamp": datetime.now().isoformat()
            }
        
        graph = self.mission_graphs[mission_id]
        analysis = self._analyze_graph_structure(graph)
        
        return {
            "status": "success",
            "mission_id": mission_id,
            "dependency_analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_critical_path(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Get critical path for mission execution"""
        
        mission_id = message.get("mission_id")
        if not mission_id or mission_id not in self.mission_graphs:
            return {
                "status": "error",
                "error": "Mission graph not found",
                "timestamp": datetime.now().isoformat()
            }
        
        graph = self.mission_graphs[mission_id]
        
        if nx.is_directed_acyclic_graph(graph):
            critical_path = nx.dag_longest_path(graph)
            return {
                "status": "success",
                "mission_id": mission_id,
                "critical_path": critical_path,
                "critical_path_length": len(critical_path),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "error": "Graph contains cycles, cannot calculate critical path",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _identify_parallel_opportunities(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Identify parallel execution opportunities"""
        
        mission_id = message.get("mission_id")
        if not mission_id or mission_id not in self.mission_graphs:
            return {
                "status": "error",
                "error": "Mission graph not found",
                "timestamp": datetime.now().isoformat()
            }
        
        graph = self.mission_graphs[mission_id]
        parallel_groups = self._find_parallel_groups(graph)
        
        return {
            "status": "success",
            "mission_id": mission_id,
            "parallel_opportunities": parallel_groups,
            "potential_time_savings": len(parallel_groups) * 0.3,  # Rough estimate
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_mission_templates(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Get available mission templates"""
        
        return {
            "status": "success",
            "templates": list(self.mission_templates.keys()),
            "template_details": self.mission_templates,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _validate_mission_graph(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Validate mission graph structure"""
        
        mission_id = message.get("mission_id")
        if not mission_id or mission_id not in self.mission_graphs:
            return {
                "status": "error",
                "error": "Mission graph not found",
                "timestamp": datetime.now().isoformat()
            }
        
        graph = self.mission_graphs[mission_id]
        
        validation_results = {
            "is_valid_dag": nx.is_directed_acyclic_graph(graph),
            "has_isolated_nodes": len(list(nx.isolates(graph))) > 0,
            "is_connected": nx.is_weakly_connected(graph),
            "node_count": len(graph.nodes()),
            "edge_count": len(graph.edges())
        }
        
        if not validation_results["is_valid_dag"]:
            validation_results["cycles"] = list(nx.simple_cycles(graph))
        
        return {
            "status": "success",
            "mission_id": mission_id,
            "validation": validation_results,
            "timestamp": datetime.now().isoformat()
        }
    
    def stop(self):
        """Stop the ZMQ service"""
        self.is_running = False
        self.socket.close()
        self.context.term()
        logger.info("MissionGraphService stopped")

# Demo function
async def demo_mission_graph_service():
    """Demo the mission graph service"""
    
    print("🎖️  MISSION GRAPH SERVICE DEMO")
    print("=" * 50)
    
    # Mock meta-reasoning for demo
    class MockMetaReasoning:
        class MockLLMBackend:
            async def complete(self, prompt, context):
                return "INTENT: design_request\nCOMPLEXITY: 3\nSTRATEGY: full_decomposition"
        
        def __init__(self):
            self.llm_backends = {"analyst": self.MockLLMBackend()}
    
    service = MissionGraphService(port=5580, meta_reasoning_engine=MockMetaReasoning())
    
    # Test mission graph creation
    test_request = {
        "type": "create_mission_graph",
        "objective": "Build a secure fintech API for high-frequency trading",
        "context": {"budget": 100000, "timeline": "3 months"},
        "mission_id": "test_mission_001"
    }
    
    print(f"📝 Creating mission graph for: {test_request['objective']}")
    
    response = await service._handle_request(test_request)
    
    if response["status"] == "success":
        print(f"✅ Mission graph created successfully!")
        print(f"   Mission ID: {response['mission_id']}")
        print(f"   Node count: {response['graph_analysis']['node_count']}")
        print(f"   Critical path length: {response['graph_analysis'].get('critical_path_length', 'N/A')}")
        print(f"   Parallel opportunities: {len(response['graph_analysis']['parallel_opportunities'])}")
        print(f"   Complexity score: {response['graph_analysis']['complexity_score']:.2f}")
    else:
        print(f"❌ Error: {response['error']}")

if __name__ == "__main__":
    asyncio.run(demo_mission_graph_service())
