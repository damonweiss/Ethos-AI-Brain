#!/usr/bin/env python3
"""
Meta-Reasoning Base Demo - ZMQ Parallel Processing POC
Single Restaurant POS scenario demonstrating sophisticated parallel intent analysis

Features:
- ZMQ parallel processing for stakeholder and node-type expansions
- JSON prompting and structured output to terminal
- Complete 7-stage analysis pipeline with enhanced graph generation
- Single POC scenario: Restaurant POS system request
- Parallel LLM calls using ZeroMQ distributed processing
- Comprehensive JSON output including parallel processing results
"""

import asyncio
import json
import time
import uuid
from typing import Dict
from meta_reasoning_base import MetaReasoningBase, ReasoningContext
from knowledge_graph import KnowledgeGraph, GraphType

async def build_graph_from_parallel_expansions(
    graph: KnowledgeGraph, 
    complexity_result: Dict, 
    stakeholder_expansions: Dict, 
    node_type_expansions: Dict
) -> KnowledgeGraph:
    """
    Build enhanced knowledge graph from parallel expansion results
    Uses the rich data from ZMQ parallel processing
    """
    print(f"🏗️ Building graph from parallel expansion results...")
    
    # Add stakeholders and their expanded objectives
    stakeholders = complexity_result.get("stakeholders", [])
    for i, stakeholder in enumerate(stakeholders[:3]):  # Limit to 3
        stakeholder_name = stakeholder.get("name", f"Stakeholder_{i}")
        stakeholder_id = f"stakeholder_{i}"
        
        graph.add_node(
            stakeholder_id,
            type="stakeholder",
            name=stakeholder_name,
            role=stakeholder.get("role", "Unknown")
        )
        
        # Add objectives with expanded details from parallel processing
        objectives = stakeholder.get("objectives", [])
        for j, objective in enumerate(objectives[:2]):  # Limit to 2 per stakeholder
            obj_name = objective.get("name", f"Objective_{j}")
            obj_id = f"objective_{i}_{j}"
            
            # Look for expansion data for this stakeholder-objective pair
            expansion_key = f"stakeholder_{stakeholder_name}_{obj_name}".replace(" ", "_").lower()
            expansion_data = stakeholder_expansions.get(expansion_key, {})
            expansion = expansion_data.get("expansion", {})
            
            # Add objective with expanded attributes
            graph.add_node(
                obj_id,
                type="primary_objective",
                name=obj_name,
                priority=objective.get("priority", "medium"),
                success_metrics=expansion.get("success_metrics", []),
                dependencies=expansion.get("dependencies", [])
            )
            
            # Link stakeholder to objective
            graph.add_edge(stakeholder_id, obj_id, relationship="owns")
    
    # Add constraints from node type expansions
    constraints_expansion = node_type_expansions.get("node_type_constraints", {})
    constraints_data = constraints_expansion.get("expansion", {}).get("constraints", [])
    
    for i, constraint in enumerate(constraints_data[:3]):  # Limit to 3
        constraint_id = f"constraint_{i}"
        constraint_type = constraint.get("type", "general")
        
        graph.add_node(
            constraint_id,
            type="constraint",
            name=f"{constraint_type.title()} Constraint",
            constraint_type=constraint_type,
            flexibility=constraint.get("flexibility", "moderate"),
            amount=constraint.get("amount", None)
        )
        
        # Link constraints to objectives
        for node_id in graph.nodes():
            if graph.get_node_attributes(node_id).get('type') == 'primary_objective':
                graph.add_edge(constraint_id, node_id, relationship="constrains")
    
    # Add technical requirements from node type expansions
    tech_expansion = node_type_expansions.get("node_type_technical_requirements", {})
    tech_data = tech_expansion.get("expansion", {}).get("technical_requirements", [])
    
    for i, tech_req in enumerate(tech_data[:4]):  # Limit to 4
        tech_id = f"tech_{i}"
        
        graph.add_node(
            tech_id,
            type="technical_requirement",
            name=tech_req.get("requirement", f"Technical Requirement {i}"),
            importance=tech_req.get("importance", "medium"),
            complexity=tech_req.get("complexity", "medium")
        )
        
        # Link tech requirements to objectives based on importance
        if tech_req.get("importance") == "critical":
            for node_id in graph.nodes():
                if graph.get_node_attributes(node_id).get('type') == 'primary_objective':
                    graph.add_edge(tech_id, node_id, relationship="critically_enables")
    
    # Add assumptions from node type expansions
    assumptions_expansion = node_type_expansions.get("node_type_assumptions", {})
    assumptions_data = assumptions_expansion.get("expansion", {}).get("assumptions", [])
    
    for i, assumption in enumerate(assumptions_data[:3]):  # Limit to 3
        assumption_id = f"assumption_{i}"
        
        graph.add_node(
            assumption_id,
            type="assumption",
            name=f"Assumption {i+1}",
            assumption=assumption.get("assumption", "Unknown assumption"),
            confidence=assumption.get("confidence", 0.5),
            impact_if_wrong=assumption.get("impact_if_wrong", "medium")
        )
        
        # High-risk assumptions link to objectives
        if assumption.get("impact_if_wrong") in ["high", "critical"]:
            for node_id in graph.nodes():
                if graph.get_node_attributes(node_id).get('type') == 'primary_objective':
                    graph.add_edge(assumption_id, node_id, relationship="high_risk_for")
    
    print(f"   Added nodes from parallel expansions: {len(graph.nodes())} total nodes")
    return graph

async def run_intent_analysis(engine: MetaReasoningBase, user_input: str, context: ReasoningContext):
    """
    Run intent analysis pipeline with JSON prompting and output
    Focus only on intent assessment from user prompts
    """
    try:
        print(f"🎯 INTENT ANALYSIS PIPELINE")
        print("=" * 60)
        print(f"Input: {user_input}")
        
        # Stage 1: Complexity Assessment with JSON output
        print(f"\n🎯 STAGE 1: INTENT COMPLEXITY ASSESSMENT")
        print("-" * 50)
        complexity_result = await engine.analyze_complexity(user_input)
        
        # Print JSON output for complexity assessment
        print(f"📋 COMPLEXITY ASSESSMENT JSON:")
        print(json.dumps(complexity_result, indent=2))
        
        if complexity_result.get("complexity") == "simple":
            simple_result = {
                "analysis_type": "simple_intent",
                "complexity_assessment": complexity_result,
                "direct_response": complexity_result.get("direct_answer", "Simple response provided"),
                "reasoning": complexity_result.get("reasoning", "Determined to be simple request"),
                "requires_further_analysis": False
            }
            print(f"\n📋 FINAL SIMPLE INTENT JSON:")
            print(json.dumps(simple_result, indent=2))
            return simple_result
        
        # Stage 2: Intent Node Type Assessment with JSON output
        print(f"\n🎯 STAGE 2: INTENT NODE TYPE ASSESSMENT")
        print("-" * 50)
        node_types_result = await engine.assess_required_node_types(user_input, complexity_result)
        
        # Print JSON output for node types
        print(f"📋 NODE TYPES ASSESSMENT JSON:")
        print(json.dumps(node_types_result, indent=2))
        
        # Stage 2.5: Parallel Stakeholder Expansion (NEW ZMQ CAPABILITY)
        print(f"\n🚀 STAGE 2.5: PARALLEL STAKEHOLDER EXPANSION")
        print("-" * 50)
        stage_2_5_start = time.time()
        stakeholder_expansions = await engine.parallel_stakeholder_expansion(user_input, complexity_result)
        stage_2_5_time = time.time() - stage_2_5_start
        
        # Print JSON output for stakeholder expansions
        print(f"📋 STAKEHOLDER EXPANSIONS JSON:")
        print(json.dumps(stakeholder_expansions, indent=2))
        
        # Stage 2.6: Parallel Node Type Expansion (NEW ZMQ CAPABILITY)
        print(f"\n🚀 STAGE 2.6: PARALLEL NODE TYPE EXPANSION")
        print("-" * 50)
        stage_2_6_start = time.time()
        node_type_expansions = await engine.parallel_node_type_expansion(user_input, node_types_result)
        stage_2_6_time = time.time() - stage_2_6_start
        
        # Print JSON output for node type expansions
        print(f"📋 NODE TYPE EXPANSIONS JSON:")
        print(json.dumps(node_type_expansions, indent=2))
        
        # Create enhanced knowledge graph using parallel expansion results
        graph = KnowledgeGraph(
            graph_type=GraphType.INTENT,
            graph_id=f"demo_{context.session_id[:8]}"
        )
        
        # Build graph from parallel expansion results
        graph = await build_graph_from_parallel_expansions(
            graph, 
            complexity_result, 
            stakeholder_expansions, 
            node_type_expansions
        )
        
        print(f"✅ Created enhanced graph from parallel expansions: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
        
        # Stage 3: Task Decomposition with JSON output
        print(f"\n🎯 STAGE 3: TASK DECOMPOSITION")
        print("-" * 50)
        tasks_result = await engine.decompose_tasks(user_input, complexity_result)
        
        # Print JSON output for tasks
        print(f"📋 TASK DECOMPOSITION JSON:")
        print(json.dumps(tasks_result, indent=2))
        
        # Stage 4: QAQC Validation with JSON output
        print(f"\n🎯 STAGE 4: QAQC VALIDATION")
        print("-" * 50)
        analysis_data = {
            "complexity_assessment": complexity_result,
            "node_types": node_types_result,
            "tasks": tasks_result
        }
        qaqc_result = await engine.validate_quality(user_input, analysis_data)
        
        # Print JSON output for QAQC
        print(f"📋 QAQC VALIDATION JSON:")
        print(json.dumps(qaqc_result, indent=2))
        
        # Stage 5: Heuristic Linking (Apply sophisticated relationships)
        print(f"\n🎯 STAGE 5: HEURISTIC LINKING")
        print("-" * 50)
        enhanced_graph = engine.apply_heuristic_linking(graph, complexity_result)
        
        # Enhance graph with QAQC results
        final_graph = engine.enhance_graph_with_qaqc(enhanced_graph, qaqc_result)
        
        # Create graph JSON representation
        graph_json = {
            "graph_id": final_graph.graph_id,
            "graph_type": final_graph.graph_type.value,
            "nodes": [],
            "edges": []
        }
        
        # Add nodes to JSON
        for node_id in final_graph.nodes():
            node_attrs = final_graph.get_node_attributes(node_id)
            graph_json["nodes"].append({
                "id": node_id,
                "attributes": node_attrs
            })
        
        # Add edges to JSON
        for edge in final_graph.edges():
            source_id, target_id = edge
            edge_attrs = final_graph.get_edge_attributes(source_id, target_id)
            graph_json["edges"].append({
                "source": source_id,
                "target": target_id,
                "attributes": edge_attrs
            })
        
        print(f"\n✅ INTENT ANALYSIS COMPLETE")
        print(f"   Final Graph: {len(final_graph.nodes())} nodes, {len(final_graph.edges())} edges")
        print(f"   Tasks: {len(tasks_result.get('task_breakdown', []))} actionable tasks")
        print(f"   Quality Score: {qaqc_result.get('overall_quality_score', 'N/A')}")
        
        # Performance Summary
        print(f"\n⏱️ PERFORMANCE SUMMARY:")
        print(f"   Stage 2.5 (Stakeholder Expansion): {stage_2_5_time:.2f}s")
        print(f"   Stage 2.6 (Node Type Expansion): {stage_2_6_time:.2f}s")
        total_parallel_time = stage_2_5_time + stage_2_6_time
        print(f"   Total Parallel Processing Time: {total_parallel_time:.2f}s")
        
        # Create comprehensive JSON result with parallel processing data
        complete_result = {
            "analysis_type": "complex_intent_with_parallel_processing",
            "user_input": user_input,
            "complexity_assessment": complexity_result,
            "node_types_assessment": node_types_result,
            "parallel_stakeholder_expansions": stakeholder_expansions,
            "parallel_node_type_expansions": node_type_expansions,
            "task_decomposition": tasks_result,
            "qaqc_validation": qaqc_result,
            "knowledge_graph": graph_json,
            "parallel_processing_summary": {
                "stakeholder_expansions_count": len(stakeholder_expansions),
                "node_type_expansions_count": len(node_type_expansions),
                "total_parallel_calls": len(stakeholder_expansions) + len(node_type_expansions)
            },
            "performance_metrics": {
                "stakeholder_expansion_time": stage_2_5_time,
                "node_type_expansion_time": stage_2_6_time,
                "total_parallel_processing_time": total_parallel_time
            },
            "summary": {
                "total_nodes": len(final_graph.nodes()),
                "total_edges": len(final_graph.edges()),
                "total_tasks": len(tasks_result.get('task_breakdown', [])),
                "quality_score": qaqc_result.get('overall_quality_score', 0.0),
                "validation_passed": qaqc_result.get('validation_passed', False)
            }
        }
        
        # Print final comprehensive JSON
        print(f"\n📋 COMPLETE INTENT ANALYSIS JSON:")
        print("=" * 80)
        print(json.dumps(complete_result, indent=2))
        print("=" * 80)
        
        return {
            "status": "complex_analysis_with_parallel_processing",
            "graph": final_graph,
            "tasks": tasks_result,
            "qaqc": qaqc_result,
            "complexity_assessment": complexity_result,
            "node_types": node_types_result,
            "stakeholder_expansions": stakeholder_expansions,
            "node_type_expansions": node_type_expansions,
            "complete_json": complete_result
        }
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

async def display_single_graph_insights(graph):
    """Display graph insights for a single graph"""
    
    print(f"\n🔍 GRAPH INSIGHTS ANALYSIS")
    print("-" * 50)
    
    # Basic graph metrics
    print(f"   Nodes: {len(graph.nodes())}")
    print(f"   Edges: {len(graph.edges())}")
    
    # Node type analysis
    node_types = {}
    for node_id in graph.nodes():
        node_attrs = graph.get_node_attributes(node_id)
        node_type = node_attrs.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    if node_types:
        print(f"   Node Types: {dict(node_types)}")
    
    # Relationship analysis
    relationships = {}
    for edge in graph.edges():
        source_id, target_id = edge
        edge_attrs = graph.get_edge_attributes(source_id, target_id)
        relationship = edge_attrs.get('relationship', 'unknown')
        relationships[relationship] = relationships.get(relationship, 0) + 1
    
    if relationships:
        print(f"   Relationships: {dict(relationships)}")
    
    # High-degree nodes (potential bottlenecks)
    high_degree_nodes = []
    for node_id in graph.nodes():
        degree = graph.degree(node_id)
        if degree > 2:  # Nodes with more than 2 connections
            node_attrs = graph.get_node_attributes(node_id)
            node_name = node_attrs.get('name', node_id)
            high_degree_nodes.append(f"{node_name} ({degree} connections)")
    
    if high_degree_nodes:
        print(f"   High-Degree Nodes: {high_degree_nodes[:3]}")  # Show top 3

async def example_meta_reasoning_analysis():
    """Example of meta-reasoning analysis - based on adaptive_intent_runner structure"""
    
    engine = MetaReasoningBase()  # Only real OpenAI backends supported
    
    # Single POC test case - Restaurant POS
    test_cases = [
        {
            "name": "RESTAURANT POS REQUEST", 
            "input": "I need a new POS system for my restaurant that handles orders, payments, and inventory with a $25k budget in 6 weeks",
            "expected": "complex_analysis"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}: {test_case['name']}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Create reasoning context
        context = ReasoningContext(
            goal=test_case["input"],
            constraints={"budget": 25000, "timeline": "6 weeks"} if "restaurant" in test_case["input"].lower() else {},
            user_preferences={"analysis_depth": "comprehensive"},
            metadata={"test_case": test_case["name"]}
        )
        
        # Run intent analysis pipeline with JSON output
        result = await run_intent_analysis(engine, test_case["input"], context)
        
        total_time = time.time() - start_time
        
        # Display results in adaptive_intent_runner format
        print(f"\n🎯 RESULT: {result['status']}")
        print(f"   Total time: {total_time:.2f}s")
        
        if result.get('graph'):
            graph = result['graph']
            print(f"   Graph Generated: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
        
        if result.get('tasks'):
            tasks = result['tasks']
            print(f"   Tasks: {len(tasks.get('task_breakdown', []))} actionable tasks")
        
        if result.get('qaqc'):
            qaqc = result['qaqc']
            quality_score = qaqc.get('overall_quality_score', 0.0)
            validation_passed = qaqc.get('validation_passed', False)
            print(f"   QAQC: {quality_score:.2f}/1.0 quality, {'PASSED' if validation_passed else 'NEEDS REVIEW'}")
        
        if result.get('error'):
            print(f"   ❌ Error: {result['error']}")
        
        # Show detailed graph analysis (similar to adaptive's graph insights)
        if result.get('graph'):
            await display_single_graph_insights(result['graph'])

async def display_graph_insights(graphs):
    """Display graph insights similar to adaptive_intent_runner's graph analysis"""
    
    print(f"\n🔍 GRAPH INSIGHTS ANALYSIS")
    print("-" * 50)
    
    for graph_name, graph in graphs.items():
        print(f"\n📊 {graph_name.upper()} GRAPH ANALYSIS:")
        
        # Basic graph metrics
        print(f"   Nodes: {len(graph.nodes())}")
        print(f"   Edges: {len(graph.edges())}")
        
        # Node type analysis
        node_types = {}
        for node_id in graph.nodes():
            node_attrs = graph.get_node_attributes(node_id)
            node_type = node_attrs.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        if node_types:
            print(f"   Node Types: {dict(node_types)}")
        
        # Relationship analysis
        relationships = {}
        for edge in graph.edges():
            source_id, target_id = edge
            edge_attrs = graph.get_edge_attributes(source_id, target_id)
            relationship = edge_attrs.get('relationship', 'unknown')
            relationships[relationship] = relationships.get(relationship, 0) + 1
        
        if relationships:
            print(f"   Relationships: {dict(relationships)}")
        
        # High-degree nodes (potential bottlenecks)
        high_degree_nodes = []
        for node_id in graph.nodes():
            degree = graph.degree(node_id)
            if degree > 2:  # Nodes with more than 2 connections
                node_attrs = graph.get_node_attributes(node_id)
                node_name = node_attrs.get('name', node_id)
                high_degree_nodes.append(f"{node_name} ({degree} connections)")
        
        if high_degree_nodes:
            print(f"   High-Degree Nodes: {high_degree_nodes[:3]}")  # Show top 3


async def main():
    """Main demo function - matches adaptive_intent_runner structure"""
    
    print("Meta-Reasoning Base Engine - Graph-Driven Pipeline Demo")
    print("Initializing Meta-Reasoning Engine...")
    
    # Run the main analysis examples
    await example_meta_reasoning_analysis()

if __name__ == "__main__":
    # Run single POC demo with JSON output
    asyncio.run(main())
