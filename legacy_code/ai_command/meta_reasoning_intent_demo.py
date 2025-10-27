#!/usr/bin/env python3
"""
Intent Analysis Engine Demo - Replicating AdaptiveIntentRunner Results
Using the new IntentAnalysisEngine subclass with sophisticated heuristics

This demo should produce the same high-quality results as AdaptiveIntentRunner
but with the clean MetaReasoningBase architecture.
"""

import asyncio
import json
import time
from meta_reasoning_intent import IntentAnalysisEngine
from knowledge_graph import KnowledgeGraph, GraphType

async def run_intent_analysis_demo():
    """
    Demo using IntentAnalysisEngine to replicate AdaptiveIntentRunner sophistication
    """
    print("Intent Analysis Engine Demo - Advanced Meta-Reasoning")
    print("=" * 80)
    
    # Initialize the intent-specific engine
    engine = IntentAnalysisEngine()  # Uses real OpenAI backends
    
    # Test case - Restaurant POS (same as AdaptiveIntentRunner demo)
    user_input = "I need a new POS system for my restaurant that handles orders, payments, and inventory with a $25k budget in 6 weeks.  Disregard operational needs like staff training.  Focus on technical design / implementation."
    
    print(f"🎯 USER REQUEST: {user_input}")
    print("=" * 80)
    
    total_start_time = time.time()
    
    try:
        # Stage 1: Complexity Assessment (inherited from base)
        print(f"\n🚀 STAGE 1: COMPLEXITY ASSESSMENT")
        print("-" * 50)
        complexity_result = await engine.analyze_complexity(user_input)
        
        print(f"📋 COMPLEXITY ASSESSMENT JSON:")
        print(json.dumps(complexity_result, indent=2))
        
        # Stage 2: Node Type Assessment (inherited from base, enhanced with intent specs)
        print(f"\n🚀 STAGE 2: NODE TYPE ASSESSMENT")
        print("-" * 50)
        node_types_result = await engine.assess_required_node_types(user_input, complexity_result)
        
        print(f"📋 NODE TYPES ASSESSMENT JSON:")
        print(json.dumps(node_types_result, indent=2))
        
        # Stage 2.5: Parallel Stakeholder Expansion (inherited from base)
        print(f"\n🚀 STAGE 2.5: PARALLEL STAKEHOLDER EXPANSION")
        print("-" * 50)
        stage_2_5_start = time.time()
        stakeholder_expansions = await engine.parallel_stakeholder_expansion(user_input, complexity_result)
        stage_2_5_time = time.time() - stage_2_5_start
        
        print(f"📋 STAKEHOLDER EXPANSIONS JSON:")
        print(json.dumps(stakeholder_expansions, indent=2))
        
        # Stage 2.6: Parallel Node Type Expansion (inherited from base, enhanced prompts)
        print(f"\n🚀 STAGE 2.6: PARALLEL NODE TYPE EXPANSION")
        print("-" * 50)
        stage_2_6_start = time.time()
        node_type_expansions = await engine.parallel_node_type_expansion(user_input, node_types_result)
        stage_2_6_time = time.time() - stage_2_6_start
        
        print(f"📋 NODE TYPE EXPANSIONS JSON:")
        print(json.dumps(node_type_expansions, indent=2))
        
        # Stage 3: Task Decomposition (inherited from base)
        print(f"\n🚀 STAGE 3: TASK DECOMPOSITION")
        print("-" * 50)
        tasks_result = await engine.decompose_tasks(user_input, {
            "complexity": complexity_result,
            "node_types": node_types_result,
            "stakeholder_expansions": stakeholder_expansions,
            "node_type_expansions": node_type_expansions
        })
        
        print(f"📋 TASK DECOMPOSITION JSON:")
        print(json.dumps(tasks_result, indent=2))
        
        # Stage 4: QAQC Validation (inherited from base)
        print(f"\n🚀 STAGE 4: QAQC VALIDATION")
        print("-" * 50)
        analysis_data = {
            "complexity": complexity_result,
            "node_types": node_types_result,
            "stakeholder_expansions": stakeholder_expansions,
            "node_type_expansions": node_type_expansions,
            "tasks": tasks_result
        }
        qaqc_result = await engine.validate_quality(user_input, analysis_data)
        
        print(f"📋 QAQC VALIDATION JSON:")
        print(json.dumps(qaqc_result, indent=2))
        
        # Stage 5: Intent-Specific Heuristic Linking (NEW - sophisticated from AdaptiveIntentRunner)
        print(f"\n🚀 STAGE 5: INTENT-SPECIFIC HEURISTIC LINKING")
        print("-" * 50)
        
        # Create initial knowledge graph
        graph = KnowledgeGraph(
            graph_type=GraphType.INTENT,
            graph_id=f"intent_demo_{int(time.time())}"
        )
        
        # Add stakeholders and objectives to graph
        stakeholders = complexity_result.get("stakeholders", [])
        for i, stakeholder in enumerate(stakeholders):
            stakeholder_id = f"stakeholder_{i}"
            graph.add_node(
                stakeholder_id,
                type="stakeholder",
                name=stakeholder.get("name"),
                role=stakeholder.get("role")
            )
            
            # Add objectives
            objectives = stakeholder.get("objectives", [])
            for j, objective in enumerate(objectives):
                objective_id = f"objective_{i}_{j}"
                graph.add_node(
                    objective_id,
                    type="primary_objective",
                    name=objective.get("name"),
                    priority=objective.get("priority"),
                    success_metrics=[],
                    dependencies=[]
                )
                
                # Link stakeholder to objective
                graph.add_edge(stakeholder_id, objective_id, relationship="owns")
        
        # Apply sophisticated heuristic linking (NEW - from AdaptiveIntentRunner)
        enhanced_analysis_data = {
            "complexity_assessment": complexity_result,
            "stakeholder_expansions": stakeholder_expansions,
            "node_type_expansions": node_type_expansions
        }
        
        enhanced_graph = engine.apply_heuristic_linking(graph, enhanced_analysis_data)
        
        # Enhance graph with QAQC results (inherited from base)
        final_graph = engine.enhance_graph_with_qaqc(enhanced_graph, qaqc_result)
        
        # Create comprehensive JSON result
        graph_json = {
            "graph_id": final_graph.graph_id,
            "graph_type": "intent",
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
        
        # Create comprehensive JSON result
        complete_result = {
            "analysis_type": "intent_analysis_with_sophisticated_heuristics",
            "user_input": user_input,
            "complexity_assessment": complexity_result,
            "node_types_assessment": node_types_result,
            "stakeholder_expansions": stakeholder_expansions,
            "node_type_expansions": node_type_expansions,
            "task_decomposition": tasks_result,
            "qaqc_validation": qaqc_result,
            "knowledge_graph": graph_json,
            "performance_metrics": {
                "stakeholder_expansion_time": stage_2_5_time,
                "node_type_expansion_time": stage_2_6_time,
                "total_parallel_processing_time": total_parallel_time
            },
            "heuristic_analysis": {
                "sophisticated_linking_applied": True,
                "intent_specific_patterns": True,
                "systems_thinking_enabled": True,
                "ai_actionability_assessed": True
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
        
        total_time = time.time() - total_start_time
        print(f"\n🎯 RESULT: intent_analysis_with_sophisticated_heuristics")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Graph Generated: {len(final_graph.nodes())} nodes, {len(final_graph.edges())} edges")
        print(f"   Tasks: {len(tasks_result.get('task_breakdown', []))} actionable tasks")
        print(f"   QAQC: {qaqc_result.get('overall_quality_score', 0.0):.2f}/1.0 quality, {'PASSED' if qaqc_result.get('validation_passed') else 'NEEDS REVIEW'}")
        
        # Graph insights analysis
        print(f"\n🔍 GRAPH INSIGHTS ANALYSIS")
        print("-" * 50)
        node_types = {}
        relationship_types = {}
        
        for node_id in final_graph.nodes():
            node_type = final_graph.get_node_attributes(node_id).get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        for edge in final_graph.edges():
            source, target = edge
            relationship = final_graph.get_edge_attributes(source, target).get('relationship', 'unknown')
            relationship_types[relationship] = relationship_types.get(relationship, 0) + 1
        
        print(f"   Nodes: {len(final_graph.nodes())}")
        print(f"   Edges: {len(final_graph.edges())}")
        print(f"   Node Types: {node_types}")
        print(f"   Relationships: {relationship_types}")
        
        return complete_result
        
    except Exception as e:
        print(f"❌ Intent analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(run_intent_analysis_demo())
