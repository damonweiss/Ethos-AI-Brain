#!/usr/bin/env python3
"""
Adaptive Intent Analysis Demo
Showcases intelligent heuristic LLM pattern with complexity assessment and stakeholder expansion
"""

import asyncio
import time
from adaptive_intent_runner import AdaptiveIntentRunner
from knowledge_graph_visualizer import MatplotlibKnowledgeGraphVisualizer

async def demo_adaptive_intent_analysis():
    """Demonstrate adaptive intent analysis with various complexity scenarios"""
    
    print("ADAPTIVE INTENT ANALYSIS DEMO")
    print("=" * 80)
    print("🧠 Intelligent heuristic pattern: complexity assessment + stakeholder expansion")
    print("⚡ Adaptive processing: simple requests get direct answers, complex get full analysis")
    print("🎯 Stakeholder-centric: dynamic scaling based on actual stakeholder-objective pairs")
    print("=" * 80)
    
    # Create adaptive runner
    runner = AdaptiveIntentRunner(base_port=6100)
    
    # Test scenarios with different complexity levels
    scenarios = [
        {
            "name": "Simple Factual Query",
            "input": "What is the capital of France?",
            "domain": "geography",
            "expected_complexity": "simple"
        },
        {
            "name": "Simple How-To Request", 
            "input": "How do I reset my password on Gmail?",
            "domain": "technical_support",
            "expected_complexity": "simple"
        },
        {
            "name": "Complex Business Project",
            "input": "I need to launch a food delivery app in my city with real-time tracking, multiple restaurant partnerships, and a $100k budget over 8 months",
            "domain": "mobile_app_development",
            "expected_complexity": "complex"
        },
        {
            "name": "Complex Restaurant System",
            "input": "I need a new POS system for my restaurant that handles orders, payments, and inventory with a $25k budget in 6 weeks",
            "domain": "restaurant_management", 
            "expected_complexity": "complex"
        },
        {
            "name": "Complex Corporate Initiative",
            "input": "We need to implement a company-wide remote work policy with new collaboration tools, security protocols, and performance metrics",
            "domain": "corporate_policy",
            "expected_complexity": "complex"
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*80}")
        print(f"SCENARIO {i}: {scenario['name'].upper()}")
        print(f"{'='*80}")
        
        print(f"📝 REQUEST: {scenario['input']}")
        print(f"🏷️ DOMAIN: {scenario['domain']}")
        print(f"🎯 EXPECTED: {scenario['expected_complexity']}")
        
        # Execute adaptive analysis
        start_time = time.time()
        
        try:
            result_type, result = await runner.analyze_intent(
                scenario["input"], 
                scenario["domain"]
            )
            
            execution_time = time.time() - start_time
            
            # Store results
            scenario_result = {
                "scenario": scenario["name"],
                "result_type": result_type,
                "execution_time": execution_time,
                "expected": scenario["expected_complexity"]
            }
            
            if result_type == "direct_answer":
                scenario_result["answer"] = result
                print(f"\n📋 DIRECT ANSWER:")
                print(f"   {result}")
                
            else:  # complex_analysis
                scenario_result["nodes"] = len(result.nodes())
                scenario_result["edges"] = len(result.edges())
                scenario_result["graph"] = result
                
                print(f"\n📊 COMPLEX ANALYSIS RESULTS:")
                print(f"   Graph Nodes: {len(result.nodes())}")
                print(f"   Graph Edges: {len(result.edges())}")
                print(f"   Domain: {result.domain_context}")
                print(f"   Sentiment: {result.user_sentiment}")
                
                # Show stakeholder breakdown
                stakeholders = [n for n in result.nodes() 
                              if result.get_node_attributes(n).get('type') == 'stakeholder']
                objectives = [n for n in result.nodes() 
                            if result.get_node_attributes(n).get('type') == 'primary_objective']
                constraints = [n for n in result.nodes() 
                             if result.get_node_attributes(n).get('type') == 'constraint']
                
                print(f"\n🔍 COMPONENT BREAKDOWN:")
                print(f"   Stakeholders: {len(stakeholders)}")
                print(f"   Objectives: {len(objectives)}")
                print(f"   Constraints: {len(constraints)}")
                
                # Show sample stakeholders and their objectives
                if stakeholders:
                    print(f"\n👥 STAKEHOLDERS IDENTIFIED:")
                    for stake_id in stakeholders[:3]:  # Show first 3
                        stake_attrs = result.get_node_attributes(stake_id)
                        stake_name = stake_attrs.get('name', 'Unknown')
                        stake_role = stake_attrs.get('role', 'participant')
                        
                        # Count objectives connected to this stakeholder
                        connected_objectives = 0
                        for obj_id in objectives:
                            if result.has_edge(stake_id, obj_id):
                                connected_objectives += 1
                        
                        print(f"   - {stake_name} ({stake_role}): {connected_objectives} objectives")
            
            print(f"\n⏱️ PERFORMANCE:")
            print(f"   Execution Time: {execution_time:.2f} seconds")
            print(f"   Result Type: {result_type}")
            print(f"   Complexity Match: {'✅' if (result_type == 'direct_answer') == (scenario['expected_complexity'] == 'simple') else '⚠️'}")
            
            results.append(scenario_result)
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            scenario_result = {
                "scenario": scenario["name"],
                "result_type": "error",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
            results.append(scenario_result)
    
    # Overall summary
    print(f"\n{'='*80}")
    print("ADAPTIVE ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    simple_count = sum(1 for r in results if r.get("result_type") == "direct_answer")
    complex_count = sum(1 for r in results if r.get("result_type") == "complex_analysis")
    error_count = sum(1 for r in results if r.get("result_type") == "error")
    
    total_time = sum(r.get("execution_time", 0) for r in results)
    avg_time = total_time / len(results) if results else 0
    
    print(f"📊 PROCESSING RESULTS:")
    print(f"   Total Scenarios: {len(scenarios)}")
    print(f"   Simple Responses: {simple_count}")
    print(f"   Complex Analyses: {complex_count}")
    print(f"   Errors: {error_count}")
    
    print(f"\n⏱️ PERFORMANCE METRICS:")
    print(f"   Total Time: {total_time:.2f} seconds")
    print(f"   Average Time: {avg_time:.2f} seconds per scenario")
    
    # Show complexity detection accuracy
    correct_predictions = 0
    for result in results:
        if result.get("result_type") == "error":
            continue
        
        scenario_name = result["scenario"]
        scenario_data = next(s for s in scenarios if s["name"] == scenario_name)
        expected_simple = scenario_data["expected_complexity"] == "simple"
        actual_simple = result["result_type"] == "direct_answer"
        
        if expected_simple == actual_simple:
            correct_predictions += 1
    
    accuracy = correct_predictions / (len(results) - error_count) if (len(results) - error_count) > 0 else 0
    
    print(f"\n🎯 COMPLEXITY DETECTION:")
    print(f"   Accuracy: {accuracy:.1%} ({correct_predictions}/{len(results) - error_count})")
    
    # Detailed breakdown
    print(f"\n📋 DETAILED BREAKDOWN:")
    print(f"{'Scenario':<25} {'Type':<15} {'Time':<8} {'Details':<30}")
    print("-" * 80)
    
    for result in results:
        scenario_name = result["scenario"][:24]
        result_type = result.get("result_type", "unknown")[:14]
        exec_time = f"{result.get('execution_time', 0):.2f}s"
        
        if result_type == "direct_answer":
            details = "Direct answer provided"
        elif result_type == "complex_analysis":
            nodes = result.get("nodes", 0)
            edges = result.get("edges", 0)
            details = f"{nodes} nodes, {edges} edges"
        else:
            details = "Error occurred"
        
        print(f"{scenario_name:<25} {result_type:<15} {exec_time:<8} {details:<30}")
    
    # Visualize one complex graph
    complex_results = [r for r in results if r.get("result_type") == "complex_analysis"]
    if complex_results:
        print(f"\n🎨 VISUALIZING COMPLEX GRAPH")
        print("-" * 40)
        
        # Pick the most interesting complex result (largest graph)
        best_result = max(complex_results, key=lambda r: r.get("nodes", 0) + r.get("edges", 0))
        
        print(f"Visualizing: {best_result['scenario']}")
        print(f"Graph: {best_result['nodes']} nodes, {best_result['edges']} edges")
        
        visualizer = MatplotlibKnowledgeGraphVisualizer(best_result["graph"])
        visualizer.render_3d_graph(
            figsize=(14, 10),
            show_labels=True,
            positioning_strategy="layered"
        )
        
        print("✅ 3D visualization displayed")
    
    print(f"\n🏁 ADAPTIVE INTENT DEMO COMPLETE")
    print("=" * 50)
    print("✅ Demonstrated intelligent complexity assessment")
    print("⚡ Showed adaptive processing based on stakeholder structure")
    print("🎯 Validated heuristic LLM pattern effectiveness")
    
    return results

if __name__ == "__main__":
    print("Starting Adaptive Intent Analysis Demo...")
    asyncio.run(demo_adaptive_intent_analysis())
