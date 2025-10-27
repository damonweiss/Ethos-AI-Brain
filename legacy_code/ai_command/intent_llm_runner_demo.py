#!/usr/bin/env python3
"""
Intent LLM Runner Demo
Demonstrates the clean intent_llm_runner.py API for transforming prompts into intent graphs
"""

import asyncio
import time
from intent_llm_runner import IntentLLMRunner
from knowledge_graph_visualizer import MatplotlibKnowledgeGraphVisualizer

async def demo_intent_llm_runner():
    """Demonstrate Intent LLM Runner with 6-parallel processing"""
    
    print("INTENT LLM RUNNER DEMO")
    print("=" * 70)
    print("🚀 Clean API for transforming prompts into intent knowledge graphs")
    print("⚡ Using 6-parallel ZMQ ROUTER-DEALER architecture")
    print("📊 Includes intelligent linking and visualization")
    print("=" * 70)
    
    # Test case - Restaurant POS system
    user_input = "I need a new POS system for my restaurant that handles orders, payments, and inventory with a $25k budget in 6 weeks"
    domain = "restaurant_management"
    
    print(f"\n📝 USER INPUT:")
    print(f"   Request: {user_input}")
    print(f"   Domain: {domain}")
    
    # Create Intent LLM Runner with 6-parallel processing
    print(f"\n🏗️ INITIALIZING INTENT LLM RUNNER")
    runner = IntentLLMRunner(base_port=5800, merge_strategy="multi_port")
    
    try:
        # Transform user input into completed intent graph
        print(f"\n🚀 EXECUTING INTENT ANALYSIS")
        start_time = time.time()
        
        intent_graph = await runner.analyze_intent(user_input, domain)
        
        analysis_time = time.time() - start_time
        
        # Display results
        print(f"\n📊 ANALYSIS RESULTS")
        print("-" * 40)
        print(f"   Analysis Time: {analysis_time:.2f} seconds")
        print(f"   Graph Nodes: {len(intent_graph.nodes())}")
        print(f"   Graph Edges: {len(intent_graph.edges())}")
        print(f"   Domain Context: {intent_graph.domain_context}")
        print(f"   User Sentiment: {intent_graph.user_sentiment}")
        
        # Detailed breakdown
        print(f"\n🔍 DETAILED BREAKDOWN")
        print("-" * 40)
        
        # Count nodes by type
        node_types = {}
        for node_id in intent_graph.nodes():
            node_attrs = intent_graph.get_node_attributes(node_id)
            node_type = node_attrs.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        for node_type, count in sorted(node_types.items()):
            print(f"   {node_type.replace('_', ' ').title()}: {count}")
        
        # Show sample nodes
        print(f"\n📋 SAMPLE NODES")
        print("-" * 40)
        
        sample_count = 0
        for node_id in intent_graph.nodes():
            if sample_count >= 6:  # Show first 6 nodes
                break
                
            node_attrs = intent_graph.get_node_attributes(node_id)
            node_type = node_attrs.get('type', 'unknown')
            node_name = node_attrs.get('name', 'Unknown')
            
            print(f"   {node_id}: {node_type} - '{node_name}'")
            sample_count += 1
        
        if len(intent_graph.nodes()) > 6:
            print(f"   ... and {len(intent_graph.nodes()) - 6} more nodes")
        
        # Show sample edges
        print(f"\n🔗 SAMPLE RELATIONSHIPS")
        print("-" * 40)
        
        edge_count = 0
        for edge in intent_graph.edges():
            if edge_count >= 8:  # Show first 8 edges
                break
                
            source_attrs = intent_graph.get_node_attributes(edge[0])
            target_attrs = intent_graph.get_node_attributes(edge[1])
            edge_attrs = intent_graph.get_edge_attributes(edge[0], edge[1])
            
            source_name = source_attrs.get('name', edge[0])[:20]
            target_name = target_attrs.get('name', edge[1])[:20]
            relationship = edge_attrs.get('relationship', 'related_to')
            
            print(f"   {source_name} --{relationship}--> {target_name}")
            edge_count += 1
        
        if len(intent_graph.edges()) > 8:
            print(f"   ... and {len(intent_graph.edges()) - 8} more relationships")
        
        # Performance analysis
        print(f"\n⚡ PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        components_analyzed = sum(node_types.values())
        components_per_second = components_analyzed / analysis_time if analysis_time > 0 else 0
        
        print(f"   Total Components: {components_analyzed}")
        print(f"   Processing Speed: {components_per_second:.1f} components/second")
        print(f"   Architecture: 6-parallel ZMQ ROUTER-DEALER")
        print(f"   LLM Calls: 6 simultaneous GPT-4 requests")
        
        # Graph quality metrics
        print(f"\n📈 GRAPH QUALITY METRICS")
        print("-" * 40)
        
        # Calculate connectivity
        total_possible_edges = len(intent_graph.nodes()) * (len(intent_graph.nodes()) - 1) // 2
        connectivity_ratio = len(intent_graph.edges()) / total_possible_edges if total_possible_edges > 0 else 0
        
        print(f"   Node Density: {len(intent_graph.nodes())} nodes")
        print(f"   Edge Density: {len(intent_graph.edges())} edges")
        print(f"   Connectivity: {connectivity_ratio:.2%}")
        print(f"   Avg Connections: {len(intent_graph.edges()) * 2 / len(intent_graph.nodes()):.1f} per node")
        
        # Check for isolated nodes
        isolated_nodes = []
        for node_id in intent_graph.nodes():
            if intent_graph.degree(node_id) == 0:
                isolated_nodes.append(node_id)
        
        if isolated_nodes:
            print(f"   ⚠️ Isolated Nodes: {len(isolated_nodes)}")
        else:
            print(f"   ✅ No Isolated Nodes: Fully connected graph")
        
        # Feasibility assessment
        print(f"\n🎯 FEASIBILITY ASSESSMENT")
        print("-" * 40)
        
        # Simple feasibility scoring based on constraints vs objectives
        objectives = [n for n in intent_graph.nodes() 
                     if intent_graph.get_node_attributes(n).get('type') == 'primary_objective']
        constraints = [n for n in intent_graph.nodes() 
                      if intent_graph.get_node_attributes(n).get('type') == 'constraint']
        assumptions = [n for n in intent_graph.nodes() 
                      if intent_graph.get_node_attributes(n).get('type') == 'assumption']
        
        # Count critical/high priority items
        critical_objectives = 0
        for obj_id in objectives:
            obj_attrs = intent_graph.get_node_attributes(obj_id)
            if obj_attrs.get('priority') in ['critical', 'high']:
                critical_objectives += 1
        
        feasibility_score = min(1.0, (len(objectives) - len(constraints) * 0.3) / max(len(objectives), 1))
        feasibility_score = max(0.0, feasibility_score)
        
        print(f"   Objectives: {len(objectives)} ({critical_objectives} high/critical)")
        print(f"   Constraints: {len(constraints)}")
        print(f"   Assumptions: {len(assumptions)}")
        print(f"   Feasibility Score: {feasibility_score:.2f} ({feasibility_score*100:.0f}%)")
        
        if feasibility_score >= 0.8:
            print(f"   ✅ HIGH FEASIBILITY - Well-defined and achievable")
        elif feasibility_score >= 0.6:
            print(f"   ⚠️ MODERATE FEASIBILITY - Some challenges expected")
        else:
            print(f"   ❌ LOW FEASIBILITY - Significant constraints or gaps")
        
        # 3D Visualization
        print(f"\n🎨 3D VISUALIZATION")
        print("-" * 40)
        
        visualizer = MatplotlibKnowledgeGraphVisualizer(intent_graph)
        print("   Rendering 3D knowledge graph...")
        
        visualizer.render_3d_graph(
            figsize=(14, 10),
            show_labels=True,
            positioning_strategy="layered",
            save_path=None
        )
        
        print("   ✅ 3D visualization displayed")
        
        # Summary
        print(f"\n🏁 DEMO SUMMARY")
        print("=" * 50)
        print(f"✅ Successfully transformed user prompt into rich intent graph")
        print(f"⚡ Processing time: {analysis_time:.2f} seconds")
        print(f"📊 Graph complexity: {len(intent_graph.nodes())} nodes, {len(intent_graph.edges())} edges")
        print(f"🔗 Intelligent linking: Auto-linking + selective stakeholder connections")
        print(f"🎯 Feasibility: {feasibility_score*100:.0f}% - {'High' if feasibility_score >= 0.8 else 'Moderate' if feasibility_score >= 0.6 else 'Low'}")
        print(f"🚀 Architecture: 6-parallel ZMQ ROUTER-DEALER with multi-port servers")
        
        return intent_graph
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise
    
    finally:
        # Runner automatically cleans up servers
        print(f"\n🧹 Cleanup completed automatically")

async def demo_comparison():
    """Compare different merge strategies"""
    
    print("\n" + "=" * 70)
    print("MERGE STRATEGY COMPARISON")
    print("=" * 70)
    
    user_input = "Build a mobile app for food delivery with real-time tracking and $50k budget"
    domain = "mobile_development"
    
    strategies = ["single_call", "multi_port"]
    results = {}
    
    for strategy in strategies:
        print(f"\n🔥 Testing {strategy.upper()} strategy...")
        
        runner = IntentLLMRunner(base_port=5900 + strategies.index(strategy) * 10, merge_strategy=strategy)
        
        start_time = time.time()
        intent_graph = await runner.analyze_intent(user_input, domain)
        execution_time = time.time() - start_time
        
        results[strategy] = {
            'time': execution_time,
            'nodes': len(intent_graph.nodes()),
            'edges': len(intent_graph.edges())
        }
        
        print(f"   ✅ {strategy}: {execution_time:.2f}s, {len(intent_graph.nodes())} nodes, {len(intent_graph.edges())} edges")
    
    # Comparison
    print(f"\n📊 STRATEGY COMPARISON RESULTS")
    print("-" * 50)
    
    for strategy, data in results.items():
        print(f"   {strategy.upper():<12}: {data['time']:.2f}s | {data['nodes']:2d} nodes | {data['edges']:2d} edges")
    
    # Winner
    fastest = min(results.keys(), key=lambda k: results[k]['time'])
    richest = max(results.keys(), key=lambda k: results[k]['nodes'] + results[k]['edges'])
    
    print(f"\n🏆 WINNERS:")
    print(f"   Fastest: {fastest.upper()} ({results[fastest]['time']:.2f}s)")
    print(f"   Richest Graph: {richest.upper()} ({results[richest]['nodes']} nodes, {results[richest]['edges']} edges)")

if __name__ == "__main__":
    print("Starting Intent LLM Runner Demo...")
    
    # Run main demo
    asyncio.run(demo_intent_llm_runner())
    
    # Run comparison demo
    asyncio.run(demo_comparison())
