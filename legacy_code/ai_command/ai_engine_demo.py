"""
AI Engine Demo
Demonstrates the minimalist AI Engine with AgentZero
"""

import asyncio
from ai_engine import initialize_ai_engine


async def main():
    """Demo the AI Engine with AgentZero"""
    print("🚀 AI Engine Demo")
    print("=" * 50)
    print("Initializing AI Engine with AgentZero...")
    
    try:
        # Initialize AI Engine (creates ZMQ engine + AgentZero)
        engine = await initialize_ai_engine()
        
        print(f"✅ AI Engine initialized")
        print(f"🤖 AgentZero status: {'Active' if engine.agent_zero else 'Not Available'}")
        
        if engine.agent_zero:
            print(f"   Agent ID: {engine.agent_zero.agent_id}")
            print(f"   Role: {engine.agent_zero.role}")
            print(f"   Capabilities: {', '.join(engine.agent_zero.capabilities)}")
            print(f"   Running: {engine.agent_zero.is_running}")
            print(f"   Brain ID: {engine.agent_zero.brain.brain_id}")
            print(f"   Brain ZMQ Engine: {'Shared' if engine.agent_zero.brain.shared_zmq_engine else 'None'}")
        
        # Test task delegation to AgentZero
        print("\n💼 Testing AgentZero Task Delegation...")
        
        test_task = {
            "type": "business_analysis",
            "request": "Analyze requirements for a restaurant POS system",
            "context": {
                "business_type": "restaurant",
                "size": "medium",
                "budget": 25000
            }
        }
        
        print(f"📤 Delegating task to AgentZero: {test_task['type']}")
        
        # Delegate to AgentZero
        result = await engine.delegate_to_agent_zero(test_task)
        
        print(f"📥 AgentZero response: {result.get('status', 'unknown')}")
        
        if result.get('result'):
            print(f"🎯 Result: {result['result']}")
        
        # Test AgentZero's pattern spawning capability
        print("\n🔧 Testing AgentZero Pattern Spawning...")
        
        from agent_orchestration_reqrep import ReqRepOrchestrator
        from agent_orchestration_base import OrchestrationPattern
        
        # AgentZero spawns a REQ-REP orchestration pattern
        req_rep_pattern = engine.agent_zero.spawn_agent_pattern(
            ReqRepOrchestrator,
            "advisor_coordination"
        )
        
        print(f"🔧 Pattern spawned: {type(req_rep_pattern).__name__}")
        print(f"   Pattern ID: {req_rep_pattern.agent_id}")
        print(f"   Pattern Type: {req_rep_pattern.orchestration_pattern.value}")
        print(f"   Shared ZMQ Engine: {'Yes' if req_rep_pattern.zmq_engine else 'No'}")
        
        # Start and execute the pattern
        print("\n⚡ Starting and executing the pattern...")
        await req_rep_pattern.start()
        
        # Execute a task through the pattern
        pattern_task = {
            "type": "coordination_test",
            "request": "Test REQ-REP coordination pattern",
            "data": {"test": "pattern execution"}
        }
        
        pattern_result = await req_rep_pattern.execute_task(pattern_task)
        print(f"⚡ Pattern execution result: {pattern_result.get('status', 'unknown')}")
        
        if pattern_result.get('result'):
            print(f"🎯 Pattern result: {pattern_result['result']}")
        
        # Stop the pattern
        await req_rep_pattern.stop()
        
        # Test AgentZero's brain
        print(f"\n🧠 Testing AgentZero's Brain...")
        print(f"   Brain Type: {type(engine.agent_zero.brain).__name__}")
        print(f"   Brain ID: {engine.agent_zero.brain.brain_id}")
        print(f"   Brain has ZMQ Engine: {'Yes' if engine.agent_zero.brain.shared_zmq_engine else 'No'}")
        print(f"   Knowledge Graph Brain: {type(engine.agent_zero.brain.knowledge_graph_brain).__name__}")
        print(f"   Reasoning Engine: {type(engine.agent_zero.brain.reasoning_engine).__name__}")
        
        # Test AgentZero's reasoning capabilities
        print(f"\n🤔 Testing AgentZero's Reasoning...")
        
        reasoning_task = "I want to increase sales at my engineering firm via social media."
        reasoning_context = {
            "constraints": {
                "budget": 25000,
                "timeline": "6 weeks",
                "team_size": 3
            },
            "preferences": {
                "tech_stack": "modern",
                "scalability": "medium",
                "deployment": "cloud"
            }
        }
        
        print(f"🎯 Reasoning Task: {reasoning_task}")
        print(f"📋 Context: Budget ${reasoning_context['constraints']['budget']}, {reasoning_context['constraints']['timeline']} timeline")
        
        reasoning_result = await engine.agent_zero.think_about(reasoning_task, reasoning_context)
        
        print(f"🧠 Reasoning Result:")
        print(f"   Status: {reasoning_result.get('status', 'unknown')}")
        print(f"   Confidence: {reasoning_result.get('confidence', 'unknown')}")
        
        if reasoning_result.get('synthesis'):
            print(f"   Synthesis: {reasoning_result['synthesis'][:100]}...")
        
        if reasoning_result.get('execution_summary'):
            summary = reasoning_result['execution_summary']
            print(f"   Steps: {summary.get('successful_steps', 0)}/{summary.get('total_steps', 0)} completed")
        
        # Test AgentZero's adaptive intent analysis
        print(f"\n🎯 Testing AgentZero's Intent Analysis...")
        
        user_request = "Help me plan a wedding."
        
        print(f"📝 User Request: {user_request}")
        
        intent_result_type, intent_data = await engine.agent_zero.analyze_user_request(user_request)
        
        print(f"🎯 Intent Analysis Result: {intent_result_type}")
        
        if intent_result_type == "complex_analysis":
            if isinstance(intent_data, dict):
                graph = intent_data.get('graph')
                tasks = intent_data.get('tasks')
                qaqc = intent_data.get('qaqc')
                
                if graph:
                    print(f"   Knowledge Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
                if tasks and tasks.get('task_breakdown'):
                    print(f"   Tasks Identified: {len(tasks['task_breakdown'])} actionable tasks")
                if qaqc:
                    print(f"   Quality Score: {qaqc.get('overall_quality_score', 'N/A')}")
            else:
                print(f"   Graph: {len(intent_data.nodes())} nodes, {len(intent_data.edges())} edges")
        else:
            print(f"   Direct Response: {intent_data[:100]}..." if isinstance(intent_data, str) else intent_data)
        
        # Show knowledge graphs in brain
        print(f"\n📊 Knowledge Graphs in Brain:")
        brain_graphs = engine.agent_zero.brain.knowledge_graph_brain.list_graphs()
        if brain_graphs:
            for graph_id, z_index, graph_type in brain_graphs:
                graph = engine.agent_zero.brain.knowledge_graph_brain.get_graph(graph_id)
                print(f"   - {graph_id}: {len(graph.nodes())} nodes, {len(graph.edges())} edges ({graph_type}, z={z_index})")
        else:
            print("   No knowledge graphs stored yet")
        
        # Show AgentZero as AIAgent instance
        print(f"\n🤖 AgentZero is an AIAgent instance:")
        print(f"   Type: {type(engine.agent_zero).__name__}")
        print(f"   Base Port: {engine.agent_zero.base_port}")
        print(f"   ZMQ Engine: {'Shared' if engine.agent_zero.shared_zmq_engine else 'None'}")
        print(f"   Can spawn patterns: {'Yes' if hasattr(engine.agent_zero, 'spawn_agent_pattern') else 'No'}")
        print(f"   Has brain: {'Yes' if hasattr(engine.agent_zero, 'brain') else 'No'}")
        
        # Shutdown
        print("\n🛑 Shutting down AI Engine...")
        await engine.shutdown()
        
        print("✅ AI Engine Demo Complete!")
        print("📋 Demonstrated: ZMQ Engine + AgentZero + Task Delegation + Brain Reasoning")
        print("🧠 AgentZero now has:")
        print("   - Knowledge Graph Management")
        print("   - Meta Reasoning Engine")
        print("   - Adaptive Intent Analysis")
        print("   - Pattern Spawning Capabilities")
        print("   - Shared ZMQ Infrastructure")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
