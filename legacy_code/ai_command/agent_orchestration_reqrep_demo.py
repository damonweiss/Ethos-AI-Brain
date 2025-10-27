"""
REQ-REP Agent Orchestration Demo
Focused demo of the core REQ-REP pattern: Supervisor with specialized advisor agents
"""

import asyncio
from agent_orchestration_reqrep import ReqRepOrchestrator
from zmq_node_base import create_llm_processor, create_custom_processor


def budget_calculator(budget_data: dict) -> dict:
    """Budget analysis logic"""
    budget = budget_data.get("budget", 0)
    requirements = budget_data.get("requirements", [])
    base_cost = len(requirements) * 5000  # $5k per requirement
    
    if budget >= base_cost * 1.2:
        feasibility = "excellent"
    elif budget >= base_cost:
        feasibility = "good"
    else:
        feasibility = "challenging"
    
    return {
        "estimated_cost": base_cost,
        "budget_provided": budget,
        "feasibility": feasibility,
        "recommendation": f"Budget analysis: {feasibility} feasibility for {len(requirements)} requirements"
    }


def technical_validator(tech_spec: dict) -> dict:
    """Technical specification validation"""
    requirements = tech_spec.get("requirements", [])
    complexity_score = len(requirements) * 2
    risk_level = "low" if complexity_score <= 5 else "medium" if complexity_score <= 10 else "high"
    
    return {
        "complexity_score": complexity_score,
        "risk_level": risk_level,
        "validation_result": f"Technical validation: {risk_level} risk, complexity score {complexity_score}"
    }


async def main():
    """
    Core REQ-REP Demo: Supervisor with Expert Advisors
    Shows the fundamental pattern without repetition
    """
    print("🚀 REQ-REP Agent Orchestration Demo")
    print("=" * 50)
    print("Supervisor coordinates with specialized advisor agents via REQ-REP")
    
    try:
        # Create supervisor with LLM coordination
        supervisor = ReqRepOrchestrator("project_supervisor", base_port=7000)
        supervisor.add_processor(create_llm_processor("supervisor_llm", "gpt-4"))
        
        await supervisor.start()
        
        # Spawn specialized advisor agents
        print("\n🐣 Spawning Expert Advisors...")
        
        await supervisor.spawn_child_agent(
            child_id="budget_advisor",
            role="financial_analysis", 
            processors=[create_custom_processor("budget_calc", budget_calculator)]
        )
        
        await supervisor.spawn_child_agent(
            child_id="tech_advisor",
            role="technical_validation",
            processors=[create_custom_processor("tech_validator", technical_validator)]
        )
        
        # Real-world scenario: POS system analysis
        print("\n💼 Analyzing POS System Requirements...")
        
        project_requirements = {
            "type": "pos_system_analysis",
            "budget": 25000,
            "requirements": ["payment_processing", "inventory_management", "reporting", "multi_location"]
        }
        
        # Request expert advice via REQ-REP
        budget_advice = await supervisor.delegate_task_to_child("budget_advisor", {
            "type": "budget_analysis",
            "data": project_requirements
        })
        
        tech_advice = await supervisor.delegate_task_to_child("tech_advisor", {
            "type": "technical_validation", 
            "data": project_requirements
        })
        
        # Display results
        print(f"\n💰 Budget Analysis: {budget_advice.get('result', {}).get('recommendation', 'N/A')}")
        print(f"🔧 Technical Analysis: {tech_advice.get('result', {}).get('validation_result', 'N/A')}")
        
        await supervisor.stop()
        
        print("\n✅ REQ-REP Demo Complete!")
        print("📋 Pattern Demonstrated: Supervisor coordinates specialized agents via synchronous REQ-REP")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
