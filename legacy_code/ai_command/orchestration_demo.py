"""
Agent Orchestration Demo
Shows hierarchical agent coordination with ZMQ + LLM integration
"""

import asyncio
import time
from typing import Dict, Any

from agent_orchestration_base import AgentOrchestrationBase, OrchestrationPattern, AgentCapability
from orchestration_patterns import (
    SupervisorOrchestrator,
    ScatterGatherOrchestrator, 
    PipelineOrchestrator,
    EventDrivenOrchestrator,
    HierarchicalOrchestrator,
    create_orchestrator
)


class SpecializedAgent(AgentOrchestrationBase):
    """
    Example specialized agent (leaf node in hierarchy)
    Represents actual work-performing agents
    """
    
    def __init__(self, agent_id: str, specialization: str, **kwargs):
        capabilities = [
            AgentCapability(
                name=f"{specialization}_analysis",
                description=f"Specialized {specialization} analysis and recommendations",
                input_schema={"task": "dict", "context": "dict"},
                output_schema={"analysis": "dict", "recommendations": "list"},
                llm_prompt_template=f"Analyze from {specialization} perspective: {{task}}"
            )
        ]
        
        super().__init__(
            agent_id=agent_id,
            orchestration_pattern=OrchestrationPattern.NETWORK,  # Leaf agents use network pattern
            capabilities=capabilities,
            **kwargs
        )
        self.specialization = specialization
    
    async def _start_orchestration(self):
        """Start specialized agent"""
        print(f"🔧 Specialized Agent {self.agent_id} ({self.specialization}) ready")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using specialized knowledge"""
        # Use LLM brain with specialization context
        analysis = await self.brain.think(
            f"Analyze task from {self.specialization} perspective: {task}",
            context={
                "specialization": self.specialization,
                "agent_id": self.agent_id,
                "task_type": task.get("type", "general")
            },
            specialization=self.specialization
        )
        
        # Simulate some processing time
        await asyncio.sleep(0.5)
        
        return {
            "agent_id": self.agent_id,
            "specialization": self.specialization,
            "analysis": analysis,
            "processing_time": 0.5,
            "confidence": 0.85
        }
    
    async def _route_message_to_child(self, child_id: str, message) -> Dict[str, Any]:
        """Leaf agents don't route to children"""
        return {"error": "Leaf agent has no children"}


async def demo_restaurant_pos_orchestration():
    """
    Demo: Restaurant POS System Analysis
    Shows hierarchical orchestration with different patterns
    """
    
    print("🏗️ Setting up Restaurant POS Analysis Orchestration")
    print("=" * 60)
    
    # Create Master Orchestrator (Hierarchical)
    master_orchestrator = HierarchicalOrchestrator(
        agent_id="master_pos_orchestrator",
        base_port=6000
    )
    
    # Create Domain Supervisors
    technical_supervisor = SupervisorOrchestrator(
        agent_id="technical_supervisor",
        parent_agent=master_orchestrator,
        base_port=6100
    )
    
    business_supervisor = SupervisorOrchestrator(
        agent_id="business_supervisor", 
        parent_agent=master_orchestrator,
        base_port=6200
    )
    
    # Create Expert Panel (Scatter-Gather)
    expert_panel = ScatterGatherOrchestrator(
        agent_id="expert_panel",
        parent_agent=business_supervisor,
        base_port=6300
    )
    
    # Create Analysis Pipeline
    analysis_pipeline = PipelineOrchestrator(
        agent_id="analysis_pipeline",
        parent_agent=technical_supervisor,
        pipeline_stages=["requirements", "architecture", "implementation", "testing"],
        base_port=6400
    )
    
    # Create Specialized Agents
    specialized_agents = [
        SpecializedAgent("restaurant_ops_expert", "restaurant_operations", base_port=6500),
        SpecializedAgent("tech_architect", "technical_architecture", base_port=6510),
        SpecializedAgent("finance_analyst", "financial_analysis", base_port=6520),
        SpecializedAgent("ux_designer", "user_experience", base_port=6530),
        SpecializedAgent("security_expert", "cybersecurity", base_port=6540),
        SpecializedAgent("requirements_analyst", "requirements_analysis", base_port=6550),
        SpecializedAgent("system_architect", "system_architecture", base_port=6560),
        SpecializedAgent("implementation_lead", "implementation_planning", base_port=6570),
        SpecializedAgent("qa_engineer", "quality_assurance", base_port=6580)
    ]
    
    # Build Hierarchy
    print("\n🏢 Building Agent Hierarchy:")
    
    # Add domain supervisors to master
    master_orchestrator.add_child_agent(technical_supervisor)
    master_orchestrator.add_child_agent(business_supervisor)
    
    # Add expert panel to business supervisor
    business_supervisor.add_child_agent(expert_panel)
    
    # Add analysis pipeline to technical supervisor
    technical_supervisor.add_child_agent(analysis_pipeline)
    
    # Add specialized agents to appropriate supervisors
    expert_agents = specialized_agents[:5]  # First 5 for expert panel
    pipeline_agents = specialized_agents[5:]  # Last 4 for pipeline
    
    for agent in expert_agents:
        expert_panel.add_child_agent(agent)
        print(f"  👥 Added {agent.agent_id} to expert panel")
    
    for agent in pipeline_agents:
        analysis_pipeline.add_child_agent(agent)
        print(f"  🔄 Added {agent.agent_id} to analysis pipeline")
    
    # Start all orchestrators
    print("\n🚀 Starting Orchestration System:")
    
    start_tasks = []
    for agent in specialized_agents:
        start_tasks.append(agent.start())
    
    start_tasks.extend([
        analysis_pipeline.start(),
        expert_panel.start(),
        technical_supervisor.start(),
        business_supervisor.start(),
        master_orchestrator.start()
    ])
    
    await asyncio.gather(*start_tasks)
    
    # Display hierarchy
    print("\n🌳 Agent Hierarchy:")
    hierarchy = master_orchestrator.get_agent_hierarchy()
    print_hierarchy(hierarchy)
    
    # Execute complex task
    print("\n📋 Executing Complex POS Analysis Task:")
    
    complex_task = {
        "id": "pos_system_analysis",
        "type": "comprehensive_analysis",
        "description": "Analyze requirements for restaurant POS system",
        "context": {
            "budget": 25000,
            "timeline": "6 weeks",
            "restaurant_type": "casual_dining",
            "covers_per_day": 200,
            "staff_count": 15
        },
        "requirements": [
            "order_management",
            "payment_processing", 
            "inventory_tracking",
            "staff_management",
            "reporting_analytics"
        ]
    }
    
    start_time = time.time()
    
    # Execute through master orchestrator
    result = await master_orchestrator.execute_task(complex_task)
    
    execution_time = time.time() - start_time
    
    print(f"\n✅ Task Execution Complete ({execution_time:.2f}s)")
    print("📊 Results Summary:")
    print(f"  Status: {result.get('status', 'unknown')}")
    print(f"  Pattern: {result.get('pattern', 'hierarchical')}")
    
    if result.get('result'):
        print("  Key Findings:")
        for key, value in result['result'].items():
            if isinstance(value, dict) and 'analysis' in value:
                print(f"    • {key}: {value['analysis'].get('summary', 'Analysis completed')}")
    
    # Demonstrate different orchestration patterns
    print("\n🎯 Demonstrating Different Orchestration Patterns:")
    
    # 1. Expert Panel (Scatter-Gather)
    print("\n1️⃣ Expert Panel Analysis (Scatter-Gather):")
    expert_task = {
        "type": "expert_consultation",
        "question": "What are the critical success factors for POS implementation?",
        "context": complex_task["context"]
    }
    
    expert_result = await expert_panel.execute_task(expert_task)
    print(f"   Expert Panel Status: {expert_result.get('status')}")
    print(f"   Experts Consulted: {len(expert_result.get('individual_results', {}))}")
    
    # 2. Analysis Pipeline
    print("\n2️⃣ Technical Analysis Pipeline:")
    pipeline_task = {
        "type": "technical_analysis",
        "input_data": complex_task,
        "pipeline_goal": "Complete technical assessment"
    }
    
    pipeline_result = await analysis_pipeline.execute_task(pipeline_task)
    print(f"   Pipeline Status: {pipeline_result.get('status')}")
    print(f"   Stages Completed: {len(pipeline_result.get('stage_results', {}))}")
    
    # Cleanup
    print("\n🛑 Shutting Down Orchestration System:")
    
    stop_tasks = [
        master_orchestrator.stop(),
        technical_supervisor.stop(),
        business_supervisor.stop(),
        expert_panel.stop(),
        analysis_pipeline.stop()
    ]
    
    for agent in specialized_agents:
        stop_tasks.append(agent.stop())
    
    await asyncio.gather(*stop_tasks, return_exceptions=True)
    
    print("✅ Demo Complete!")


def print_hierarchy(hierarchy: Dict[str, Any], indent: int = 0):
    """Print agent hierarchy in tree format"""
    prefix = "  " * indent
    agent_id = hierarchy["agent_id"]
    pattern = hierarchy["orchestration_pattern"]
    capabilities = hierarchy.get("capabilities", [])
    
    print(f"{prefix}🤖 {agent_id} ({pattern})")
    if capabilities:
        print(f"{prefix}   Capabilities: {', '.join(capabilities)}")
    
    for child_id, child_hierarchy in hierarchy.get("children", {}).items():
        print_hierarchy(child_hierarchy, indent + 1)


async def demo_simple_supervisor():
    """Simple demo showing supervisor pattern"""
    
    print("\n🎯 Simple Supervisor Demo")
    print("-" * 30)
    
    # Create supervisor
    supervisor = SupervisorOrchestrator("demo_supervisor", base_port=7000)
    
    # Create specialized agents
    agents = [
        SpecializedAgent("budget_agent", "budget_analysis", base_port=7100),
        SpecializedAgent("tech_agent", "technical_analysis", base_port=7110)
    ]
    
    # Add agents to supervisor
    for agent in agents:
        supervisor.add_child_agent(agent)
    
    # Start system
    await supervisor.start()
    for agent in agents:
        await agent.start()
    
    # Execute task
    task = {
        "type": "budget_analysis",
        "description": "Analyze budget requirements for POS system",
        "budget": 25000
    }
    
    result = await supervisor.execute_task(task)
    print(f"Result: {result.get('status')} - Agent: {result.get('assigned_agent')}")
    
    # Cleanup
    await supervisor.stop()
    for agent in agents:
        await agent.stop()


if __name__ == "__main__":
    print("🚀 Agent Orchestration Demo")
    print("=" * 50)
    
    # Run simple demo first
    asyncio.run(demo_simple_supervisor())
    
    # Run complex demo
    asyncio.run(demo_restaurant_pos_orchestration())
