"""
Processor-Based Orchestration Demo
Shows ZMQ nodes as containers with different processor types
"""

import asyncio
import time
from typing import Dict, Any

from agent_orchestration_base import AgentOrchestrationBase, OrchestrationPattern, AgentCapability
from orchestration_patterns import SupervisorOrchestrator, ScatterGatherOrchestrator
from zmq_node_base import (
    ProcessorType, ProcessingRequest, ProcessingResponse,
    LLMProcessor, MCPToolProcessor, UserInputProcessor, CustomLogicProcessor,
    create_llm_processor, create_mcp_processor, create_user_input_processor, create_custom_processor
)


# Custom logic functions for demonstration
async def budget_analysis_logic(budget_data: Dict[str, Any]) -> Dict[str, Any]:
    """Custom budget analysis logic"""
    budget = budget_data.get("budget", 0)
    requirements = budget_data.get("requirements", [])
    
    # Simulate complex budget analysis
    await asyncio.sleep(0.5)  # Simulate processing time
    
    analysis = {
        "total_budget": budget,
        "requirement_count": len(requirements),
        "budget_per_requirement": budget / max(len(requirements), 1),
        "feasibility": "feasible" if budget > 10000 else "challenging",
        "recommendations": [
            "Prioritize core features",
            "Consider phased implementation",
            "Negotiate with vendors"
        ]
    }
    
    return analysis


def technical_validation_logic(tech_specs: Dict[str, Any]) -> Dict[str, Any]:
    """Custom technical validation logic"""
    specs = tech_specs.get("specifications", {})
    
    validation_results = {
        "validated_specs": specs,
        "compatibility_score": 0.85,
        "potential_issues": [
            "Integration complexity with legacy systems",
            "Scalability concerns for peak hours"
        ],
        "recommended_technologies": [
            "Cloud-based POS solution",
            "RESTful API architecture",
            "Real-time inventory sync"
        ]
    }
    
    return validation_results


class MultiProcessorAgent(AgentOrchestrationBase):
    """
    Agent that demonstrates multiple processor types in one node
    """
    
    def __init__(self, agent_id: str, **kwargs):
        # Create diverse processors
        processors = [
            create_llm_processor(f"{agent_id}_llm", "gpt-4"),
            create_mcp_processor(f"{agent_id}_mcp", "expense_tracker"),
            create_user_input_processor(f"{agent_id}_user_input", "terminal"),
            create_custom_processor(f"{agent_id}_budget_logic", budget_analysis_logic),
            create_custom_processor(f"{agent_id}_tech_validation", technical_validation_logic)
        ]
        
        capabilities = [
            AgentCapability(
                name="comprehensive_analysis",
                description="Multi-processor analysis using LLM, MCP tools, and custom logic",
                input_schema={"task": "dict", "context": "dict"},
                output_schema={"analysis": "dict", "recommendations": "list"},
                processor_types=[ProcessorType.LLM, ProcessorType.MCP_TOOL, ProcessorType.CUSTOM_LOGIC]
            ),
            AgentCapability(
                name="human_consultation",
                description="Interactive consultation with human input",
                input_schema={"question": "str", "options": "list"},
                output_schema={"response": "str", "confidence": "float"},
                processor_types=[ProcessorType.USER_INPUT, ProcessorType.LLM]
            )
        ]
        
        super().__init__(
            agent_id=agent_id,
            orchestration_pattern=OrchestrationPattern.NETWORK,
            processors=processors,
            capabilities=capabilities,
            **kwargs
        )
    
    async def _start_orchestration(self):
        """Start multi-processor agent"""
        print(f"🔧 Multi-Processor Agent {self.agent_id} ready")
        print(f"   Processors: {[p.processor_type.value for p in self.node.processors.values()]}")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using multiple processors in sequence"""
        task_type = task.get("type", "general")
        results = {}
        
        if task_type == "comprehensive_pos_analysis":
            # Step 1: LLM Analysis
            llm_request = ProcessingRequest(
                request_id=f"llm_{task.get('id', 'unknown')}",
                processor_type=ProcessorType.LLM,
                operation="analysis",
                input_data={
                    "data": task,
                    "analysis_type": "pos_system_analysis"
                }
            )
            
            llm_response = await self.process_with_node(llm_request)
            results["llm_analysis"] = llm_response.output_data
            
            # Step 2: Custom Budget Analysis
            if "budget" in task:
                budget_request = ProcessingRequest(
                    request_id=f"budget_{task.get('id', 'unknown')}",
                    processor_type=ProcessorType.CUSTOM_LOGIC,
                    operation="budget_analysis_logic",
                    input_data={
                        "args": [{"budget": task["budget"], "requirements": task.get("requirements", [])}],
                        "kwargs": {}
                    }
                )
                
                budget_response = await self.process_with_node(budget_request)
                results["budget_analysis"] = budget_response.output_data["result"]
            
            # Step 3: Technical Validation
            if "technical_specs" in task:
                tech_request = ProcessingRequest(
                    request_id=f"tech_{task.get('id', 'unknown')}",
                    processor_type=ProcessorType.CUSTOM_LOGIC,
                    operation="technical_validation_logic",
                    input_data={
                        "args": [{"specifications": task["technical_specs"]}],
                        "kwargs": {}
                    }
                )
                
                tech_response = await self.process_with_node(tech_request)
                results["technical_validation"] = tech_response.output_data["result"]
            
            # Step 4: MCP Tool Integration (expense tracking)
            mcp_request = ProcessingRequest(
                request_id=f"mcp_{task.get('id', 'unknown')}",
                processor_type=ProcessorType.MCP_TOOL,
                operation="expense_tracker_operation",
                input_data={
                    "operation": "track_budget",
                    "parameters": {
                        "budget": task.get("budget", 0),
                        "category": "pos_system"
                    }
                }
            )
            
            mcp_response = await self.process_with_node(mcp_request)
            results["expense_tracking"] = mcp_response.output_data
            
            return {
                "status": "completed",
                "task_type": task_type,
                "multi_processor_results": results,
                "processing_summary": {
                    "processors_used": ["LLM", "Custom Logic (Budget)", "Custom Logic (Tech)", "MCP Tool"],
                    "total_steps": 4
                }
            }
        
        elif task_type == "human_consultation":
            # Interactive consultation using user input + LLM
            user_input_request = ProcessingRequest(
                request_id=f"user_{task.get('id', 'unknown')}",
                processor_type=ProcessorType.USER_INPUT,
                operation="user_query",
                input_data={
                    "question": task.get("question", "What are your preferences for the POS system?"),
                    "options": task.get("options", ["Cloud-based", "On-premise", "Hybrid"])
                }
            )
            
            user_response = await self.process_with_node(user_input_request)
            
            # Analyze user response with LLM
            llm_analysis_request = ProcessingRequest(
                request_id=f"llm_user_{task.get('id', 'unknown')}",
                processor_type=ProcessorType.LLM,
                operation="analysis",
                input_data={
                    "data": user_response.output_data,
                    "analysis_type": "user_preference_analysis"
                }
            )
            
            llm_analysis_response = await self.process_with_node(llm_analysis_request)
            
            return {
                "status": "completed",
                "task_type": task_type,
                "user_input": user_response.output_data,
                "llm_analysis": llm_analysis_response.output_data,
                "consultation_complete": True
            }
        
        else:
            return {"error": f"Unknown task type: {task_type}"}
    
    async def _route_message_to_child(self, child_id: str, message) -> Dict[str, Any]:
        """Network agents don't route to children"""
        return {"error": "Network agent has no children"}


async def demo_processor_orchestration():
    """
    Demo: Multi-Processor Agent Orchestration
    Shows how ZMQ nodes contain different processor types
    """
    
    print("🏗️ Setting up Processor-Based Orchestration Demo")
    print("=" * 60)
    
    # Create Multi-Processor Agents with well-separated port ranges
    analysis_agent = MultiProcessorAgent(
        agent_id="comprehensive_analyzer",
        base_port=8000  # Use 8000+ range
    )
    
    consultation_agent = MultiProcessorAgent(
        agent_id="human_consultant", 
        base_port=9000  # Use 9000+ range
    )
    
    # Create Supervisor to coordinate them
    supervisor = SupervisorOrchestrator(
        agent_id="processor_supervisor",
        base_port=10000  # Use 10000+ range
    )
    
    # Add LLM processor to supervisor for intelligent routing
    supervisor.add_processor(create_llm_processor("supervisor_llm", "gpt-4"))
    
    # Build hierarchy
    supervisor.add_child_agent(analysis_agent)
    supervisor.add_child_agent(consultation_agent)
    
    print("\n🚀 Starting Multi-Processor System:")
    
    # Start all agents
    await analysis_agent.start()
    await consultation_agent.start()
    await supervisor.start()
    
    print("\n📊 System Status:")
    for agent in [analysis_agent, consultation_agent, supervisor]:
        status = agent.node.get_node_status()
        print(f"  Agent {agent.agent_id}:")
        print(f"    Processors: {status['processor_count']}")
        print(f"    Types: {[p['processor_type'] for p in status['processors'].values()]}")
    
    # Execute Complex Task
    print("\n📋 Executing Comprehensive POS Analysis:")
    
    complex_task = {
        "id": "pos_analysis_001",
        "type": "comprehensive_pos_analysis",
        "description": "Complete POS system analysis using multiple processor types",
        "budget": 25000,
        "requirements": [
            "order_management",
            "payment_processing",
            "inventory_tracking",
            "reporting"
        ],
        "technical_specs": {
            "concurrent_users": 50,
            "transaction_volume": "500/day",
            "integration_requirements": ["accounting_system", "inventory_system"]
        }
    }
    
    start_time = time.time()
    
    # Route through supervisor
    result = await supervisor.execute_task(complex_task)
    
    execution_time = time.time() - start_time
    
    print(f"\n✅ Analysis Complete ({execution_time:.2f}s)")
    print("📊 Results Summary:")
    print(f"  Status: {result.get('status')}")
    print(f"  Assigned Agent: {result.get('assigned_agent')}")
    
    if result.get('result', {}).get('multi_processor_results'):
        results = result['result']['multi_processor_results']
        print("  Multi-Processor Results:")
        for processor_type, processor_result in results.items():
            print(f"    • {processor_type}: {type(processor_result).__name__}")
    
    # Demonstrate Human Consultation
    print("\n❓ Demonstrating Human Consultation:")
    
    consultation_task = {
        "id": "consultation_001",
        "type": "human_consultation",
        "question": "Which POS deployment model do you prefer?",
        "options": ["Cloud-based (SaaS)", "On-premise", "Hybrid Cloud"]
    }
    
    # This would normally wait for user input
    print("  (Simulating user interaction - in real demo, user would provide input)")
    
    # Cleanup
    print("\n🛑 Shutting Down Multi-Processor System:")
    
    await supervisor.stop()
    await analysis_agent.stop()
    await consultation_agent.stop()
    
    print("✅ Demo Complete!")


async def demo_processor_types():
    """Simple demo showing different processor types"""
    
    print("\n🔧 Processor Types Demo")
    print("-" * 30)
    
    # Create agent with different processor types (use unique port)
    agent = MultiProcessorAgent("demo_agent", base_port=9000)
    await agent.start()
    
    # Test LLM Processor
    print("\n1️⃣ Testing LLM Processor:")
    llm_request = ProcessingRequest(
        request_id="llm_test",
        processor_type=ProcessorType.LLM,
        operation="text_generation",
        input_data={"prompt": "Analyze the benefits of cloud-based POS systems"}
    )
    
    llm_response = await agent.process_with_node(llm_request)
    print(f"  Status: {llm_response.status}")
    print(f"  Processing Time: {llm_response.processing_time:.2f}s")
    
    # Test Custom Logic Processor
    print("\n2️⃣ Testing Custom Logic Processor:")
    custom_request = ProcessingRequest(
        request_id="custom_test",
        processor_type=ProcessorType.CUSTOM_LOGIC,
        operation="budget_analysis_logic",
        input_data={
            "args": [{"budget": 15000, "requirements": ["POS", "Inventory", "Reports"]}],
            "kwargs": {}
        }
    )
    
    custom_response = await agent.process_with_node(custom_request)
    print(f"  Status: {custom_response.status}")
    print(f"  Result: {custom_response.output_data['result']['feasibility']}")
    
    # Test MCP Tool Processor
    print("\n3️⃣ Testing MCP Tool Processor:")
    mcp_request = ProcessingRequest(
        request_id="mcp_test",
        processor_type=ProcessorType.MCP_TOOL,
        operation="expense_tracker_operation",
        input_data={
            "operation": "create_budget",
            "parameters": {"amount": 25000, "category": "technology"}
        }
    )
    
    mcp_response = await agent.process_with_node(mcp_request)
    print(f"  Status: {mcp_response.status}")
    print(f"  Tool: {mcp_response.output_data['result']['tool']}")
    
    await agent.stop()


if __name__ == "__main__":
    print("🚀 Processor-Based Orchestration Demo")
    print("=" * 50)
    
    # Run processor types demo first
    asyncio.run(demo_processor_types())
    
    # Run full orchestration demo
    asyncio.run(demo_processor_orchestration())
