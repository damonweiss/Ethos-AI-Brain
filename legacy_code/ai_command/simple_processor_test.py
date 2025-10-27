"""
Simple Processor Test - Minimal demo to verify the architecture works
"""

import asyncio
from zmq_node_base import (
    ProcessorType, ProcessingRequest, ProcessingResponse,
    LLMProcessor, MCPToolProcessor, UserInputProcessor, CustomLogicProcessor,
    create_llm_processor, create_mcp_processor, create_user_input_processor, create_custom_processor
)
from agent_orchestration_base import AgentOrchestrationBase, OrchestrationPattern, AgentCapability


def simple_calculation(x: int, y: int) -> int:
    """Simple calculation function for testing"""
    return x + y


class SimpleTestAgent(AgentOrchestrationBase):
    """Minimal test agent with processors"""
    
    def __init__(self, agent_id: str, **kwargs):
        # Create simple processors
        processors = [
            create_llm_processor(f"{agent_id}_llm", "gpt-4"),
            create_custom_processor(f"{agent_id}_calc", simple_calculation)
        ]
        
        super().__init__(
            agent_id=agent_id,
            orchestration_pattern=OrchestrationPattern.NETWORK,
            processors=processors,
            **kwargs
        )
    
    def _setup_network_zmq(self):
        """Override to disable network ZMQ setup that causes conflicts"""
        print("🔧 Skipping network ZMQ setup to avoid port conflicts")
        self.network_router = None
    
    async def _start_orchestration(self):
        """Start simple test agent"""
        print(f"🔧 Simple Test Agent {self.agent_id} ready")
        print(f"   Processors: {[p.processor_type.value for p in self.node.processors.values()]}")
    
    async def execute_task(self, task):
        """Execute task using processors"""
        return {"status": "completed", "agent": self.agent_id}
    
    async def _route_message_to_child(self, child_id: str, message):
        """No children in simple test"""
        return {"error": "No children"}


async def test_simple_processor():
    """Test basic processor functionality"""
    
    print("🧪 Simple Processor Architecture Test")
    print("=" * 40)
    
    # Create agent with random port range to avoid conflicts
    import random
    random_port = random.randint(25000, 35000)
    agent = SimpleTestAgent("test_agent", base_port=random_port)
    print(f"🔧 Using random base port: {random_port}")
    
    try:
        print("\n🚀 Starting agent...")
        await agent.start()
        
        print("\n📊 Agent Status:")
        status = agent.node.get_node_status()
        print(f"  Node ID: {status['node_id']}")
        print(f"  Processors: {status['processor_count']}")
        print(f"  Running: {status['is_running']}")
        
        print("\n1️⃣ Testing Custom Logic Processor:")
        calc_request = ProcessingRequest(
            request_id="calc_test",
            processor_type=ProcessorType.CUSTOM_LOGIC,
            operation="simple_calculation",
            input_data={
                "args": [5, 3],
                "kwargs": {}
            }
        )
        
        calc_response = await agent.process_with_node(calc_request)
        print(f"  Status: {calc_response.status}")
        print(f"  Result: {calc_response.output_data.get('result', 'N/A')}")
        
        print("\n2️⃣ Testing LLM Processor:")
        llm_request = ProcessingRequest(
            request_id="llm_test",
            processor_type=ProcessorType.LLM,
            operation="text_generation",
            input_data={
                "prompt": "What is 2+2?",
                "context": {}
            }
        )
        
        llm_response = await agent.process_with_node(llm_request)
        print(f"  Status: {llm_response.status}")
        print(f"  Processing Time: {llm_response.processing_time:.2f}s")
        
        print("\n✅ Architecture Test PASSED!")
        
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        
    finally:
        print("\n🛑 Stopping agent...")
        await agent.stop()
        print("✅ Test Complete!")


if __name__ == "__main__":
    asyncio.run(test_simple_processor())
