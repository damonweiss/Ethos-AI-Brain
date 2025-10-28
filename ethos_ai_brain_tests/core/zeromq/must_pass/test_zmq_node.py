"""
Test ZMQNode - Must Pass
Tests each function in the ZMQNode class with real code only
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.core.zeromq.zmq_node_base import (
    ZMQNode, ProcessorInterface, ProcessorType, ProcessingRequest, ProcessingResponse,
    LLMProcessor, MCPToolProcessor, UserInputProcessor, CustomLogicProcessor
)


def test_zmq_node_creation():
    """Test ZMQNode.__init__ - node creation and initialization"""
    node = ZMQNode("test_node_001")
    
    print(f"Expected node_id: test_node_001, Actual: {node.node_id}")
    print(f"Expected base_port: 5555, Actual: {node.base_port}")
    print(f"Expected empty processors: {{}}, Actual: {node.processors}")
    print(f"Expected is_running: False, Actual: {node.is_running}")
    
    assert node.node_id == "test_node_001"
    assert node.base_port == 5555
    assert node.processors == {}
    assert node.is_running == False
    assert hasattr(node, 'zmq_engine')
    assert hasattr(node, 'request_handlers')
    
    print("[SUCCESS] ZMQNode creation works correctly")


def test_zmq_node_creation_with_custom_port():
    """Test ZMQNode.__init__ with custom base_port"""
    node = ZMQNode("test_node_002", base_port=8080)
    
    print(f"Expected base_port: 8080, Actual: {node.base_port}")
    
    assert node.base_port == 8080
    
    print("[SUCCESS] ZMQNode creation with custom port works correctly")


def test_zmq_node_add_processor():
    """Test ZMQNode.add_processor - adding processors to node"""
    node = ZMQNode("test_node_003")
    
    # Create a real processor
    llm_processor = LLMProcessor("llm_001", "gpt-4")
    
    # Test adding processor
    node.add_processor(llm_processor)
    
    print(f"Expected 1 processor, Actual: {len(node.processors)}")
    print(f"Expected processor in dict: {'llm_001' in node.processors}")
    print(f"Processor ID: {node.processors['llm_001'].processor_id}")
    
    assert len(node.processors) == 1
    assert "llm_001" in node.processors
    assert node.processors["llm_001"] == llm_processor
    assert node.processors["llm_001"].processor_id == "llm_001"
    
    print("[SUCCESS] Add processor works correctly")


def test_zmq_node_add_multiple_processors():
    """Test ZMQNode.add_processor with multiple processors"""
    node = ZMQNode("test_node_004")
    
    # Create multiple processors
    llm_processor = LLMProcessor("llm_002", "gpt-4")
    mcp_processor = MCPToolProcessor("mcp_001", "file_tool")
    
    # Add processors
    node.add_processor(llm_processor)
    node.add_processor(mcp_processor)
    
    print(f"Expected 2 processors, Actual: {len(node.processors)}")
    print(f"Processor IDs: {list(node.processors.keys())}")
    
    assert len(node.processors) == 2
    assert "llm_002" in node.processors
    assert "mcp_001" in node.processors
    
    print("[SUCCESS] Add multiple processors works correctly")


def test_zmq_node_remove_processor():
    """Test ZMQNode.remove_processor - removing processors from node"""
    node = ZMQNode("test_node_005")
    
    # Add a processor first
    llm_processor = LLMProcessor("llm_003", "gpt-4")
    node.add_processor(llm_processor)
    
    print(f"Before removal - processors: {len(node.processors)}")
    
    # Remove the processor
    node.remove_processor("llm_003")
    
    print(f"After removal - processors: {len(node.processors)}")
    print(f"Processor still in dict: {'llm_003' in node.processors}")
    
    assert len(node.processors) == 0
    assert "llm_003" not in node.processors
    
    print("[SUCCESS] Remove processor works correctly")


def test_zmq_node_remove_nonexistent_processor():
    """Test ZMQNode.remove_processor with non-existent processor"""
    node = ZMQNode("test_node_006")
    
    # Try to remove non-existent processor - should not crash
    node.remove_processor("nonexistent_processor")
    
    print(f"Processors after removing nonexistent: {len(node.processors)}")
    
    assert len(node.processors) == 0
    
    print("[SUCCESS] Remove nonexistent processor handled correctly")


def test_zmq_node_get_processor():
    """Test ZMQNode.get_processor - retrieving specific processor"""
    node = ZMQNode("test_node_007")
    
    # Add a processor
    llm_processor = LLMProcessor("llm_004", "gpt-4")
    node.add_processor(llm_processor)
    
    # Test getting existing processor
    retrieved_processor = node.get_processor("llm_004")
    
    print(f"Retrieved processor ID: {retrieved_processor.processor_id if retrieved_processor else None}")
    print(f"Is same processor: {retrieved_processor is llm_processor}")
    
    assert retrieved_processor is not None
    assert retrieved_processor == llm_processor
    assert retrieved_processor.processor_id == "llm_004"
    
    # Test getting non-existent processor
    nonexistent = node.get_processor("nonexistent")
    
    print(f"Nonexistent processor result: {nonexistent}")
    
    assert nonexistent is None
    
    print("[SUCCESS] Get processor works correctly")


def test_zmq_node_get_processors_by_type():
    """Test ZMQNode.get_processors_by_type - filtering processors by type"""
    node = ZMQNode("test_node_008")
    
    # Add processors of different types
    llm_processor1 = LLMProcessor("llm_005", "gpt-4")
    llm_processor2 = LLMProcessor("llm_006", "claude")
    mcp_processor = MCPToolProcessor("mcp_002", "database_tool")
    
    node.add_processor(llm_processor1)
    node.add_processor(llm_processor2)
    node.add_processor(mcp_processor)
    
    # Test getting LLM processors
    llm_processors = node.get_processors_by_type(ProcessorType.LLM)
    
    print(f"Expected 2 LLM processors, Actual: {len(llm_processors)}")
    print(f"LLM processor IDs: {[p.processor_id for p in llm_processors]}")
    
    assert len(llm_processors) == 2
    assert all(p.processor_type == ProcessorType.LLM for p in llm_processors)
    
    # Test getting MCP processors
    mcp_processors = node.get_processors_by_type(ProcessorType.MCP_TOOL)
    
    print(f"Expected 1 MCP processor, Actual: {len(mcp_processors)}")
    
    assert len(mcp_processors) == 1
    assert mcp_processors[0].processor_type == ProcessorType.MCP_TOOL
    
    # Test getting non-existent type
    user_processors = node.get_processors_by_type(ProcessorType.USER_INPUT)
    
    print(f"Expected 0 USER_INPUT processors, Actual: {len(user_processors)}")
    
    assert len(user_processors) == 0
    
    print("[SUCCESS] Get processors by type works correctly")


def test_zmq_node_get_node_status():
    """Test ZMQNode.get_node_status - getting comprehensive node status"""
    node = ZMQNode("test_node_009")
    
    # Add a processor
    llm_processor = LLMProcessor("llm_007", "gpt-4")
    node.add_processor(llm_processor)
    
    # Get status
    status = node.get_node_status()
    
    print(f"Status keys: {list(status.keys())}")
    print(f"Node ID: {status.get('node_id')}")
    print(f"Is running: {status.get('is_running')}")
    print(f"Processor count: {status.get('processor_count')}")
    
    assert isinstance(status, dict)
    assert status["node_id"] == "test_node_009"
    assert status["is_running"] == False
    assert status["processor_count"] == 1
    assert "processors" in status
    assert "capabilities" in status
    
    # Check processor status
    processor_status = status["processors"]
    assert "llm_007" in processor_status
    assert processor_status["llm_007"]["processor_id"] == "llm_007"
    
    print("[SUCCESS] Get node status works correctly")


def test_zmq_node_get_all_capabilities():
    """Test ZMQNode.get_all_capabilities - getting all processor capabilities"""
    node = ZMQNode("test_node_010")
    
    # Test with no processors
    capabilities = node.get_all_capabilities()
    
    print(f"Expected empty capabilities list, Actual: {len(capabilities)}")
    
    assert isinstance(capabilities, list)
    assert len(capabilities) == 0
    
    # Add a processor and test again
    llm_processor = LLMProcessor("llm_008", "gpt-4")
    node.add_processor(llm_processor)
    
    capabilities = node.get_all_capabilities()
    
    print(f"Capabilities after adding processor: {len(capabilities)}")
    
    assert isinstance(capabilities, list)
    # Note: capabilities will be empty until processor is initialized
    # This tests the method works, not the processor initialization
    
    print("[SUCCESS] Get all capabilities works correctly")


def test_zmq_node_creation_with_processors():
    """Test ZMQNode.__init__ with processors parameter"""
    # Create processors
    llm_processor = LLMProcessor("llm_009", "gpt-4")
    mcp_processor = MCPToolProcessor("mcp_003", "web_tool")
    
    # Create node with processors
    node = ZMQNode("test_node_011", processors=[llm_processor, mcp_processor])
    
    print(f"Expected 2 processors, Actual: {len(node.processors)}")
    print(f"Processor IDs: {list(node.processors.keys())}")
    
    assert len(node.processors) == 2
    assert "llm_009" in node.processors
    assert "mcp_003" in node.processors
    
    print("[SUCCESS] Node creation with processors works correctly")


def test_zmq_node_custom_logic_processor():
    """Test ZMQNode with CustomLogicProcessor"""
    
    # Define a simple custom function
    def simple_math(a: int, b: int) -> int:
        return a + b
    
    # Create custom processor
    custom_processor = CustomLogicProcessor("custom_001", simple_math)
    
    # Create node and add processor
    node = ZMQNode("test_node_012")
    node.add_processor(custom_processor)
    
    print(f"Expected 1 processor, Actual: {len(node.processors)}")
    print(f"Processor type: {node.processors['custom_001'].processor_type}")
    
    assert len(node.processors) == 1
    assert node.processors["custom_001"].processor_type == ProcessorType.CUSTOM_LOGIC
    
    print("[SUCCESS] Custom logic processor integration works correctly")
