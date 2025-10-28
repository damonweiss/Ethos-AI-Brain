"""
Test ProcessorInterface and Implementations - Must Pass
Tests each function in ProcessorInterface and its concrete implementations
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.core.zeromq.zmq_node_base import (
    ProcessorInterface, ProcessorType, ProcessingRequest, ProcessingResponse,
    LLMProcessor, MCPToolProcessor, UserInputProcessor, CustomLogicProcessor,
    ProcessorCapability
)


def test_llm_processor_creation():
    """Test LLMProcessor.__init__ - processor creation"""
    processor = LLMProcessor("llm_test_001", "gpt-4")
    
    print(f"Expected processor_id: llm_test_001, Actual: {processor.processor_id}")
    print(f"Expected processor_type: LLM, Actual: {processor.processor_type}")
    print(f"Expected model_name: gpt-4, Actual: {processor.model_name}")
    print(f"Expected is_initialized: False, Actual: {processor.is_initialized}")
    print(f"Expected is_busy: False, Actual: {processor.is_busy}")
    
    assert processor.processor_id == "llm_test_001"
    assert processor.processor_type == ProcessorType.LLM
    assert processor.model_name == "gpt-4"
    assert processor.is_initialized == False
    assert processor.is_busy == False
    assert isinstance(processor.capabilities, list)
    
    print("[SUCCESS] LLMProcessor creation works correctly")


def test_llm_processor_initialize():
    """Test LLMProcessor.initialize - processor initialization"""
    processor = LLMProcessor("llm_test_002", "gpt-3.5-turbo")
    
    # Test initialization (synchronous)
    import asyncio
    result = asyncio.run(processor.initialize())
    
    print(f"Expected initialization result: True, Actual: {result}")
    print(f"Expected is_initialized: True, Actual: {processor.is_initialized}")
    print(f"Expected capabilities count > 0: {len(processor.capabilities) > 0}")
    
    assert result == True
    assert processor.is_initialized == True
    assert len(processor.capabilities) > 0
    
    # Check capabilities structure
    for capability in processor.capabilities:
        print(f"Capability: {capability.name}, Type: {capability.processor_type}")
        assert isinstance(capability, ProcessorCapability)
        assert capability.processor_type == ProcessorType.LLM
    
    print("[SUCCESS] LLMProcessor initialization works correctly")


def test_llm_processor_get_status():
    """Test LLMProcessor.get_status - status reporting"""
    processor = LLMProcessor("llm_test_003", "claude")
    
    status = processor.get_status()
    
    print(f"Status keys: {list(status.keys())}")
    print(f"Processor ID: {status.get('processor_id')}")
    print(f"Processor type: {status.get('processor_type')}")
    print(f"Is initialized: {status.get('is_initialized')}")
    print(f"Is busy: {status.get('is_busy')}")
    
    assert isinstance(status, dict)
    assert status["processor_id"] == "llm_test_003"
    assert status["processor_type"] == "llm"
    assert status["is_initialized"] == False
    assert status["is_busy"] == False
    assert "capabilities_count" in status
    
    print("[SUCCESS] LLMProcessor get_status works correctly")


def test_llm_processor_get_capabilities():
    """Test LLMProcessor.get_capabilities - capabilities listing"""
    processor = LLMProcessor("llm_test_004", "gpt-4")
    
    # Before initialization - should be empty
    capabilities = processor.get_capabilities()
    
    print(f"Expected empty capabilities before init: {len(capabilities) == 0}")
    
    assert isinstance(capabilities, list)
    assert len(capabilities) == 0
    
    print("[SUCCESS] LLMProcessor get_capabilities works correctly")


def test_mcp_tool_processor_creation():
    """Test MCPToolProcessor.__init__ - processor creation"""
    processor = MCPToolProcessor("mcp_test_001", "file_operations")
    
    print(f"Expected processor_id: mcp_test_001, Actual: {processor.processor_id}")
    print(f"Expected processor_type: MCP_TOOL, Actual: {processor.processor_type}")
    print(f"Expected tool_name: file_operations, Actual: {processor.tool_name}")
    print(f"Expected is_initialized: False, Actual: {processor.is_initialized}")
    
    assert processor.processor_id == "mcp_test_001"
    assert processor.processor_type == ProcessorType.MCP_TOOL
    assert processor.tool_name == "file_operations"
    assert processor.is_initialized == False
    assert processor.mcp_client is None
    
    print("[SUCCESS] MCPToolProcessor creation works correctly")


def test_mcp_tool_processor_initialize():
    """Test MCPToolProcessor.initialize - processor initialization"""
    processor = MCPToolProcessor("mcp_test_002", "test_server")
    
    # Test initialization (synchronous)
    import asyncio
    result = asyncio.run(processor.initialize())
    
    print(f"Expected initialization result: True, Actual: {result}")
    print(f"Expected is_initialized: True, Actual: {processor.is_initialized}")
    print(f"Expected mcp_client set: {processor.mcp_client is not None}")
    
    assert result == True
    assert processor.is_initialized == True
    assert processor.mcp_client is not None
    assert len(processor.capabilities) > 0
    
    print("[SUCCESS] MCPToolProcessor initialization works correctly")


def test_mcp_tool_processor_shutdown():
    """Test MCPToolProcessor.shutdown - processor cleanup"""
    processor = MCPToolProcessor("mcp_test_003", "test_server")
    
    # Initialize first (synchronous)
    import asyncio
    asyncio.run(processor.initialize())
    
    # Test shutdown (synchronous)
    result = asyncio.run(processor.shutdown())
    
    print(f"Expected shutdown result: True, Actual: {result}")
    print(f"Expected is_initialized: False, Actual: {processor.is_initialized}")
    print(f"Expected mcp_client cleared: {processor.mcp_client is None}")
    
    assert result == True
    assert processor.is_initialized == False
    assert processor.mcp_client is None
    
    print("[SUCCESS] MCPToolProcessor shutdown works correctly")


def test_user_input_processor_creation():
    """Test UserInputProcessor.__init__ - processor creation"""
    processor = UserInputProcessor("user_test_001", "terminal")
    
    print(f"Expected processor_id: user_test_001, Actual: {processor.processor_id}")
    print(f"Expected processor_type: USER_INPUT, Actual: {processor.processor_type}")
    print(f"Expected input_method: terminal, Actual: {processor.input_method}")
    
    assert processor.processor_id == "user_test_001"
    assert processor.processor_type == ProcessorType.USER_INPUT
    assert processor.input_method == "terminal"
    assert isinstance(processor.pending_requests, dict)
    
    print("[SUCCESS] UserInputProcessor creation works correctly")


def test_user_input_processor_default_method():
    """Test UserInputProcessor.__init__ with default input method"""
    processor = UserInputProcessor("user_test_002")
    
    print(f"Expected default input_method: terminal, Actual: {processor.input_method}")
    
    assert processor.input_method == "terminal"
    
    print("[SUCCESS] UserInputProcessor default method works correctly")


def test_user_input_processor_initialize():
    """Test UserInputProcessor.initialize - processor initialization"""
    processor = UserInputProcessor("user_test_003", "gui")
    
    # Test initialization (synchronous)
    import asyncio
    result = asyncio.run(processor.initialize())
    
    print(f"Expected initialization result: True, Actual: {result}")
    print(f"Expected is_initialized: True, Actual: {processor.is_initialized}")
    print(f"Expected capabilities count > 0: {len(processor.capabilities) > 0}")
    
    assert result == True
    assert processor.is_initialized == True
    assert len(processor.capabilities) > 0
    
    # Check capability structure
    capability = processor.capabilities[0]
    print(f"Capability name: {capability.name}")
    print(f"Capability type: {capability.processor_type}")
    
    assert capability.processor_type == ProcessorType.USER_INPUT
    assert "user_query" in capability.name
    
    print("[SUCCESS] UserInputProcessor initialization works correctly")


def test_custom_logic_processor_creation():
    """Test CustomLogicProcessor.__init__ - processor creation"""
    
    def test_function(x: int, y: int) -> int:
        return x * y
    
    processor = CustomLogicProcessor("custom_test_001", test_function)
    
    print(f"Expected processor_id: custom_test_001, Actual: {processor.processor_id}")
    print(f"Expected processor_type: CUSTOM_LOGIC, Actual: {processor.processor_type}")
    print(f"Expected function_name: test_function, Actual: {processor.function_name}")
    
    assert processor.processor_id == "custom_test_001"
    assert processor.processor_type == ProcessorType.CUSTOM_LOGIC
    assert processor.logic_function == test_function
    assert processor.function_name == "test_function"
    
    print("[SUCCESS] CustomLogicProcessor creation works correctly")


def test_custom_logic_processor_lambda():
    """Test CustomLogicProcessor.__init__ with lambda function"""
    
    lambda_func = lambda a, b: a + b
    processor = CustomLogicProcessor("custom_test_002", lambda_func)
    
    print(f"Expected function_name: <lambda>, Actual: {processor.function_name}")
    
    assert processor.logic_function == lambda_func
    assert processor.function_name == "<lambda>"  # Actual lambda name
    
    print("[SUCCESS] CustomLogicProcessor with lambda works correctly")


def test_custom_logic_processor_initialize():
    """Test CustomLogicProcessor.initialize - processor initialization"""
    
    def math_operation(a: float, b: float) -> float:
        return a + b
    
    processor = CustomLogicProcessor("custom_test_002", math_operation)
    
    # Test initialization (synchronous)
    import asyncio
    result = asyncio.run(processor.initialize())
    
    print(f"Expected initialization result: True, Actual: {result}")
    print(f"Expected is_initialized: True, Actual: {processor.is_initialized}")
    print(f"Expected capabilities count > 0: {len(processor.capabilities) > 0}")
    
    assert result == True
    assert processor.is_initialized == True
    assert len(processor.capabilities) > 0
    
    # Check capability
    capability = processor.capabilities[0]
    print(f"Capability name: {capability.name}")
    
    assert capability.name == "math_operation"
    assert capability.processor_type == ProcessorType.CUSTOM_LOGIC
    
    print("[SUCCESS] CustomLogicProcessor initialization works correctly")


def test_processing_request_creation():
    """Test ProcessingRequest dataclass creation"""
    request = ProcessingRequest(
        request_id="req_001",
        processor_type=ProcessorType.LLM,
        operation="text_generation",
        input_data={"prompt": "Hello world"}
    )
    
    print(f"Request ID: {request.request_id}")
    print(f"Processor type: {request.processor_type}")
    print(f"Operation: {request.operation}")
    print(f"Input data: {request.input_data}")
    print(f"Default timeout: {request.timeout}")
    print(f"Default priority: {request.priority}")
    
    assert request.request_id == "req_001"
    assert request.processor_type == ProcessorType.LLM
    assert request.operation == "text_generation"
    assert request.input_data == {"prompt": "Hello world"}
    assert request.timeout == 30.0  # Default
    assert request.priority == 1    # Default
    
    print("[SUCCESS] ProcessingRequest creation works correctly")


def test_processing_response_creation():
    """Test ProcessingResponse dataclass creation"""
    response = ProcessingResponse(
        request_id="req_002",
        processor_type=ProcessorType.MCP_TOOL,
        status="success",
        output_data={"result": "operation completed"}
    )
    
    print(f"Request ID: {response.request_id}")
    print(f"Processor type: {response.processor_type}")
    print(f"Status: {response.status}")
    print(f"Output data: {response.output_data}")
    print(f"Default error_message: {response.error_message}")
    print(f"Default processing_time: {response.processing_time}")
    
    assert response.request_id == "req_002"
    assert response.processor_type == ProcessorType.MCP_TOOL
    assert response.status == "success"
    assert response.output_data == {"result": "operation completed"}
    assert response.error_message is None  # Default
    assert response.processing_time == 0.0  # Default
    
    print("[SUCCESS] ProcessingResponse creation works correctly")


def test_processor_capability_creation():
    """Test ProcessorCapability dataclass creation"""
    capability = ProcessorCapability(
        name="text_analysis",
        processor_type=ProcessorType.LLM,
        input_schema={"text": "str", "options": "dict"},
        output_schema={"analysis": "dict", "confidence": "float"},
        description="Advanced text analysis capability"
    )
    
    print(f"Capability name: {capability.name}")
    print(f"Processor type: {capability.processor_type}")
    print(f"Input schema: {capability.input_schema}")
    print(f"Output schema: {capability.output_schema}")
    print(f"Description: {capability.description}")
    print(f"Default cost_estimate: {capability.cost_estimate}")
    print(f"Default reliability: {capability.reliability}")
    
    assert capability.name == "text_analysis"
    assert capability.processor_type == ProcessorType.LLM
    assert capability.input_schema == {"text": "str", "options": "dict"}
    assert capability.output_schema == {"analysis": "dict", "confidence": "float"}
    assert capability.description == "Advanced text analysis capability"
    assert capability.cost_estimate == 0.0  # Default
    assert capability.reliability == 1.0    # Default
    
    print("[SUCCESS] ProcessorCapability creation works correctly")
