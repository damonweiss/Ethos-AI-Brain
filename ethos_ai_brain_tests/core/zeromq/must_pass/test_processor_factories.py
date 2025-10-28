"""
Test Processor Factory Functions - Must Pass
Tests each factory function in zmq_node_base module
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.core.zeromq.zmq_node_base import (
    create_llm_processor, create_mcp_processor, create_user_input_processor, create_custom_processor,
    LLMProcessor, MCPToolProcessor, UserInputProcessor, CustomLogicProcessor,
    ProcessorType
)


def test_create_llm_processor_default():
    """Test create_llm_processor factory function with default model"""
    processor = create_llm_processor("factory_llm_001")
    
    print(f"Expected processor type: LLMProcessor, Actual: {type(processor).__name__}")
    print(f"Expected processor_id: factory_llm_001, Actual: {processor.processor_id}")
    print(f"Expected model_name: gpt-4, Actual: {processor.model_name}")
    print(f"Expected processor_type: LLM, Actual: {processor.processor_type}")
    
    assert isinstance(processor, LLMProcessor)
    assert processor.processor_id == "factory_llm_001"
    assert processor.model_name == "gpt-4"  # Default model
    assert processor.processor_type == ProcessorType.LLM
    
    print("[SUCCESS] create_llm_processor with default model works correctly")


def test_create_llm_processor_custom_model():
    """Test create_llm_processor factory function with custom model"""
    processor = create_llm_processor("factory_llm_002", "claude-3")
    
    print(f"Expected processor_id: factory_llm_002, Actual: {processor.processor_id}")
    print(f"Expected model_name: claude-3, Actual: {processor.model_name}")
    
    assert isinstance(processor, LLMProcessor)
    assert processor.processor_id == "factory_llm_002"
    assert processor.model_name == "claude-3"
    assert processor.processor_type == ProcessorType.LLM
    
    print("[SUCCESS] create_llm_processor with custom model works correctly")


def test_create_mcp_processor():
    """Test create_mcp_processor factory function"""
    processor = create_mcp_processor("factory_mcp_001", "file_operations")
    
    print(f"Expected processor type: MCPToolProcessor, Actual: {type(processor).__name__}")
    print(f"Expected processor_id: factory_mcp_001, Actual: {processor.processor_id}")
    print(f"Expected tool_name: file_operations, Actual: {processor.tool_name}")
    print(f"Expected processor_type: MCP_TOOL, Actual: {processor.processor_type}")
    
    assert isinstance(processor, MCPToolProcessor)
    assert processor.processor_id == "factory_mcp_001"
    assert processor.tool_name == "file_operations"
    assert processor.processor_type == ProcessorType.MCP_TOOL
    
    print("[SUCCESS] create_mcp_processor works correctly")


def test_create_user_input_processor_default():
    """Test create_user_input_processor factory function with default method"""
    processor = create_user_input_processor("factory_user_001")
    
    print(f"Expected processor type: UserInputProcessor, Actual: {type(processor).__name__}")
    print(f"Expected processor_id: factory_user_001, Actual: {processor.processor_id}")
    print(f"Expected input_method: terminal, Actual: {processor.input_method}")
    print(f"Expected processor_type: USER_INPUT, Actual: {processor.processor_type}")
    
    assert isinstance(processor, UserInputProcessor)
    assert processor.processor_id == "factory_user_001"
    assert processor.input_method == "terminal"  # Default method
    assert processor.processor_type == ProcessorType.USER_INPUT
    
    print("[SUCCESS] create_user_input_processor with default method works correctly")


def test_create_user_input_processor_custom_method():
    """Test create_user_input_processor factory function with custom method"""
    processor = create_user_input_processor("factory_user_002", "web_interface")
    
    print(f"Expected processor_id: factory_user_002, Actual: {processor.processor_id}")
    print(f"Expected input_method: web_interface, Actual: {processor.input_method}")
    
    assert isinstance(processor, UserInputProcessor)
    assert processor.processor_id == "factory_user_002"
    assert processor.input_method == "web_interface"
    assert processor.processor_type == ProcessorType.USER_INPUT
    
    print("[SUCCESS] create_user_input_processor with custom method works correctly")


def test_create_custom_processor_regular_function():
    """Test create_custom_processor factory function with regular function"""
    
    def multiply_numbers(x: float, y: float) -> float:
        return x * y
    
    processor = create_custom_processor("factory_custom_001", multiply_numbers)
    
    print(f"Expected processor type: CustomLogicProcessor, Actual: {type(processor).__name__}")
    print(f"Expected processor_id: factory_custom_001, Actual: {processor.processor_id}")
    print(f"Expected function_name: multiply_numbers, Actual: {processor.function_name}")
    print(f"Expected processor_type: CUSTOM_LOGIC, Actual: {processor.processor_type}")
    
    assert isinstance(processor, CustomLogicProcessor)
    assert processor.processor_id == "factory_custom_001"
    assert processor.function_name == "multiply_numbers"
    assert processor.processor_type == ProcessorType.CUSTOM_LOGIC
    assert processor.logic_function == multiply_numbers
    
    print("[SUCCESS] create_custom_processor with regular function works correctly")


def test_create_custom_processor_lambda():
    """Test create_custom_processor factory function with lambda"""
    
    lambda_func = lambda a, b: a - b
    processor = create_custom_processor("factory_custom_002", lambda_func)
    
    print(f"Expected processor_id: factory_custom_002, Actual: {processor.processor_id}")
    print(f"Expected function_name: <lambda>, Actual: {processor.function_name}")
    
    assert isinstance(processor, CustomLogicProcessor)
    assert processor.processor_id == "factory_custom_002"
    assert processor.function_name == "<lambda>"
    assert processor.processor_type == ProcessorType.CUSTOM_LOGIC
    assert processor.logic_function == lambda_func
    
    print("[SUCCESS] create_custom_processor with lambda works correctly")


def test_create_custom_processor_class_method():
    """Test create_custom_processor factory function with class method"""
    
    class MathOperations:
        @staticmethod
        def divide(a: float, b: float) -> float:
            return a / b if b != 0 else 0.0
    
    processor = create_custom_processor("factory_custom_003", MathOperations.divide)
    
    print(f"Expected processor_id: factory_custom_003, Actual: {processor.processor_id}")
    print(f"Expected function_name: divide, Actual: {processor.function_name}")
    
    assert isinstance(processor, CustomLogicProcessor)
    assert processor.processor_id == "factory_custom_003"
    assert processor.function_name == "divide"
    assert processor.processor_type == ProcessorType.CUSTOM_LOGIC
    assert processor.logic_function == MathOperations.divide
    
    print("[SUCCESS] create_custom_processor with class method works correctly")


def test_factory_functions_return_different_instances():
    """Test that factory functions return different instances"""
    
    # Create multiple processors with same parameters
    processor1 = create_llm_processor("test_id", "gpt-4")
    processor2 = create_llm_processor("test_id", "gpt-4")
    
    print(f"Processor1 ID: {id(processor1)}")
    print(f"Processor2 ID: {id(processor2)}")
    print(f"Are different instances: {processor1 is not processor2}")
    
    assert processor1 is not processor2  # Different instances
    assert processor1.processor_id == processor2.processor_id  # Same configuration
    assert processor1.model_name == processor2.model_name
    
    print("[SUCCESS] Factory functions return different instances correctly")


def test_all_factory_functions_exist():
    """Test that all expected factory functions exist and are callable"""
    
    factory_functions = [
        create_llm_processor,
        create_mcp_processor, 
        create_user_input_processor,
        create_custom_processor
    ]
    
    for func in factory_functions:
        print(f"Checking function: {func.__name__}")
        assert callable(func), f"Function {func.__name__} is not callable"
    
    print("[SUCCESS] All factory functions exist and are callable")


def test_factory_functions_parameter_validation():
    """Test factory functions handle parameters correctly"""
    
    # Test that processor_id is required (should not crash with valid ID)
    llm_proc = create_llm_processor("valid_id")
    mcp_proc = create_mcp_processor("valid_id", "tool_name")
    user_proc = create_user_input_processor("valid_id")
    
    def dummy_func():
        pass
    
    custom_proc = create_custom_processor("valid_id", dummy_func)
    
    print(f"LLM processor created: {llm_proc.processor_id}")
    print(f"MCP processor created: {mcp_proc.processor_id}")
    print(f"User processor created: {user_proc.processor_id}")
    print(f"Custom processor created: {custom_proc.processor_id}")
    
    assert all([
        llm_proc.processor_id == "valid_id",
        mcp_proc.processor_id == "valid_id", 
        user_proc.processor_id == "valid_id",
        custom_proc.processor_id == "valid_id"
    ])
    
    print("[SUCCESS] Factory functions handle parameters correctly")
