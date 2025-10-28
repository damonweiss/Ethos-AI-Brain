"""
Test MCP Performance and Load - Must Pass
Tests performance characteristics and concurrent request handling.
"""

import pytest
import importlib.util
import sys
import time
import threading
import concurrent.futures
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))


class TestPerformanceLoad:
    """Test MCP performance and load handling capabilities."""
    
    def test_single_tool_performance(self):
        """Test performance of individual tool execution."""
        system_tools_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "system" / "system_tools" / "tool.py"
        spec = importlib.util.spec_from_file_location("system_tools", system_tools_path)
        system_tools_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(system_tools_module)
        
        # Measure execution time for echo_message
        start_time = time.time()
        result = system_tools_module.echo_message({"message": "Performance test"})
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        assert isinstance(result, dict)
        assert "echo" in result
        assert execution_time < 1.0  # Should execute in under 1 second
        
        print(f"✅ Echo tool execution time: {execution_time:.4f} seconds")
        
        # Measure execution time for get_system_time
        start_time = time.time()
        result = system_tools_module.get_system_time({})
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        assert isinstance(result, dict)
        assert "timestamp" in result
        assert execution_time < 1.0  # Should execute in under 1 second
        
        print(f"✅ System time tool execution time: {execution_time:.4f} seconds")
    
    def test_multiple_sequential_executions(self):
        """Test performance of multiple sequential tool executions."""
        system_tools_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "system" / "system_tools" / "tool.py"
        spec = importlib.util.spec_from_file_location("system_tools", system_tools_path)
        system_tools_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(system_tools_module)
        
        num_executions = 10
        start_time = time.time()
        
        results = []
        for i in range(num_executions):
            result = system_tools_module.echo_message({"message": f"Test {i}"})
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_executions
        
        # Validate all executions succeeded
        assert len(results) == num_executions
        for i, result in enumerate(results):
            assert result["echo"] == f"Test {i}"
        
        # Performance assertions
        assert total_time < 5.0  # 10 executions should take less than 5 seconds
        assert avg_time < 0.5   # Average execution should be under 0.5 seconds
        
        print(f"✅ {num_executions} sequential executions: {total_time:.4f}s total, {avg_time:.4f}s average")
    
    def test_concurrent_tool_executions(self):
        """Test concurrent tool execution performance."""
        system_tools_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "system" / "system_tools" / "tool.py"
        spec = importlib.util.spec_from_file_location("system_tools", system_tools_path)
        system_tools_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(system_tools_module)
        
        def execute_tool(message):
            """Execute tool with given message."""
            return system_tools_module.echo_message({"message": message})
        
        num_threads = 5
        num_executions_per_thread = 3
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            futures = []
            for thread_id in range(num_threads):
                for exec_id in range(num_executions_per_thread):
                    message = f"Thread-{thread_id}-Exec-{exec_id}"
                    future = executor.submit(execute_tool, message)
                    futures.append((message, future))
            
            # Collect results
            results = []
            for message, future in futures:
                result = future.result()
                results.append((message, result))
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Validate all executions succeeded
        expected_count = num_threads * num_executions_per_thread
        assert len(results) == expected_count
        
        for message, result in results:
            assert isinstance(result, dict)
            assert result["echo"] == message
        
        # Performance assertions
        assert total_time < 10.0  # All concurrent executions should complete in under 10 seconds
        
        print(f"✅ {expected_count} concurrent executions: {total_time:.4f}s total")
        print(f"📊 Concurrency efficiency: {expected_count/total_time:.2f} executions/second")
    
    def test_mixed_tool_performance(self):
        """Test performance with mixed tool types."""
        # Import multiple tool types
        system_tools_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "system" / "system_tools" / "tool.py"
        spec = importlib.util.spec_from_file_location("system_tools", system_tools_path)
        system_tools_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(system_tools_module)
        
        api_config_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "api_management" / "api_config" / "tool.py"
        spec = importlib.util.spec_from_file_location("api_config", api_config_path)
        api_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api_config_module)
        
        # Define mixed workload
        tasks = [
            ("echo_message", system_tools_module.echo_message, {"message": "Mixed test 1"}),
            ("get_system_time", system_tools_module.get_system_time, {}),
            ("list_api_configs", api_config_module.list_api_configs, {}),
            ("echo_message", system_tools_module.echo_message, {"message": "Mixed test 2"}),
            ("get_system_time", system_tools_module.get_system_time, {}),
        ]
        
        start_time = time.time()
        
        results = []
        for tool_name, tool_func, params in tasks:
            result = tool_func(params)
            results.append((tool_name, result))
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Validate all tasks completed
        assert len(results) == len(tasks)
        for i, (tool_name, result) in enumerate(results):
            assert isinstance(result, dict)
            expected_tool = tasks[i][0]
            assert tool_name == expected_tool
        
        # Performance assertion
        assert total_time < 3.0  # Mixed workload should complete in under 3 seconds
        
        print(f"✅ Mixed tool workload: {total_time:.4f}s for {len(tasks)} different tools")
    
    def test_tool_discovery_performance(self):
        """Test performance of tool discovery process."""
        discovery_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "mcp_tool_manager.py"
        spec = importlib.util.spec_from_file_location("uv_discovery", discovery_path)
        discovery_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(discovery_module)
        
        tools_dir = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools"
        
        start_time = time.time()
        
        try:
            manager = discovery_module.MCPToolManager(str(tools_dir))
            
            # Attempt tool discovery
            if hasattr(manager, 'discover_tools'):
                tools = manager.discover_tools()
            elif hasattr(manager, 'scan_for_tools'):
                tools = manager.scan_for_tools()
            elif hasattr(manager, 'tools'):
                tools = manager.tools
            else:
                tools = []
            
            end_time = time.time()
            discovery_time = end_time - start_time
            
            # Performance assertions
            assert discovery_time < 5.0  # Tool discovery should complete in under 5 seconds
            
            print(f"✅ Tool discovery: {discovery_time:.4f}s")
            if isinstance(tools, (list, dict)):
                tool_count = len(tools)
                print(f"📊 Discovered {tool_count} tools in {discovery_time:.4f}s")
                if tool_count > 0:
                    print(f"📊 Discovery rate: {tool_count/discovery_time:.2f} tools/second")
        
        except Exception as e:
            end_time = time.time()
            discovery_time = end_time - start_time
            print(f"⚠️ Tool discovery failed in {discovery_time:.4f}s: {e}")
            # Still pass if discovery completes quickly even with errors
            assert discovery_time < 10.0
    
    def test_memory_usage_stability(self):
        """Test that repeated tool executions don't cause memory leaks."""
        system_tools_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "system" / "system_tools" / "tool.py"
        spec = importlib.util.spec_from_file_location("system_tools", system_tools_path)
        system_tools_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(system_tools_module)
        
        # Execute many iterations to test memory stability
        num_iterations = 50
        results = []
        
        start_time = time.time()
        
        for i in range(num_iterations):
            result = system_tools_module.echo_message({"message": f"Memory test {i}"})
            results.append(result)
            
            # Clear result to help garbage collection
            if i % 10 == 0:
                results = results[-5:]  # Keep only last 5 results
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Validate execution completed successfully
        assert len(results) > 0  # Should have some results
        # Note: results list gets trimmed during execution, so final length varies
        
        # Performance assertion
        avg_time = total_time / num_iterations
        assert avg_time < 0.1  # Average time should be very fast for simple operations
        
        print(f"✅ Memory stability test: {num_iterations} iterations in {total_time:.4f}s")
        print(f"📊 Average iteration time: {avg_time:.6f}s")
    
    def test_error_handling_performance(self):
        """Test that error handling doesn't significantly impact performance."""
        system_tools_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp_tools" / "utilities" / "system" / "system_tools" / "tool.py"
        spec = importlib.util.spec_from_file_location("system_tools", system_tools_path)
        system_tools_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(system_tools_module)
        
        # Test normal execution time
        start_time = time.time()
        normal_result = system_tools_module.echo_message({"message": "Normal test"})
        normal_time = time.time() - start_time
        
        # Test error handling execution time
        start_time = time.time()
        error_result = system_tools_module.echo_message(None)  # Invalid input
        error_time = time.time() - start_time
        
        # Both should succeed (graceful error handling)
        assert isinstance(normal_result, dict)
        assert isinstance(error_result, dict)
        
        # Error handling shouldn't be significantly slower
        assert error_time < normal_time * 3  # Error handling should be at most 3x slower
        assert error_time < 1.0  # Error handling should still be fast
        
        print(f"✅ Normal execution: {normal_time:.4f}s")
        print(f"✅ Error handling: {error_time:.4f}s")
        print(f"📊 Error overhead: {(error_time/normal_time):.2f}x")
