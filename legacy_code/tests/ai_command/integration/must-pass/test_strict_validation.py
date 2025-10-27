#!/usr/bin/env python3
"""
Strict Validation Tests - NO MOCKS, NO FALLBACKS ALLOWED
Tests that will FAIL if real dependencies are missing or mocked
"""

import sys
import os
import pytest
import logging

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'ai_command'))
sys.path.append(r'C:\Users\DamonWeiss\PycharmProjects\Ethos-ZeroMQ')

from major_general import MajorGeneral
from zmq_command_bridge import ZMQCommandBridge
from ai_agent_manager import AIAgentManager, AgentType

logger = logging.getLogger(__name__)

class TestStrictValidation:
    """STRICT tests that will FAIL if mocks/fallbacks are used"""
    
    def setup_method(self):
        """Setup for each test"""
        self.components_to_cleanup = []
    
    def teardown_method(self):
        """Cleanup after each test"""
        for component in self.components_to_cleanup:
            try:
                if hasattr(component, 'agent_manager') and hasattr(component.agent_manager, 'zmq_engine'):
                    component.agent_manager.zmq_engine.stop_all_servers()
            except:
                pass
    
    def test_meta_reasoning_engine_is_real(self):
        """FAIL if MetaReasoningEngine is mocked"""
        major_general = MajorGeneral()
        self.components_to_cleanup.append(major_general)
        
        # Check if we got the mock class
        reasoning_result = major_general.meta_reasoning
        
        # The mock returns {"analysis": "mock"} - this should NOT happen in real tests
        assert reasoning_result is not None, "MetaReasoningEngine should exist"
        
        # Check if it's the mock class by testing its behavior
        import asyncio
        async def test_reasoning():
            result = await reasoning_result.reason("test goal", None)
            return result
        
        result = asyncio.run(test_reasoning())
        
        # STRICT: If we get the mock response, FAIL the test
        assert result != {"analysis": "mock"}, "MetaReasoningEngine is MOCKED! This hides real failures!"
        
        # Real MetaReasoningEngine should return more complex structure
        assert isinstance(result, dict), "MetaReasoningEngine should return dict"
        assert "analysis" in result, "MetaReasoningEngine should have analysis field"
    
    def test_zmq_servers_have_required_methods(self):
        """FAIL if ZMQ servers don't have required methods"""
        bridge = ZMQCommandBridge()
        self.components_to_cleanup.append(bridge)
        
        # Check that servers actually have the methods we expect
        for server_name, server in bridge.command_servers.items():
            assert server is not None, f"Server {server_name} should exist"
            
            # STRICT: These methods MUST exist, no fallbacks allowed
            assert hasattr(server, 'send_direct_message'), f"Server {server_name} missing send_direct_message"
            assert hasattr(server, 'route_message'), f"Server {server_name} missing route_message"
            assert hasattr(server, 'register_handler'), f"Server {server_name} missing register_handler"
            
            # STRICT: Methods must be callable
            assert callable(server.send_direct_message), f"send_direct_message not callable on {server_name}"
            assert callable(server.route_message), f"route_message not callable on {server_name}"
            assert callable(server.register_handler), f"register_handler not callable on {server_name}"
    
    def test_zmq_message_routing_is_real(self):
        """FAIL if ZMQ message routing returns fallback responses"""
        bridge = ZMQCommandBridge()
        self.components_to_cleanup.append(bridge)
        
        # Test command sending
        test_command = {"type": "test", "data": "validation"}
        
        import asyncio
        async def test_command_sending():
            response = await bridge.send_command('major_general', test_command)
            return response
        
        response = asyncio.run(test_command_sending())
        
        # STRICT: These responses indicate fallbacks/mocks - FAIL if found
        forbidden_responses = [
            {"status": "no_handler"},
            {"status": "mock_response"},
            {"status": "mock_broadcast"},
            {"status": "mock_expert_response"},
            {"status": "mock"}
        ]
        
        for forbidden in forbidden_responses:
            for key, value in forbidden.items():
                if key in response and response[key] == value:
                    pytest.fail(f"ZMQ routing returned fallback response: {response}. This hides real failures!")
        
        # Response should be real ZMQ response, not a fallback
        assert isinstance(response, dict), "Should return dict response"
        assert response != {}, "Should not return empty response"
    
    def test_biological_patterns_are_real_servers(self):
        """FAIL if biological patterns are not real ZMQ servers"""
        bridge = ZMQCommandBridge()
        self.components_to_cleanup.append(bridge)
        
        # Check that biological patterns have real servers
        expected_patterns = ['reflex_arc', 'spreading_activation']
        
        for pattern in expected_patterns:
            assert pattern in bridge.biological_patterns, f"Pattern {pattern} should be registered"
            assert pattern in bridge.command_servers, f"Pattern {pattern} should have server"
            
            server = bridge.command_servers[pattern]
            assert server is not None, f"Pattern {pattern} server should exist"
            assert hasattr(server, 'is_running'), f"Pattern {pattern} server should have is_running"
            assert server.is_running is True, f"Pattern {pattern} server should be running"
    
    def test_agent_spawning_creates_real_zmq_servers(self):
        """FAIL if agent spawning doesn't create real ZMQ servers"""
        manager = AIAgentManager()
        self.components_to_cleanup.append(manager)
        
        # Spawn an agent
        agent_id = manager.spawn_agent(AgentType.STRATEGIC_ADVISOR)
        
        # STRICT: Agent must have real ZMQ server
        agent = manager.get_agent(agent_id)
        assert agent is not None, "Agent should exist after spawning"
        assert agent.zmq_server is not None, "Agent should have ZMQ server"
        assert hasattr(agent.zmq_server, 'is_running'), "ZMQ server should have is_running"
        assert agent.zmq_server.is_running is True, "ZMQ server should be running"
        
        # STRICT: ZMQ server must have required methods
        assert hasattr(agent.zmq_server, 'send_direct_message'), "Agent ZMQ server missing send_direct_message"
        assert hasattr(agent.zmq_server, 'route_message'), "Agent ZMQ server missing route_message"
        
        # Clean up
        manager.terminate_agent(agent_id)
    
    def test_no_pattern_unavailable_responses(self):
        """FAIL if biological patterns return 'pattern_unavailable'"""
        bridge = ZMQCommandBridge()
        self.components_to_cleanup.append(bridge)
        
        # Test available patterns
        available_patterns = ['reflex_arc', 'spreading_activation']
        
        for pattern in available_patterns:
            import asyncio
            async def test_pattern():
                response = await bridge.execute_biological_pattern(pattern, {"test": "data"})
                return response
            
            response = asyncio.run(test_pattern())
            
            # STRICT: Should not return unavailable status for available patterns
            assert response.get('status') != 'pattern_unavailable', f"Pattern {pattern} should be available"
            assert response.get('status') != 'no_handler', f"Pattern {pattern} should have handler"
            assert response.get('status') != 'mock', f"Pattern {pattern} should not be mocked"
    
    def test_import_paths_are_valid(self):
        """FAIL if import paths are broken"""
        # Test ZMQ import
        try:
            from ethos_zeromq import ZeroMQEngine
            engine = ZeroMQEngine()
            assert engine is not None, "ZeroMQEngine should be importable and creatable"
        except ImportError as e:
            pytest.fail(f"ZeroMQ import failed: {e}. This will cause fallbacks!")
        
        # Test meta-reasoning import (this might fail, but we should know about it)
        try:
            sys.path.append(r'C:\Users\DamonWeiss\NodeProjects\Ethos-Application\.Project Brain\09 - MCP\production')
            from meta_reasoning_engine import MetaReasoningEngine
            engine = MetaReasoningEngine()
            assert engine is not None, "MetaReasoningEngine should be importable"
        except ImportError as e:
            pytest.fail(f"MetaReasoningEngine import failed: {e}. Using mock instead!")
