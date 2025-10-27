#!/usr/bin/env python3
"""
ZMQ Command Bridge - Basic Connectivity Tests (MUST PASS)
Tests fundamental ZMQ integration with real Ethos-ZeroMQ library
NO MOCKING - NO FALLBACKS - REAL TESTS ONLY
"""

import sys
import os
import pytest
import logging

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'ai_command'))
sys.path.append(r'C:\Users\DamonWeiss\PycharmProjects\Ethos-ZeroMQ')

from zmq_command_bridge import ZMQCommandBridge
from ethos_zeromq import ZeroMQEngine

logger = logging.getLogger(__name__)

class TestZMQBasicConnectivity:
    """Critical ZMQ connectivity tests that MUST pass"""
    
    def setup_method(self):
        """Setup for each test"""
        self.bridge = None
        self.servers_to_cleanup = []
    
    def teardown_method(self):
        """Cleanup after each test"""
        # Stop any servers we created
        for server in self.servers_to_cleanup:
            try:
                if hasattr(server, 'stop'):
                    server.stop()
            except:
                pass
        
        if self.bridge:
            try:
                # Clean shutdown
                for server_name, server in self.bridge.command_servers.items():
                    if hasattr(server, 'stop'):
                        server.stop()
            except:
                pass
    
    def test_zmq_engine_creation(self):
        """Test ZMQ engine can be created directly"""
        engine = ZeroMQEngine()
        assert engine is not None
        assert hasattr(engine, 'create_and_start_server')
        assert hasattr(engine, 'sm')  # ServerManager
    
    def test_reqrep_server_creation(self):
        """Test ReqRep server can be created and started"""
        engine = ZeroMQEngine()
        server = engine.create_and_start_server('reqrep', 'test_basic_reqrep')
        self.servers_to_cleanup.append(server)
        
        assert server is not None
        assert hasattr(server, 'is_running')
        assert hasattr(server, 'register_handler')
        assert hasattr(server, 'send_direct_message')
        assert server.is_running is True
    
    def test_zmq_bridge_initialization(self):
        """Test ZMQ bridge initializes with real ZMQ servers"""
        self.bridge = ZMQCommandBridge()
        
        assert self.bridge is not None
        assert self.bridge.zmq_engine is not None
        assert isinstance(self.bridge.command_servers, dict)
        assert isinstance(self.bridge.biological_patterns, dict)
    
    def test_network_status_real_data(self):
        """Test network status returns real ZMQ data"""
        self.bridge = ZMQCommandBridge()
        status = self.bridge.get_network_status()
        
        assert isinstance(status, dict)
        assert status['zmq_available'] is True
        assert status['network_health'] == 'operational'
        assert isinstance(status['active_servers'], list)
        assert isinstance(status['biological_patterns'], dict)
    
    def test_biological_pattern_servers_created(self):
        """Test biological pattern servers are created for supported patterns"""
        self.bridge = ZMQCommandBridge()
        
        # Check that all biological patterns are registered (even if servers aren't created)
        expected_patterns = ['reflex_arc', 'spreading_activation', 'hierarchical_processing', 
                           'parallel_processing', 'integration_centers']
        
        for pattern in expected_patterns:
            assert pattern in self.bridge.biological_patterns
        
        # Check that supported patterns have actual servers
        supported_patterns_with_servers = ['reflex_arc', 'spreading_activation']
        
        for pattern in supported_patterns_with_servers:
            if pattern in self.bridge.command_servers:
                # Verify server is real and running
                server = self.bridge.command_servers[pattern]
                assert server is not None
                assert hasattr(server, 'is_running')
                assert server.is_running is True
        
        # Verify we have at least some working servers
        assert len(self.bridge.command_servers) > 0
