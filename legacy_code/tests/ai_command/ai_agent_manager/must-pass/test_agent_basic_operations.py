#!/usr/bin/env python3
"""
AI Agent Manager - Basic Operations Tests (MUST PASS)
Tests fundamental agent lifecycle with real ZMQ servers
NO MOCKING - NO FALLBACKS - REAL TESTS ONLY
"""

import sys
import os
import pytest
import logging

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'ai_command'))
sys.path.append(r'C:\Users\DamonWeiss\PycharmProjects\Ethos-ZeroMQ')

from ai_agent_manager import AIAgentManager, AgentType, AgentStatus
from ethos_zeromq import ZeroMQEngine

logger = logging.getLogger(__name__)

class TestAgentBasicOperations:
    """Critical agent operations that MUST pass"""
    
    def setup_method(self):
        """Setup for each test"""
        self.manager = None
        self.spawned_agents = []
    
    def teardown_method(self):
        """Cleanup after each test"""
        if self.manager:
            # Terminate all spawned agents
            for agent_id in self.spawned_agents:
                try:
                    self.manager.terminate_agent(agent_id)
                except:
                    pass
    
    def test_agent_manager_initialization(self):
        """Test agent manager initializes with real ZMQ engine"""
        self.manager = AIAgentManager()
        
        assert self.manager is not None
        assert self.manager.zmq_engine is not None
        assert isinstance(self.manager.agents, dict)
        assert isinstance(self.manager.allocated_ports, set)
        assert len(self.manager.agents) == 0
    
    def test_zmq_engine_in_manager(self):
        """Test ZMQ engine is properly integrated"""
        self.manager = AIAgentManager()
        
        # Verify ZMQ engine is real
        assert hasattr(self.manager.zmq_engine, 'create_and_start_server')
        assert hasattr(self.manager.zmq_engine, 'sm')
    
    def test_spawn_strategic_advisor_agent(self):
        """Test spawning strategic advisor with real ZMQ server"""
        self.manager = AIAgentManager()
        
        agent_id = self.manager.spawn_agent(AgentType.STRATEGIC_ADVISOR)
        self.spawned_agents.append(agent_id)
        
        assert agent_id is not None
        assert agent_id in self.manager.agents
        
        agent = self.manager.agents[agent_id]
        assert agent.agent_type == AgentType.STRATEGIC_ADVISOR
        assert agent.status == AgentStatus.ACTIVE
        assert agent.zmq_server is not None
        assert agent.zmq_port is not None
        
        # Verify ZMQ server is real and running
        assert hasattr(agent.zmq_server, 'is_running')
        assert agent.zmq_server.is_running is True
        assert hasattr(agent.zmq_server, 'register_handler')
    
    def test_spawn_multiple_agents_different_types(self):
        """Test spawning multiple agents with different types"""
        self.manager = AIAgentManager()
        
        # Spawn different agent types
        advisor_id = self.manager.spawn_agent(AgentType.STRATEGIC_ADVISOR)
        planner_id = self.manager.spawn_agent(AgentType.MISSION_PLANNER)
        qa_id = self.manager.spawn_agent(AgentType.QUALITY_ASSURANCE)
        
        self.spawned_agents.extend([advisor_id, planner_id, qa_id])
        
        assert len(self.manager.agents) == 3
        assert len(self.manager.allocated_ports) == 3
        
        # Verify each agent has unique port and running ZMQ server
        ports = set()
        for agent_id in [advisor_id, planner_id, qa_id]:
            agent = self.manager.agents[agent_id]
            assert agent.zmq_server.is_running is True
            assert agent.zmq_port not in ports
            ports.add(agent.zmq_port)
    
    def test_terminate_agent_real_cleanup(self):
        """Test agent termination properly closes ZMQ server"""
        self.manager = AIAgentManager()
        
        agent_id = self.manager.spawn_agent(AgentType.STRATEGIC_ADVISOR)
        agent = self.manager.agents[agent_id]
        port = agent.zmq_port
        
        # Verify agent is running
        assert agent.zmq_server.is_running is True
        assert port in self.manager.allocated_ports
        
        # Terminate agent
        result = self.manager.terminate_agent(agent_id)
        
        assert result is True
        assert agent_id not in self.manager.agents
        assert port not in self.manager.allocated_ports
    
    def test_port_allocation_ranges(self):
        """Test port allocation follows defined ranges"""
        self.manager = AIAgentManager()
        
        # Test strategic advisor port range (5551-5559)
        advisor_id = self.manager.spawn_agent(AgentType.STRATEGIC_ADVISOR)
        self.spawned_agents.append(advisor_id)
        
        advisor = self.manager.agents[advisor_id]
        assert 5551 <= advisor.zmq_port <= 5559
        
        # Test mission planner port range (5560-5569)
        planner_id = self.manager.spawn_agent(AgentType.MISSION_PLANNER)
        self.spawned_agents.append(planner_id)
        
        planner = self.manager.agents[planner_id]
        assert 5560 <= planner.zmq_port <= 5569
