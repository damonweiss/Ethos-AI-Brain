#!/usr/bin/env python3
"""
Integration - End-to-End Basic Tests (MUST PASS)
Tests complete AI Command System integration with real components
NO MOCKING - NO FALLBACKS - REAL TESTS ONLY
"""

import sys
import os
import pytest
import logging
import asyncio

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'ai_command'))
sys.path.append(r'C:\Users\DamonWeiss\PycharmProjects\Ethos-ZeroMQ')

from major_general import MajorGeneral
from ai_agent_manager import AgentType
from mission_parser import MissionPriority

logger = logging.getLogger(__name__)

class TestEndToEndBasic:
    """Critical end-to-end integration tests that MUST pass"""
    
    def setup_method(self):
        """Setup for each test"""
        self.major_general = None
        self.spawned_agents = []
    
    def teardown_method(self):
        """Cleanup after each test"""
        if self.major_general:
            # Clean up spawned agents
            for agent_id in self.spawned_agents:
                try:
                    self.major_general.agent_manager.terminate_agent(agent_id)
                except:
                    pass
            
            # Clean up Major General agent
            try:
                if hasattr(self.major_general, 'agent_id') and self.major_general.agent_id:
                    self.major_general.agent_manager.terminate_agent(self.major_general.agent_id)
            except:
                pass
            
            # Stop all ZMQ servers in the engine
            try:
                if hasattr(self.major_general.agent_manager, 'zmq_engine'):
                    self.major_general.agent_manager.zmq_engine.stop_all_servers()
            except:
                pass
    
    def test_complete_system_initialization(self):
        """Test entire AI Command System initializes correctly"""
        self.major_general = MajorGeneral()
        
        # Verify Major General
        assert self.major_general is not None
        assert self.major_general.agent_id is not None
        
        # Verify all subsystems
        assert self.major_general.mission_parser is not None
        assert self.major_general.agent_manager is not None
        assert self.major_general.meta_reasoning is not None
        
        # Verify ZMQ infrastructure through agent manager
        assert self.major_general.agent_manager.zmq_engine is not None
        
        # Verify Major General is registered as agent
        mg_agent = self.major_general.agent_manager.get_agent(self.major_general.agent_id)
        assert mg_agent is not None
        assert mg_agent.zmq_server.is_running is True
    
    @pytest.mark.asyncio
    async def test_mission_to_agent_workflow(self):
        """Test complete workflow: mission → parsing → agent spawning"""
        self.major_general = MajorGeneral()
        
        # Step 1: Receive mission
        mission_objective = "Strategic analysis of system architecture"
        mission_id = await self.major_general.receive_mission(mission_objective)
        
        assert isinstance(mission_id, str)
        assert mission_id is not None
        
        # Step 2: Verify mission was parsed correctly
        mission = self.major_general.active_missions[mission_id]
        assert mission.objective == mission_objective
        assert mission.priority == MissionPriority.NORMAL
        
        # Step 3: Spawn strategic advisor agent
        advisor_id = self.major_general.agent_manager.spawn_agent(AgentType.STRATEGIC_ADVISOR)
        self.spawned_agents.append(advisor_id)
        
        # Step 4: Verify agent is operational
        advisor = self.major_general.agent_manager.get_agent(advisor_id)
        assert advisor is not None
        assert advisor.zmq_server.is_running is True
        assert advisor.agent_type == AgentType.STRATEGIC_ADVISOR
    
    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self):
        """Test coordination between multiple agents"""
        self.major_general = MajorGeneral()
        
        # Receive complex mission
        mission_result = await self.major_general.receive_mission(
            "Complex project: Analyze, plan, and document system improvements"
        )
        
        # Spawn multiple agents for different aspects
        advisor_id = self.major_general.agent_manager.spawn_agent(AgentType.STRATEGIC_ADVISOR)
        planner_id = self.major_general.agent_manager.spawn_agent(AgentType.MISSION_PLANNER)
        doc_id = self.major_general.agent_manager.spawn_agent(AgentType.DOCUMENTATION_SPECIALIST)
        
        self.spawned_agents.extend([advisor_id, planner_id, doc_id])
        
        # Verify all agents are operational
        agents = [
            self.major_general.agent_manager.get_agent(advisor_id),
            self.major_general.agent_manager.get_agent(planner_id),
            self.major_general.agent_manager.get_agent(doc_id)
        ]
        
        for agent in agents:
            assert agent is not None
            assert agent.zmq_server.is_running is True
        
        # Verify agents have different ports
        ports = [agent.zmq_port for agent in agents]
        assert len(set(ports)) == 3  # All unique ports
    
    def test_agent_zmq_integration(self):
        """Test agents are integrated with ZMQ system"""
        self.major_general = MajorGeneral()
        
        # Spawn an agent to test ZMQ integration
        advisor_id = self.major_general.agent_manager.spawn_agent(AgentType.STRATEGIC_ADVISOR)
        self.spawned_agents.append(advisor_id)
        
        # Verify agent has ZMQ server
        advisor = self.major_general.agent_manager.get_agent(advisor_id)
        assert advisor.zmq_server is not None
        assert advisor.zmq_server.is_running is True
        assert advisor.zmq_port is not None
    
    @pytest.mark.asyncio
    async def test_priority_based_mission_handling(self):
        """Test system handles missions based on priority"""
        self.major_general = MajorGeneral()
        
        # Submit missions with different priorities
        critical_mission_id = await self.major_general.receive_mission("Critical: Security breach detected")
        normal_mission_id = await self.major_general.receive_mission("Analyze quarterly reports")
        routine_mission_id = await self.major_general.receive_mission("Routine: Update logs")
        
        # Verify missions are tracked
        assert len(self.major_general.active_missions) == 3
        
        # Verify priority assignment
        critical_mission = self.major_general.active_missions[critical_mission_id]
        normal_mission = self.major_general.active_missions[normal_mission_id]
        routine_mission = self.major_general.active_missions[routine_mission_id]
        
        assert critical_mission.priority == MissionPriority.CRITICAL
        assert normal_mission.priority == MissionPriority.NORMAL
        assert routine_mission.priority == MissionPriority.LOW
    
    def test_networkx_intelligence_integration(self):
        """Test NetworkX intelligence is integrated across components"""
        self.major_general = MajorGeneral()
        
        # Verify Mission Parser has NetworkX graphs
        assert hasattr(self.major_general.mission_parser, 'mission_relationship_graph')
        assert hasattr(self.major_general.mission_parser, 'priority_dependency_graph')
        
        # Verify Agent Manager has NetworkX graphs
        assert hasattr(self.major_general.agent_manager, 'agent_dependency_graph')
        assert hasattr(self.major_general.agent_manager, 'capability_graph')
        
        # Verify Major General has execution graphs
        assert hasattr(self.major_general, 'mission_execution_graphs')
    
    def test_real_zmq_communication_infrastructure(self):
        """Test real ZMQ communication infrastructure is operational"""
        self.major_general = MajorGeneral()
        
        # Verify ZMQ engine is real
        zmq_engine = self.major_general.agent_manager.zmq_engine
        assert zmq_engine is not None
        assert hasattr(zmq_engine, 'sm')  # ServerManager
        
        # Verify Major General agent has ZMQ server
        mg_agent = self.major_general.agent_manager.get_agent(self.major_general.agent_id)
        assert mg_agent is not None
        assert mg_agent.zmq_server is not None
        assert hasattr(mg_agent.zmq_server, 'is_running')
        assert mg_agent.zmq_server.is_running is True
        assert hasattr(mg_agent.zmq_server, 'register_handler')
        assert hasattr(mg_agent.zmq_server, 'send_direct_message')
