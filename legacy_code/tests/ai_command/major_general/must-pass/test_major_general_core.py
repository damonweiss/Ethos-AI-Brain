#!/usr/bin/env python3
"""
Major General - Core Functionality Tests (MUST PASS)
Tests fundamental Major General operations with real components
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
from mission_parser import MissionPriority

logger = logging.getLogger(__name__)

class TestMajorGeneralCore:
    """Critical Major General tests that MUST pass"""
    
    def setup_method(self):
        """Setup for each test"""
        self.major_general = None
        self.spawned_agents = []
    
    def teardown_method(self):
        """Cleanup after each test"""
        if self.major_general:
            # Clean up any spawned agents
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
    
    def test_major_general_initialization(self):
        """Test Major General initializes with all components"""
        self.major_general = MajorGeneral()
        
        assert self.major_general is not None
        assert self.major_general.agent_id is not None
        assert hasattr(self.major_general, 'mission_parser')
        assert hasattr(self.major_general, 'agent_manager')
        assert hasattr(self.major_general, 'command_network')
        assert hasattr(self.major_general, 'meta_reasoning')
        
        # Verify components are real objects
        assert self.major_general.mission_parser is not None
        assert self.major_general.agent_manager is not None
        assert self.major_general.meta_reasoning is not None
    
    def test_major_general_agent_registration(self):
        """Test Major General registers itself as an agent"""
        self.major_general = MajorGeneral()
        
        # Major General should register itself
        mg_agent = self.major_general.agent_manager.get_agent(self.major_general.agent_id)
        assert mg_agent is not None
        assert mg_agent.agent_type.value == "major_general"
        assert mg_agent.zmq_server is not None
        assert mg_agent.zmq_server.is_running is True
    
    @pytest.mark.asyncio
    async def test_receive_mission_basic(self):
        """Test Major General can receive and parse mission"""
        self.major_general = MajorGeneral()
        
        mission_objective = "Analyze system performance and generate report"
        
        mission_id = await self.major_general.receive_mission(mission_objective)
        
        assert isinstance(mission_id, str)
        assert mission_id is not None
        assert len(mission_id) > 0
        
        # Verify mission was parsed and stored
        assert mission_id in self.major_general.active_missions
        
        mission = self.major_general.active_missions[mission_id]
        assert mission.objective == mission_objective
    
    @pytest.mark.asyncio
    async def test_mission_priority_handling(self):
        """Test Major General handles different mission priorities"""
        self.major_general = MajorGeneral()
        
        # Test critical mission
        critical_mission_id = await self.major_general.receive_mission("Critical: System failure detected")
        critical_mission = self.major_general.active_missions[critical_mission_id]
        
        assert critical_mission.priority == MissionPriority.CRITICAL
        
        # Test routine mission
        routine_mission_id = await self.major_general.receive_mission("Routine: Update documentation")
        routine_mission = self.major_general.active_missions[routine_mission_id]
        
        assert routine_mission.priority == MissionPriority.LOW
    
    def test_component_integration(self):
        """Test all components are properly integrated"""
        self.major_general = MajorGeneral()
        
        # Test mission parser integration
        assert self.major_general.mission_parser is not None
        test_mission = self.major_general.mission_parser.parse_mission("Test integration")
        assert test_mission is not None
        
        # Test agent manager integration
        assert self.major_general.agent_manager is not None
        assert self.major_general.agent_manager.zmq_engine is not None
        
        # Test meta reasoning integration
        assert self.major_general.meta_reasoning is not None
    
    def test_mission_execution_graphs(self):
        """Test Major General creates execution graphs"""
        self.major_general = MajorGeneral()
        
        # Verify execution graphs dictionary exists
        assert hasattr(self.major_general, 'mission_execution_graphs')
        assert isinstance(self.major_general.mission_execution_graphs, dict)
    
    @pytest.mark.asyncio
    async def test_multiple_missions_handling(self):
        """Test Major General can handle multiple concurrent missions"""
        self.major_general = MajorGeneral()
        
        # Submit multiple missions
        mission1_id = await self.major_general.receive_mission("First mission: Analyze data")
        mission2_id = await self.major_general.receive_mission("Second mission: Generate report")
        mission3_id = await self.major_general.receive_mission("Third mission: Update systems")
        
        # Verify all missions are tracked
        assert len(self.major_general.active_missions) == 3
        assert mission1_id in self.major_general.active_missions
        assert mission2_id in self.major_general.active_missions
        assert mission3_id in self.major_general.active_missions
        
        # Verify each mission has unique ID
        mission_ids = [mission1_id, mission2_id, mission3_id]
        assert len(set(mission_ids)) == 3
