#!/usr/bin/env python3
"""
Mission Parser - Basic Parsing Tests (MUST PASS)
Tests fundamental mission parsing and validation
NO MOCKING - NO FALLBACKS - REAL TESTS ONLY
"""

import sys
import os
import pytest
import logging
from datetime import datetime

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'ai_command'))

from mission_parser import MissionParser, Mission, MissionStatus, MissionPriority

logger = logging.getLogger(__name__)

class TestMissionBasicParsing:
    """Critical mission parsing tests that MUST pass"""
    
    def setup_method(self):
        """Setup for each test"""
        self.parser = MissionParser()
    
    def test_mission_parser_initialization(self):
        """Test mission parser initializes properly"""
        assert self.parser is not None
        assert hasattr(self.parser, 'mission_keywords')
        assert hasattr(self.parser, 'mission_relationship_graph')
        assert hasattr(self.parser, 'priority_dependency_graph')
        assert hasattr(self.parser, 'resource_conflict_graph')
        assert hasattr(self.parser, 'mission_similarity_graph')
    
    def test_parse_simple_mission(self):
        """Test parsing a simple mission objective"""
        objective = "Analyze system performance metrics"
        
        mission = self.parser.parse_mission(objective)
        
        assert isinstance(mission, Mission)
        assert mission.objective == objective
        assert mission.id is not None
        assert mission.status == MissionStatus.RECEIVED
        assert isinstance(mission.created_at, datetime)
        assert isinstance(mission.priority, MissionPriority)
    
    def test_parse_urgent_mission(self):
        """Test parsing urgent mission gets high priority"""
        objective = "Urgent: Fix critical security vulnerability"
        
        mission = self.parser.parse_mission(objective)
        
        assert mission.priority == MissionPriority.HIGH
        assert 'urgent' in objective.lower()
    
    def test_parse_critical_mission(self):
        """Test parsing critical mission gets critical priority"""
        objective = "Critical system failure - immediate response required"
        
        mission = self.parser.parse_mission(objective)
        
        assert mission.priority == MissionPriority.CRITICAL
    
    def test_parse_routine_mission(self):
        """Test parsing routine mission gets low priority"""
        objective = "Routine maintenance check of backup systems"
        
        mission = self.parser.parse_mission(objective)
        
        assert mission.priority == MissionPriority.LOW
    
    def test_mission_id_uniqueness(self):
        """Test each mission gets unique ID"""
        objective1 = "First mission"
        objective2 = "Second mission"
        
        mission1 = self.parser.parse_mission(objective1)
        mission2 = self.parser.parse_mission(objective2)
        
        assert mission1.id != mission2.id
        assert len(mission1.id) > 0
        assert len(mission2.id) > 0
    
    def test_mission_validation_empty_objective(self):
        """Test mission validation fails for empty objective"""
        try:
            # This should raise an exception for empty objective
            mission = self.parser.parse_mission("")
            assert False, "Expected ValueError for empty objective"
        except ValueError as e:
            assert "empty" in str(e).lower() or "cannot be empty" in str(e).lower()
    
    def test_mission_validation_short_objective(self):
        """Test mission validation fails for too short objective"""
        try:
            # This should raise an exception for short objective
            mission = self.parser.parse_mission("short")
            assert False, "Expected ValueError for short objective"
        except ValueError as e:
            assert "short" in str(e).lower() or "minimum" in str(e).lower()
    
    def test_priority_keyword_detection(self):
        """Test priority keywords are properly detected"""
        test_cases = [
            ("analyze data", MissionPriority.NORMAL),
            ("urgent fix needed", MissionPriority.HIGH),
            ("critical system down", MissionPriority.CRITICAL),
            ("routine check", MissionPriority.LOW),
            ("emergency response", MissionPriority.CRITICAL)
        ]
        
        for objective, expected_priority in test_cases:
            mission = self.parser.parse_mission(objective)
            assert mission.priority == expected_priority, f"Failed for: {objective}"
    
    def test_mission_status_lifecycle(self):
        """Test mission status can be updated"""
        mission = self.parser.parse_mission("Test mission")
        
        # Initial status
        assert mission.status == MissionStatus.RECEIVED
        
        # Update status
        mission.update_status(MissionStatus.EXECUTING)
        assert mission.status == MissionStatus.EXECUTING
        
        mission.update_status(MissionStatus.COMPLETED)
        assert mission.status == MissionStatus.COMPLETED
