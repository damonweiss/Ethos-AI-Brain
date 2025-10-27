"""
Test AIAgent Basic Functionality - Must Pass
Light tests focusing on core AI agent features
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.core.ai_agent.ai_agent import AIAgent


def test_ai_agent_creation():
    """Test creating AIAgent"""
    agent = AIAgent(agent_id="test_agent", role="analyst")
    
    assert isinstance(agent, AIAgent)
    assert agent.agent_id == "test_agent"
    assert agent.role == "analyst"


def test_ai_agent_attributes():
    """Test AIAgent basic attributes and methods"""
    agent = AIAgent(agent_id="test_agent", role="analyst")
    
    # Check for basic attributes
    assert hasattr(agent, 'agent_id')
    assert hasattr(agent, 'role')
    assert hasattr(agent, 'is_running')
    assert agent.is_running == False  # Should start as not running
    
    # Check for brain
    assert hasattr(agent, 'brain')
    assert agent.brain is not None
