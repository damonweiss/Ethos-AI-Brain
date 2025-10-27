"""
Test AIBrain Basic Functionality - Must Pass
Light tests focusing on core AI brain features
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.core.ai_agent.ai_brain import AIBrain


def test_ai_brain_creation():
    """Test creating AIBrain"""
    brain = AIBrain(brain_name="test_brain")
    
    assert isinstance(brain, AIBrain)
    assert hasattr(brain, 'brain_name')


# def test_ai_brain_has_required_methods():
#     """Test AIBrain has expected methods"""
#     brain = AIBrain(brain_name="test_brain")
#     
#     # These methods should exist (will fail if they don't)
#     assert hasattr(brain, 'think') or hasattr(brain, 'process'), "AIBrain should have think or process method"


def test_ai_brain_attributes():
    """Test AIBrain basic attributes"""
    brain = AIBrain(brain_name="test_brain")
    
    # Check basic attributes exist
    assert hasattr(brain, 'brain_name')
    # Add more attribute checks as needed
