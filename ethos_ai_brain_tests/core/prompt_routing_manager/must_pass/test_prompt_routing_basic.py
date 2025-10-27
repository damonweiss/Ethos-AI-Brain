"""
Test PromptRoutingManager Basic Functionality - Must Pass
Light tests focusing on core routing features
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.core.prompt_routing_manager.prompt_routing_manager import PromptRoutingManager


def test_routing_manager_creation():
    """Test creating PromptRoutingManager"""
    router = PromptRoutingManager()
    
    assert isinstance(router, PromptRoutingManager)
    assert hasattr(router, 'prompt_manager')
    assert hasattr(router, 'schema_registry')


def test_prompt_execution():
    """Test prompt execution routing"""
    router = PromptRoutingManager()
    
    # Test if execute_prompt method exists
    assert hasattr(router, 'execute_prompt')
    
    # Test basic execution - this tests the integration between PromptManager and PromptRoutingManager
    result = router.execute_prompt("general_analysis", {"text": "test"})
    
    # Check if we get a proper response structure
    assert isinstance(result, dict)
    assert "success" in result
    
    # Expected to fail due to missing inference engines, but should return proper error structure
    if not result.get("success"):
        assert "error" in result
