"""
Test PromptManager Basic Functionality - Must Pass
Light tests focusing on core prompt management features
"""

import pytest
import sys
from pathlib import Path
from pydantic import BaseModel, Field

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.core.prompt_manager.prompt_manager import PromptManager


# Test schemas to avoid warnings
class GreetingSchema(BaseModel):
    greeting: str = Field(..., description="Greeting message")

class MessageSchema(BaseModel):
    message: str = Field(..., description="Message content")


def test_prompt_manager_creation():
    """Test creating PromptManager and basic initialization"""
    manager = PromptManager()
    
    assert isinstance(manager, PromptManager)
    assert hasattr(manager, 'prompt_registry')
    assert len(manager.prompt_registry) > 0  # Should have default prompts


def test_prompt_registration():
    """Test registering and retrieving prompts"""
    manager = PromptManager()
    initial_count = len(manager.prompt_registry)
    
    # Register a simple prompt
    manager.register_prompt(
        "test_prompt",
        template="Hello {{ name }}!",
        output_schema=GreetingSchema
    )
    
    new_count = len(manager.prompt_registry)
    
    assert new_count == initial_count + 1
    assert "test_prompt" in manager.prompt_registry
    
    # Test retrieval
    retrieved = manager.get_prompt("test_prompt")
    
    assert retrieved is not None
    assert "template" in retrieved


def test_template_rendering():
    """Test basic template rendering with variables"""
    manager = PromptManager()
    
    # Register a template with variables
    manager.register_prompt(
        "greeting_template",
        template="Hello {{ name }}, welcome to {{ place }}!",
        output_schema=MessageSchema
    )
    
    # Test rendering
    rendered = manager.render_template("greeting_template", {
        "name": "Alice",
        "place": "AI Brain"
    })
    
    assert "Alice" in rendered
    assert "AI Brain" in rendered
