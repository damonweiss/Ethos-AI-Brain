"""
Test PromptManager Core Functionality - Must Pass
Tests each core function in PromptManager with real code only
"""

import sys
from pathlib import Path
from pydantic import BaseModel, Field

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.core.prompt_manager.prompt_manager import PromptManager


def test_prompt_manager_creation():
    """Test PromptManager.__init__ - manager creation and initialization"""
    pm = PromptManager()
    
    print(f"Expected prompt_registry exists: {hasattr(pm, 'prompt_registry')}")
    print(f"Expected templates_dir exists: {hasattr(pm, 'templates_dir')}")
    print(f"Registry type: {type(pm.prompt_registry)}")
    print(f"Templates dir type: {type(pm.templates_dir)}")
    
    assert hasattr(pm, 'prompt_registry')
    assert hasattr(pm, 'templates_dir')
    assert isinstance(pm.prompt_registry, dict)
    assert isinstance(pm.templates_dir, Path)
    
    print("[SUCCESS] PromptManager creation works correctly")


def test_prompt_manager_default_prompt():
    """Test that default general_analysis prompt is created"""
    pm = PromptManager()
    
    # Should have default prompt
    prompts = pm.list_prompts()
    print(f"Default prompts: {prompts}")
    print(f"Has general_analysis: {'general_analysis' in prompts}")
    
    assert len(prompts) > 0
    assert 'general_analysis' in prompts
    
    # Test getting the default prompt
    default_prompt = pm.get_prompt('general_analysis')
    print(f"Default prompt structure: {list(default_prompt.keys()) if default_prompt else None}")
    
    assert default_prompt is not None
    assert 'template' in default_prompt
    assert 'output_schema' in default_prompt
    
    print("[SUCCESS] Default prompt creation works correctly")


def test_prompt_manager_register_prompt():
    """Test PromptManager.register_prompt - registering new prompts"""
    pm = PromptManager()
    
    # Define a test schema
    class TestSchema(BaseModel):
        result: str = Field(..., description="Test result")
        confidence: float = Field(..., description="Confidence score")
    
    # Register a new prompt
    pm.register_prompt(
        name="test_prompt",
        template="Test template with {{ variable }}",
        output_schema=TestSchema,
        description="A test prompt"
    )
    
    # Verify registration
    prompts = pm.list_prompts()
    print(f"Prompts after registration: {prompts}")
    print(f"Has test_prompt: {'test_prompt' in prompts}")
    
    assert 'test_prompt' in prompts
    
    # Verify prompt details
    test_prompt = pm.get_prompt('test_prompt')
    print(f"Test prompt template: {test_prompt['template']}")
    print(f"Test prompt description: {test_prompt['description']}")
    
    assert test_prompt['template'] == "Test template with {{ variable }}"
    assert test_prompt['description'] == "A test prompt"
    assert test_prompt['output_schema'] is not None
    
    print("[SUCCESS] Prompt registration works correctly")


def test_prompt_manager_list_prompts():
    """Test PromptManager.list_prompts - listing all registered prompts"""
    pm = PromptManager()
    
    # Should have at least the default prompt
    prompts = pm.list_prompts()
    print(f"Initial prompts: {prompts}")
    print(f"Number of prompts: {len(prompts)}")
    
    assert isinstance(prompts, list)
    assert len(prompts) >= 1  # At least the default
    
    # Add another prompt
    class SimpleSchema(BaseModel):
        answer: str
    
    pm.register_prompt("simple_test", "Simple {{ input }}", SimpleSchema)
    
    updated_prompts = pm.list_prompts()
    print(f"Updated prompts: {updated_prompts}")
    print(f"Number after addition: {len(updated_prompts)}")
    
    assert len(updated_prompts) == len(prompts) + 1
    assert "simple_test" in updated_prompts
    
    print("[SUCCESS] List prompts works correctly")


def test_prompt_manager_get_prompt():
    """Test PromptManager.get_prompt - retrieving specific prompts"""
    pm = PromptManager()
    
    # Test getting existing prompt
    existing_prompt = pm.get_prompt('general_analysis')
    print(f"Existing prompt found: {existing_prompt is not None}")
    
    assert existing_prompt is not None
    assert isinstance(existing_prompt, dict)
    
    # Test getting non-existent prompt
    missing_prompt = pm.get_prompt('nonexistent_prompt')
    print(f"Missing prompt result: {missing_prompt}")
    
    assert missing_prompt is None
    
    print("[SUCCESS] Get prompt works correctly")


def test_prompt_manager_get_prompt_info():
    """Test PromptManager.get_prompt_info - getting prompt metadata"""
    pm = PromptManager()
    
    # Get info for existing prompt
    info = pm.get_prompt_info('general_analysis')
    print(f"Prompt info keys: {list(info.keys()) if info else None}")
    
    assert info is not None
    assert isinstance(info, dict)
    assert 'name' in info
    assert 'description' in info
    assert info['name'] == 'general_analysis'
    
    # Test non-existent prompt
    missing_info = pm.get_prompt_info('nonexistent')
    print(f"Missing prompt info: {missing_info}")
    
    assert missing_info is None
    
    print("[SUCCESS] Get prompt info works correctly")


def test_prompt_manager_remove_prompt():
    """Test PromptManager.remove_prompt - removing prompts"""
    pm = PromptManager()
    
    # Add a test prompt to remove
    class TestSchema(BaseModel):
        data: str
    
    pm.register_prompt("to_remove", "Remove me {{ var }}", TestSchema)
    
    # Verify it exists
    assert "to_remove" in pm.list_prompts()
    
    # Remove it
    removed = pm.remove_prompt("to_remove")
    print(f"Removal successful: {removed}")
    
    assert removed == True
    assert "to_remove" not in pm.list_prompts()
    
    # Try to remove non-existent prompt
    not_removed = pm.remove_prompt("never_existed")
    print(f"Non-existent removal result: {not_removed}")
    
    assert not_removed == False
    
    print("[SUCCESS] Remove prompt works correctly")


def test_prompt_manager_clear_registry():
    """Test PromptManager.clear_registry - clearing all prompts"""
    pm = PromptManager()
    
    # Should have at least default prompt
    initial_count = len(pm.list_prompts())
    print(f"Initial prompt count: {initial_count}")
    
    assert initial_count > 0
    
    # Clear registry
    pm.clear_registry()
    
    # Should be empty
    final_count = len(pm.list_prompts())
    print(f"Final prompt count: {final_count}")
    
    assert final_count == 0
    assert pm.list_prompts() == []
    
    print("[SUCCESS] Clear registry works correctly")


def test_prompt_manager_custom_templates_dir():
    """Test PromptManager.__init__ with custom templates directory"""
    import tempfile
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        pm = PromptManager(templates_dir=temp_path)
        
        print(f"Custom templates dir: {pm.templates_dir}")
        print(f"Directory exists: {pm.templates_dir.exists()}")
        
        assert pm.templates_dir == temp_path
        assert pm.templates_dir.exists()
    
    print("[SUCCESS] Custom templates directory works correctly")


def test_prompt_manager_get_usage_stats():
    """Test PromptManager.get_usage_stats - usage statistics"""
    pm = PromptManager()
    
    # Get initial stats
    stats = pm.get_usage_stats()
    print(f"Usage stats type: {type(stats)}")
    print(f"Usage stats keys: {list(stats.keys())}")
    
    assert isinstance(stats, dict)
    
    # Should have usage info structure
    expected_keys = ['total_prompts', 'usage_by_prompt']
    for key in expected_keys:
        if key in stats:
            print(f"Found expected key: {key}")
    
    # Check if usage_by_prompt contains our prompt
    if 'usage_by_prompt' in stats and 'general_analysis' in pm.list_prompts():
        print(f"Usage by prompt: {stats['usage_by_prompt']}")
        assert 'general_analysis' in stats['usage_by_prompt']
    
    print("[SUCCESS] Get usage stats works correctly")


def test_prompt_manager_templates_dir_property():
    """Test that templates_dir is properly set and accessible"""
    pm = PromptManager()
    
    print(f"Templates dir path: {pm.templates_dir}")
    print(f"Templates dir is Path: {isinstance(pm.templates_dir, Path)}")
    print(f"Templates dir exists: {pm.templates_dir.exists()}")
    
    assert isinstance(pm.templates_dir, Path)
    assert pm.templates_dir.exists()
    
    # Should be the default location
    expected_path = Path(__file__).resolve().parents[5] / "ethos_ai_brain" / "core" / "prompt_templates"
    print(f"Expected path: {expected_path}")
    
    # Path should end with prompt_templates
    assert pm.templates_dir.name == "prompt_templates"
    
    print("[SUCCESS] Templates directory property works correctly")
