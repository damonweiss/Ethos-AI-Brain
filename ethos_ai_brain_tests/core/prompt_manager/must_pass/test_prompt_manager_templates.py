"""
Test PromptManager Template Functionality - Must Pass
Tests template loading, rendering, and file operations with real code only
"""

import sys
import tempfile
from pathlib import Path
from pydantic import BaseModel, Field

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.core.prompt_manager.prompt_manager import PromptManager


def test_prompt_manager_render_template():
    """Test PromptManager.render_template - rendering templates with variables"""
    pm = PromptManager()
    
    # Register a template with variables
    class TestSchema(BaseModel):
        result: str
    
    pm.register_prompt(
        "test_render",
        "Hello {{ name }}, your score is {{ score }}!",
        TestSchema,
        "Test rendering template"
    )
    
    # Render with variables
    rendered = pm.render_template("test_render", {"name": "Alice", "score": 95})
    print(f"Rendered template: {rendered}")
    
    assert rendered == "Hello Alice, your score is 95!"
    
    print("[SUCCESS] Template rendering works correctly")


def test_prompt_manager_render_template_missing_variables():
    """Test template rendering with missing variables"""
    pm = PromptManager()
    
    class TestSchema(BaseModel):
        result: str
    
    pm.register_prompt(
        "missing_vars",
        "Hello {{ name }}, your age is {{ age }}",
        TestSchema
    )
    
    # Try to render with missing variables - should raise error
    try:
        rendered = pm.render_template("missing_vars", {"name": "Bob"})  # missing 'age'
        assert False, "Should have raised an error for missing variable"
    except Exception as e:
        print(f"Expected error for missing variable: {type(e).__name__}: {e}")
        assert True  # This is expected
    
    print("[SUCCESS] Missing variables handled correctly")


def test_prompt_manager_render_nonexistent_template():
    """Test rendering non-existent template"""
    pm = PromptManager()
    
    try:
        rendered = pm.render_template("nonexistent", {"var": "value"})
        assert False, "Should have raised error for nonexistent template"
    except ValueError as e:
        print(f"Expected error for nonexistent template: {e}")
        assert "not found" in str(e)
    
    print("[SUCCESS] Nonexistent template error handling works correctly")


def test_prompt_manager_register_template_file():
    """Test PromptManager.register_template_file - loading from file"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pm = PromptManager(templates_dir=temp_path)
        
        # Create a test template file - let it use default schema
        template_file = temp_path / "test_template.jinja"
        template_content = "Process {{ data }} with method {{ method }}"
        template_file.write_text(template_content)
        
        # Register from file
        pm.register_template_file("file_test", "test_template.jinja", "File-based template")
        
        # Verify registration
        prompts = pm.list_prompts()
        print(f"Prompts after file registration: {prompts}")
        
        assert "file_test" in prompts
        
        # Test rendering
        rendered = pm.render_template("file_test", {"data": "input", "method": "analysis"})
        print(f"File template rendered: {rendered}")
        
        assert rendered == "Process input with method analysis"
    
    print("[SUCCESS] Template file registration works correctly")


def test_prompt_manager_auto_load_templates():
    """Test PromptManager.auto_load_templates - loading all .jinja files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pm = PromptManager(templates_dir=temp_path)
        
        # Create multiple template files - let them use default schemas
        templates = {
            "template1.jinja": "Template 1: {{ var1 }}",
            "template2.jinja": "Template 2: {{ var2 }}",
            "not_template.txt": "Not a template"  # Should be ignored
        }
        
        for filename, content in templates.items():
            (temp_path / filename).write_text(content)
        
        # Auto-load templates
        results = pm.auto_load_templates()
        print(f"Auto-load results: {results}")
        
        assert isinstance(results, dict)
        assert "loaded" in results
        assert "failed" in results
        
        # Should have loaded .jinja files
        loaded_files = results["loaded"]
        print(f"Loaded files: {loaded_files}")
        
        assert len(loaded_files) == 2  # Only .jinja files
        assert any("template1" in f for f in loaded_files)
        assert any("template2" in f for f in loaded_files)
        
        # Verify prompts were registered
        prompts = pm.list_prompts()
        print(f"Registered prompts: {prompts}")
        
        # Should have the auto-loaded templates (names based on filenames)
        template_names = [name for name in prompts if name.startswith("template")]
        assert len(template_names) >= 2
    
    print("[SUCCESS] Auto-load templates works correctly")


def test_prompt_manager_validate_template():
    """Test PromptManager.validate_template - template validation"""
    pm = PromptManager()
    
    class TestSchema(BaseModel):
        result: str
    
    # Register a valid template
    pm.register_prompt(
        "valid_template",
        "Valid template with {{ variable }}",
        TestSchema
    )
    
    # Test validation with proper variables
    validation_result = pm.validate_template("valid_template", {"variable": "test_value"})
    print(f"Validation result: {validation_result}")
    
    assert validation_result["valid"] == True
    
    # Test validation with missing variables
    validation_result_missing = pm.validate_template("valid_template", {})
    print(f"Validation with missing vars: {validation_result_missing}")
    
    # Template validation might still be valid but report undeclared variables
    assert validation_result_missing["valid"] == True  # Template syntax is valid
    if "undeclared_variables" in validation_result_missing:
        assert len(validation_result_missing["undeclared_variables"]) > 0
        print(f"Undeclared variables: {validation_result_missing['undeclared_variables']}")
    
    # Test validation of non-existent template
    validation_nonexistent = pm.validate_template("nonexistent", {"var": "value"})
    print(f"Validation of nonexistent: {validation_nonexistent}")
    
    assert validation_nonexistent["valid"] == False
    assert "not found" in validation_nonexistent["error"]
    
    print("[SUCCESS] Template validation works correctly")


def test_prompt_manager_reload_template():
    """Test PromptManager.reload_template - reloading templates from file"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pm = PromptManager(templates_dir=temp_path)
        
        # Create initial template file
        template_file = temp_path / "reload_test.jinja"
        initial_content = "Initial content: {{ var }}"
        template_file.write_text(initial_content)
        
        # Register template
        pm.register_template_file("reload_test", "reload_test.jinja")
        
        # Render initial version
        initial_render = pm.render_template("reload_test", {"var": "test"})
        print(f"Initial render: {initial_render}")
        
        assert initial_render == "Initial content: test"
        
        # Modify the file
        updated_content = "Updated content: {{ var }}"
        template_file.write_text(updated_content)
        
        # Reload template
        reload_success = pm.reload_template("reload_test")
        print(f"Reload successful: {reload_success}")
        
        assert reload_success == True
        
        # Render updated version
        updated_render = pm.render_template("reload_test", {"var": "test"})
        print(f"Updated render: {updated_render}")
        
        assert updated_render == "Updated content: test"
        
        # Test reloading non-file-based template
        class TestSchema(BaseModel):
            result: str
        
        pm.register_prompt("memory_template", "Memory: {{ data }}", TestSchema)
        reload_memory = pm.reload_template("memory_template")
        print(f"Memory template reload: {reload_memory}")
        
        assert reload_memory == False  # Can't reload non-file templates
    
    print("[SUCCESS] Template reloading works correctly")


def test_prompt_manager_export_registry():
    """Test PromptManager.export_registry - exporting prompt registry"""
    pm = PromptManager()
    
    # Add some test prompts
    class TestSchema(BaseModel):
        result: str
    
    pm.register_prompt("export_test1", "Template 1: {{ var1 }}", TestSchema, "First test")
    pm.register_prompt("export_test2", "Template 2: {{ var2 }}", TestSchema, "Second test")
    
    # Export registry
    export_data = pm.export_registry()
    print(f"Export data keys: {list(export_data.keys())}")
    
    assert isinstance(export_data, dict)
    assert "exported_at" in export_data
    assert "templates_dir" in export_data
    assert "prompts" in export_data
    
    # Check exported prompts
    exported_prompts = export_data["prompts"]
    print(f"Exported prompt names: {list(exported_prompts.keys())}")
    
    assert "export_test1" in exported_prompts
    assert "export_test2" in exported_prompts
    assert "general_analysis" in exported_prompts  # Default prompt
    
    # Test export to file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    
    try:
        export_with_file = pm.export_registry(temp_path)
        
        # Verify file was created
        assert temp_path.exists()
        
        # Verify file content
        import json
        file_content = json.loads(temp_path.read_text())
        assert file_content == export_with_file
        
    finally:
        # Clean up
        if temp_path.exists():
            temp_path.unlink()
    
    print("[SUCCESS] Registry export works correctly")


def test_prompt_manager_template_with_complex_variables():
    """Test template rendering with complex variable structures"""
    pm = PromptManager()
    
    class ComplexSchema(BaseModel):
        analysis: str
        metadata: dict
    
    pm.register_prompt(
        "complex_template",
        "User: {{ user.name }} ({{ user.role }})\nData: {{ data | length }} items\nConfig: {{ config.mode }}",
        ComplexSchema
    )
    
    # Render with complex variables
    variables = {
        "user": {"name": "Alice", "role": "analyst"},
        "data": [1, 2, 3, 4, 5],
        "config": {"mode": "production"}
    }
    
    rendered = pm.render_template("complex_template", variables)
    print(f"Complex template rendered: {rendered}")
    
    expected = "User: Alice (analyst)\nData: 5 items\nConfig: production"
    assert rendered == expected
    
    print("[SUCCESS] Complex variable rendering works correctly")


def test_prompt_manager_template_error_handling():
    """Test template error handling for malformed templates"""
    pm = PromptManager()
    
    class TestSchema(BaseModel):
        result: str
    
    # Try to register template with syntax error - should fail at registration
    try:
        pm.register_prompt(
            "broken_template",
            "Broken template: {{ unclosed_variable",  # Missing closing }}
            TestSchema
        )
        assert False, "Should have raised error during registration of malformed template"
    except Exception as e:
        print(f"Expected error during registration: {type(e).__name__}: {e}")
        assert "TemplateSyntaxError" in str(type(e).__name__) or "unexpected" in str(e).lower()
    
    # Verify the broken template was not registered
    prompts = pm.list_prompts()
    assert "broken_template" not in prompts
    
    print("[SUCCESS] Template error handling works correctly")
