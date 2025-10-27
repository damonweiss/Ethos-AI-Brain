#!/usr/bin/env python3
import pytest
import json
import sys
import warnings
from pathlib import Path
from pydantic import BaseModel

# Add the parent directory to sys.path to import prompt_manager
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompt_manager import PromptManager


# ------------------ Fixtures ------------------

@pytest.fixture
def tmp_templates_dir(tmp_path: Path):
    """Temporary templates directory with one sample file."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    template_file = templates_dir / "hello.jinja"
    template_file.write_text("Hello {{ name }}! {# SCHEMA: {\"greeting\": \"string\"} #}")
    return templates_dir


@pytest.fixture
def manager(tmp_templates_dir: Path):
    """PromptManager instance with tmp templates dir."""
    return PromptManager(templates_dir=tmp_templates_dir)


# ------------------ Tests ------------------

def test_register_and_get_prompt_json(manager):
    """Register with JSON schema -> should warn and build Pydantic model."""
    with warnings.catch_warnings(record=True) as w:
        manager.register_prompt("test", "Hello {{ who }}", {"result": "string"})
        assert any("Schema field 'result'" in str(x.message) for x in w)

    prompt = manager.get_prompt("test")
    assert prompt is not None
    assert "pydantic_model" in prompt
    assert issubclass(prompt["pydantic_model"], BaseModel)
    assert prompt["template"] == "Hello {{ who }}"


def test_register_and_get_prompt_pydantic(manager):
    """Register with explicit Pydantic model."""
    class OutModel(BaseModel):
        message: str

    manager.register_prompt("pmodel", "Hi {{ person }}", OutModel)
    prompt = manager.get_prompt("pmodel")
    assert issubclass(prompt["pydantic_model"], BaseModel)
    assert prompt["output_schema"] is OutModel


def test_register_template_file(manager, tmp_templates_dir):
    manager.register_template_file("hello", "hello.jinja")
    prompt = manager.get_prompt("hello")
    assert prompt is not None
    assert "Hello {{ name }}" in prompt["template"]
    assert "greeting" in prompt["output_schema"]


def test_auto_load_templates(manager):
    results = manager.auto_load_templates()
    assert "hello" in results["loaded"] or "hello" in manager.list_prompts()


def test_render_template(manager):
    manager.register_prompt("test", "Hello {{ who }}", {"result": "string"})
    rendered = manager.render_template("test", {"who": "World"})
    assert rendered == "Hello World"
    stats = manager.get_usage_stats()
    assert stats["usage_by_prompt"]["test"] == 1


def test_validate_template_success(manager):
    manager.register_prompt("valid", "Hi {{ person }}", {"result": "string"})
    result = manager.validate_template("valid", {"person": "Alice"})
    assert result["valid"] is True
    assert "person" in result["undeclared_variables"]
    stats = manager.get_usage_stats()
    assert stats["validation_by_prompt"]["valid"] == 1


def test_validate_template_failure(manager):
    manager.register_prompt("fail", "Hi {{ missing_var }}", {"result": "string"})
    result = manager.validate_template("fail", {})
    assert result["valid"] is True  # still valid Jinja, but undeclared
    assert "missing_var" in result["undeclared_variables"]
    stats = manager.get_usage_stats()
    assert stats["validation_by_prompt"]["fail"] == 1


def test_validate_output_json(manager):
    """Validate dict output against JSON schema -> coerced to Pydantic model."""
    manager.register_prompt("out", "dummy", {"field": "string"})
    result = manager.validate_output("out", {"field": "abc"})
    assert result["valid"] is True
    assert result["data"]["field"] == "abc"


def test_validate_output_failure(manager):
    manager.register_prompt("out", "dummy", {"field": "int"})
    # Wrong type should raise validation error
    result = manager.validate_output("out", {"field": "not-an-int"})
    assert result["valid"] is False
    assert "error" in result


def test_reload_template(manager, tmp_templates_dir):
    manager.register_template_file("hello", "hello.jinja")
    filepath = tmp_templates_dir / "hello.jinja"
    filepath.write_text("Hi {{ name }}!")
    assert manager.reload_template("hello") is True
    updated = manager.get_prompt("hello")
    assert "Hi {{ name }}" in updated["template"]


def test_export_registry(manager):
    manager.register_prompt("sample", "Test", {"result": "string"})
    data = manager.export_registry()
    assert "prompts" in data
    assert "sample" in data["prompts"]
    # round-trip JSON
    json.dumps(data)


def test_remove_and_clear(manager):
    manager.register_prompt("temp", "x", {"result": "string"})
    assert manager.remove_prompt("temp") is True
    assert manager.get_prompt("temp") is None
    manager.register_prompt("another", "y", {"result": "string"})
    manager.clear_registry()
    assert manager.list_prompts() == []


def test_usage_stats(manager):
    manager.register_prompt("stats", "Hello", {"result": "string"})
    initial_stats = manager.get_usage_stats()
    assert initial_stats["total_usage"] == 0
    assert initial_stats["validation_by_prompt"]["stats"] == 0

    manager.render_template("stats")
    stats = manager.get_usage_stats()
    assert stats["total_usage"] == 1
    assert stats["most_used"] == "stats"
    assert stats["validation_by_prompt"]["stats"] == 0


def test_invalid_schema_file(tmp_path: Path):
    """Invalid JSON in SCHEMA comment should raise."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    bad_file = templates_dir / "bad.jinja"
    bad_file.write_text("Hello {{ who }} {# SCHEMA: {bad json} #}")

    mgr = PromptManager(templates_dir=templates_dir)
    with pytest.raises(ValueError):
        mgr.register_template_file("bad", "bad.jinja")


# ------------------ CLI Entry ------------------

if __name__ == "__main__":
    pytest.main([__file__])
