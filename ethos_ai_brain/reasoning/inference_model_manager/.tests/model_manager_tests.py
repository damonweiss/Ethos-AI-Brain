import pytest
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(project_root))

from ethos_mcp.mcp_managers.inference_model_manager import InferenceModelManager
from ethos_mcp.mcp_managers.inference_engines_manager import InferenceEnginesManager


@pytest.fixture(scope="module")
def engines_manager():
    return InferenceEnginesManager()


def test_registry_has_builtin_engines(engines_manager):
    types = engines_manager.types()
    assert "llm" in types


def test_engine_info(engines_manager):
    info = engines_manager.info("llm")
    assert isinstance(info, dict)
    assert info["engine_type"] == "llm"
    assert "module" in info
    assert "class" in info


def test_create_llm_engine_without_model(engines_manager):
    with pytest.raises(ValueError):
        engines_manager.create("llm", instance_key="bad-llm")  # missing required model


def test_create_llm_engine_with_model(engines_manager):
    engine = engines_manager.create("llm", instance_key="gpt4-test", model="gpt-4o")
    assert engine is not None
    assert engine.model == "gpt-4o"

    # Ensure get_or_create returns same instance
    same_engine = engines_manager.get_or_create("llm", "gpt4-test", model="gpt-4o")
    assert same_engine is engine


@pytest.mark.parametrize("text,expected", [
    ("summarize this passage", "llm"),
    ("analyze and explain reasoning", "llm"),
])
def test_select_engine_type(engines_manager, text, expected):
    selected = engines_manager.select(text)
    assert selected == expected


def test_engine_stats_and_cache_clear(engines_manager):
    stats_before = engines_manager.stats()
    assert "registered" in stats_before
    assert stats_before["registered"] >= 1

    engines_manager.clear()
    stats_after = engines_manager.stats()
    assert stats_after["loaded"] == []
    assert stats_after["instances"] == {}


def main():
    """Run pytest programmatically when this script is executed directly."""
    sys.exit(pytest.main([__file__]))


if __name__ == "__main__":
    main()
