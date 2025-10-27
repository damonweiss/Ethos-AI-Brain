#!/usr/bin/env python3
"""
PromptRoutingManager - Universal Router for Prompts → Engines

- Works with LLMEngine, VisionEngine, EmbeddingsEngine.
- Accepts either:
  - Engine-agnostic params (input_data, image_url, schema, etc.)
  - Explicit engine/model preferences
- Infers best engine/model from prompt metadata + inputs.
- Executes, validates, and returns normalized results.
"""

from datetime import datetime
from typing import Dict, Any, Optional

from jinja2 import Template

# Import actual classes from the correct locations
try:
    from ethos_ai_brain.core.prompt_manager.prompt_manager import PromptManager
except ImportError:
    PromptManager = None

try:
    from ethos_ai_brain.core.schema_manager.schema_manager import SchemaRegistry, SchemaValidator, ValidationMode
except ImportError:
    SchemaRegistry = None
    SchemaValidator = None
    ValidationMode = None

try:
    from ethos_ai_brain.reasoning.inference_model_manager.inference_model_manager import InferenceModelManager
except ImportError:
    InferenceModelManager = None

try:
    from ethos_ai_brain.reasoning.inference_engines.llm_engine import LLMEngine
    from ethos_ai_brain.reasoning.inference_engines.vision_engine import VisionEngine
    from ethos_ai_brain.reasoning.inference_engines.embeddings_engine import EmbeddingsEngine
except ImportError:
    LLMEngine = None
    VisionEngine = None
    EmbeddingsEngine = None


class PromptRoutingManager:
    """Routes prompts to the best engine/model, engine-agnostic."""

    def __init__(self):
        self.prompt_manager = PromptManager() if PromptManager else None
        self.model_manager = InferenceModelManager(auto_discover=True) if InferenceModelManager else None

        self.schema_registry = SchemaRegistry() if SchemaRegistry else None
        self.schema_validator = SchemaValidator(self.schema_registry, default_mode=ValidationMode.COERCE) if SchemaValidator and self.schema_registry else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fail(self, message: str, **kwargs) -> Dict[str, Any]:
        return {"success": False, "error": message, **kwargs}

    def _get_engine(self, model: str, engine_type: str = None):
        """
        Pick correct engine class for a model.
        """
        meta = self.model_manager.available_models.get(model, {})
        etype = engine_type or meta.get("engine_type", "llm")

        if etype == "vision":
            return VisionEngine(model)
        elif etype == "embeddings":
            return EmbeddingsEngine(model)
        return LLMEngine(model)  # default

    def _infer_engine_type(self, variables: Dict[str, Any], schema: Dict[str, Any]) -> str:
        """
        Infer engine type from inputs and schema.
        """
        if "image_url" in variables:
            return "vision"
        if schema and any("embedding" in str(v).lower() for v in schema.values()):
            return "embeddings"
        return "llm"

    def _select_model(self, engine_type: str, model_preferences: Dict[str, Any]) -> Optional[str]:
        """
        Select a model compatible with the requested engine_type.
        """
        if not self.model_manager:
            return None
            
        # Use available models from the model manager
        available = list(self.model_manager.available_models.keys())

        # Explicit model takes precedence
        if "model" in model_preferences:
            return model_preferences["model"]

        # Filter by engine_type
        valid = [
            m for m in available
            if self.model_manager.available_models.get(m, {}).get("engine_type") == engine_type
        ]
        if not valid:
            return None

        # Prefer cheapest if requested
        if model_preferences.get("prefer_low_cost", False):
            costs = []
            for m in valid:
                ci = self.model_manager.get_model_cost_info(m)
                if ci:
                    cost = ci.get("input_cost_per_token", 0) + ci.get("output_cost_per_token", 0)
                    costs.append((m, cost))
            if costs:
                return sorted(costs, key=lambda x: x[1])[0][0]

        return valid[0]  # fallback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_prompt(
        self,
        prompt_name: str,
        variables: Dict[str, Any],
        model_preferences: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a registered prompt with the right engine.
        """
        prompt_data = self.prompt_manager.get_prompt(prompt_name)
        if not prompt_data:
            return self._fail(f"Prompt '{prompt_name}' not found")

        schema = prompt_data.get("output_schema", {})
        
        # Handle both dict and Pydantic model schemas
        if hasattr(schema, 'model_json_schema'):
            # It's a Pydantic model, convert to dict
            schema_dict = schema.model_json_schema()
        elif isinstance(schema, dict):
            # It's already a dict
            schema_dict = schema
        else:
            return self._fail("Invalid schema format", prompt=prompt_name)

        # Register schema with rebuild (only if it's a simple dict schema)
        if self.schema_registry and isinstance(schema, dict):
            self.schema_registry.create_dynamic(prompt_name, schema_dict)
        # For Pydantic models, we can use them directly without re-creating

        # Infer engine type
        engine_type = self._infer_engine_type(variables, schema_dict)

        # Pick model
        model_prefs = model_preferences or {}
        selected_model = self._select_model(engine_type, model_prefs)
        if not selected_model:
            return self._fail("No suitable model found", prompt=prompt_name)

        # Render template
        try:
            rendered_prompt = Template(prompt_data["template"]).render(**variables)
        except Exception as e:
            return self._fail(f"Failed to render template: {e}")

        # Dispatch to engine
        engine = self._get_engine(selected_model, engine_type)
        run_kwargs = dict(schema=schema, temperature=model_prefs.get("temperature", 0.1))
        if engine_type == "vision":
            run_kwargs["image_url"] = variables.get("image_url")

        result = engine.run(input_data=rendered_prompt, **run_kwargs)

        if not result.get("success"):
            return self._fail("Engine execution failed", error=result.get("error"))

        # Validate result
        validation = self.schema_validator.validate(
            result.get("result", {}),
            schema_name=prompt_name
        )
        if not validation["success"]:
            return self._fail("Schema validation failed", errors=validation["errors"])

        return {
            "success": True,
            "result": validation["data"],
            "engine_type": engine_type,
            "model": selected_model,
            "cost": result.get("cost", 0.0),
            "usage": result.get("usage", {}),
            "metadata": {
                "prompt_name": prompt_name,
                "engine_type": engine_type,
                "model": selected_model,
                "timestamp": datetime.now().isoformat(),
            },
        }
