#!/usr/bin/env python3
"""
PromptManager - Prompt Registry and Template Management (Pydantic-Aware)

- Manages prompt templates and schemas.
- Supports Jinja2 templating with auto-variable detection.
- Prefers Pydantic v2 models for schema validation (JSON-style schemas allowed).
"""

import json
import re
import warnings
from typing import Dict, Any, List, Optional, Type
from datetime import datetime
from pathlib import Path

try:
    from jinja2 import Template, Environment, meta, TemplateError
except ImportError as e:
    raise ImportError("Jinja2 is required for PromptManager") from e

try:
    from pydantic import BaseModel, create_model
except ImportError:
    raise ImportError("Pydantic is required for PromptManager schema handling")


class PromptManager:
    """Registry and manager for prompt templates with Pydantic-aware schemas."""

    def __init__(self, templates_dir: Optional[Path] = None):
        self.prompt_registry: Dict[str, Dict[str, Any]] = {}
        self.templates_dir = templates_dir or self._get_default_templates_dir()
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        # Fallback example prompt
        if not self.prompt_registry:
            # Create a proper static Pydantic schema
            from pydantic import Field
            
            class DefaultAnalysisSchema(BaseModel):
                analysis: str = Field(..., description="Analysis result")
            
            self.register_prompt(
                "general_analysis",
                template="Analyze the following text: {{ text }}",
                output_schema=DefaultAnalysisSchema,
            )

    def _get_default_templates_dir(self) -> Path:
        return Path(__file__).parent.parent.parent / "prompt_templates"

    # ------------------ Schema Conversion ------------------

    def _convert_schema_to_pydantic(self, name: str, schema: Any) -> Type[BaseModel]:
        """Convert schema (dict or Pydantic model) into a usable Pydantic model."""
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema

        if not isinstance(schema, dict):
            raise TypeError("Schema must be dict or subclass of BaseModel")

        type_map = {
            "string": str,
            "int": int,
            "integer": int,
            "float": float,
            "bool": bool,
            "boolean": bool,
        }

        fields = {}
        for field, declared in schema.items():
            if isinstance(declared, str):
                warnings.warn(
                    f"Schema field '{field}' uses JSON-style type '{declared}', "
                    "mapping to Python type."
                )
                py_type = type_map.get(declared.lower(), str)
                fields[field] = (py_type, ...)
            else:
                fields[field] = (declared, ...)

        model = create_model(f"{name}_Schema", **fields)
        model.model_rebuild(force=True)   # ✅ critical for Pydantic v2
        return model


    # ------------------ Prompt Registration ------------------

    def register_prompt(
        self,
        name: str,
        template: str,
        output_schema: Any,
        description: str = "",
        template_file: Optional[str] = None,
    ) -> None:
        """Register a template with schema validation support."""
        env = Environment()
        parsed = env.parse(template)
        variables = list(meta.find_undeclared_variables(parsed))

        schema_model = self._convert_schema_to_pydantic(name, output_schema)

        self.prompt_registry[name] = {
            "template": template,
            "output_schema": output_schema,
            "pydantic_model": schema_model,
            "description": description,
            "variables": variables,
            "registered_at": datetime.now().isoformat(),
            "usage_count": 0,
            "validation_count": 0,
            "template_file": template_file,
        }

    def _register_from_file(self, name: str, filepath: Path, description: str = "") -> None:
        if not filepath.exists():
            raise FileNotFoundError(f"Template file not found: {filepath}")

        template_content = filepath.read_text(encoding="utf-8")
        output_schema = self._extract_schema_from_template(template_content)
        self.register_prompt(
            name,
            template_content,
            output_schema,
            description or f"Loaded from {filepath.name}",
            filepath.name,
        )

    def register_template_file(self, name: str, filename: str, description: str = "") -> None:
        self._register_from_file(name, self.templates_dir / filename, description)

    def auto_load_templates(self) -> Dict[str, Any]:
        """Load all .jinja templates in directory."""
        results = {"loaded": [], "failed": [], "errors": []}
        for f in self.templates_dir.glob("*.jinja"):
            try:
                self._register_from_file(f.stem, f)
                results["loaded"].append(f.stem)
            except Exception as e:
                results["failed"].append(f.name)
                results["errors"].append(str(e))
        return results

    # ------------------ Retrieval ------------------

    def get_prompt(self, name: str) -> Optional[Dict[str, Any]]:
        return self.prompt_registry.get(name)

    def list_prompts(self) -> List[str]:
        return list(self.prompt_registry.keys())

    def get_prompt_info(self, name: str) -> Optional[Dict[str, Any]]:
        if name not in self.prompt_registry:
            return None
        return self._serialize_prompt_metadata(name, self.prompt_registry[name])

    def _serialize_prompt_metadata(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": name,
            "description": data["description"],
            "variables": data["variables"],
            "registered_at": data["registered_at"],
            "usage_count": data["usage_count"],
            "validation_count": data["validation_count"],
            "template_file": data.get("template_file"),
            "has_schema": bool(data["output_schema"]),
            "schema_type": (
                "pydantic"
                if isinstance(data["pydantic_model"], type) and issubclass(data["pydantic_model"], BaseModel)
                else "json"
            ),
        }

    # ------------------ Schema / Reload ------------------

    def _extract_schema_from_template(self, content: str) -> Dict[str, Any]:
        """Extract schema from `{# SCHEMA: {...} #}` block."""
        match = re.search(r"\{\#\s*SCHEMA:\s*(\{.*?\})\s*\#\}", content, re.DOTALL)
        if not match:
            return {"result": "string"}
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid schema JSON in template: {e}")

    def reload_template(self, name: str) -> bool:
        prompt = self.prompt_registry.get(name)
        if not prompt or not prompt.get("template_file"):
            return False

        filepath = self.templates_dir / prompt["template_file"]
        if not filepath.exists():
            return False

        content = filepath.read_text(encoding="utf-8")
        output_schema = self._extract_schema_from_template(content)
        schema_model = self._convert_schema_to_pydantic(name, output_schema)

        env = Environment()
        parsed = env.parse(content)
        variables = list(meta.find_undeclared_variables(parsed))

        prompt.update({
            "template": content,
            "output_schema": output_schema,
            "pydantic_model": schema_model,
            "variables": variables,
        })
        return True

    # ------------------ Export / Stats ------------------

    def export_registry(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        export = {
            "exported_at": datetime.now().isoformat(),
            "templates_dir": str(self.templates_dir),
            "total_prompts": len(self.prompt_registry),
            "prompts": {
                n: self._serialize_prompt_metadata(n, d) for n, d in self.prompt_registry.items()
            },
        }
        if output_path:
            Path(output_path).write_text(json.dumps(export, indent=2), encoding="utf-8")
        return export

    def get_usage_stats(self) -> Dict[str, Any]:
        if not self.prompt_registry:
            return {"message": "No prompts registered"}
        usage = {n: d["usage_count"] for n, d in self.prompt_registry.items()}
        validation = {n: d["validation_count"] for n, d in self.prompt_registry.items()}
        total_usage = sum(usage.values())
        return {
            "total_prompts": len(self.prompt_registry),
            "total_usage": total_usage,
            "total_validation": sum(validation.values()),
            "average_usage": total_usage / len(self.prompt_registry),
            "most_used": max(usage, key=usage.get, default=None),
            "least_used": min(usage, key=usage.get, default=None),
            "usage_by_prompt": usage,
            "validation_by_prompt": validation,
        }

    # ------------------ Render / Validate ------------------

    def render_template(self, name: str, variables: Dict[str, Any] = None) -> str:
        prompt = self.prompt_registry.get(name)
        if not prompt:
            raise ValueError(f"Template '{name}' not found")
        try:
            rendered = Template(prompt["template"]).render(variables or {})
            prompt["usage_count"] += 1
            return rendered
        except Exception as e:
            raise TemplateError(f"Failed to render '{name}': {e}")

    def validate_output(self, name: str, output: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.prompt_registry.get(name)
        if not prompt:
            return {"valid": False, "error": f"Template '{name}' not found"}
        prompt["validation_count"] += 1
        try:
            validated = prompt["pydantic_model"](**output)
            return {"valid": True, "data": validated.model_dump()}
        except Exception as e:
            return {"valid": False, "error": str(e), "error_type": type(e).__name__}

    def validate_template(self, name: str, test_vars: Dict[str, Any] = None) -> Dict[str, Any]:
        prompt = self.prompt_registry.get(name)
        if not prompt:
            return {"valid": False, "error": f"Template '{name}' not found"}
        env = Environment()
        parsed = env.parse(prompt["template"])
        undeclared = list(meta.find_undeclared_variables(parsed))
        prompt["validation_count"] += 1
        try:
            Template(prompt["template"]).render(test_vars or {})
            return {"valid": True, "undeclared_variables": undeclared}
        except Exception as e:
            return {"valid": False, "error": str(e), "error_type": type(e).__name__}

    # ------------------ Management ------------------

    def remove_prompt(self, name: str) -> bool:
        return self.prompt_registry.pop(name, None) is not None

    def clear_registry(self) -> None:
        self.prompt_registry.clear()
