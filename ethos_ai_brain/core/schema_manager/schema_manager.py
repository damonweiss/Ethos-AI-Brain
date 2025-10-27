#!/usr/bin/env python3
"""
Schema Management System (Pydantic v2 only)

Components:
- SchemaRegistry: storage & metadata for Pydantic schemas
- SchemaValidator: validation in multiple modes
- SchemaLoader: discovery and auto-loading from filesystem
"""

import json
import importlib
import inspect
from pathlib import Path
from enum import Enum
from datetime import datetime
from typing import Dict, Any, List, Type, Optional
from collections import Counter, deque

from pydantic import BaseModel, Field, ValidationError, create_model


# -------------------------
# ENUM: Validation Modes
# -------------------------
class ValidationMode(str, Enum):
    STRICT = "strict"      # Must match exactly
    COERCE = "coerce"      # Convert types when possible
    PARTIAL = "partial"    # Ignore missing fields
    LENIENT = "lenient"    # Field-by-field salvage


# -------------------------
# Schema Registry
# -------------------------
class SchemaRegistry:
    """Registry for Pydantic schemas with rebuild safeguard."""

    def __init__(self):
        self._schemas: Dict[str, Type[BaseModel]] = {}

    def register(self, name: str, schema_class: Type[BaseModel]) -> None:
        if not issubclass(schema_class, BaseModel):
            raise ValueError("Schema must inherit from BaseModel")
        # Ensure schema is fully built
        try:
            schema_class.model_rebuild(force=True)
        except Exception:
            pass
        self._schemas[name] = schema_class

    def create_dynamic(self, name: str, fields: Dict[str, Any]) -> Type[BaseModel]:
        """Create and register a dynamic schema, mapping JSON-style types → Python types."""
        type_mapping = {
            "string": str,
            "str": str,
            "integer": int,
            "int": int,
            "float": float,
            "double": float,
            "number": float,
            "bool": bool,
            "boolean": bool,
            "dict": dict,
            "list": list,
            "array": list,
        }

        pydantic_fields = {}
        for field_name, field_def in fields.items():
            if isinstance(field_def, str):
                # JSON-style type string
                py_type = type_mapping.get(field_def.lower(), str)
                pydantic_fields[field_name] = (py_type, ...)
            elif isinstance(field_def, dict):
                f_type = field_def.get("type", str)
                if isinstance(f_type, str):
                    f_type = type_mapping.get(f_type.lower(), str)

                f_default = field_def.get("default", ...)
                f_desc = field_def.get("description", "")
                if f_default is ...:
                    pydantic_fields[field_name] = (f_type, Field(description=f_desc))
                else:
                    pydantic_fields[field_name] = (
                        f_type,
                        Field(default=f_default, description=f_desc),
                    )
            else:
                # already a Python type
                pydantic_fields[field_name] = (field_def, ...)

        dynamic_model = create_model(name, **pydantic_fields)
        dynamic_model.model_rebuild(force=True)  # ✅ Safe rebuild
        self.register(name, dynamic_model)
        return dynamic_model


    def get(self, name: str) -> Optional[Type[BaseModel]]:
        return self._schemas.get(name)

    def list_schemas(self) -> List[str]:
        return list(self._schemas.keys())

    def info(self, name: str) -> Dict[str, Any]:
        model_class = self._schemas.get(name)
        if not model_class:
            return {"error": f"Schema '{name}' not found"}
        return {
            "schema_name": name,
            "fields": list(model_class.model_fields.keys()),
            "schema_json": json.dumps(model_class.model_json_schema(), indent=2),
        }

    def export(self) -> Dict[str, Any]:
        """Export schema definitions with examples."""
        export_data = {}
        for name, model in self._schemas.items():
            try:
                model.model_rebuild(force=True)
            except Exception:
                pass
            export_data[name] = {
                "class_name": model.__name__,
                "schema": model.model_json_schema(),
                "example": self._generate_example(model),
            }
        return export_data

    def _generate_example(self, model_class: Type[BaseModel]) -> Dict[str, Any]:
        """Generate example data for a schema."""
        example = {}
        for field_name, field in model_class.model_fields.items():
            if field.default is not None and field.default is not ...:
                example[field_name] = field.default
            else:
                f_type = field.annotation
                if f_type == str:
                    example[field_name] = f"example_{field_name}"
                elif f_type == int:
                    example[field_name] = 42
                elif f_type == float:
                    example[field_name] = 3.14
                elif f_type == bool:
                    example[field_name] = True
                else:
                    example[field_name] = f"<{getattr(f_type, '__name__', 'unknown')}>"
        return example


# -------------------------
# Schema Validator
# -------------------------
class SchemaValidator:
    """Handles schema validation in multiple modes with rebuild safety."""

    def __init__(
        self,
        registry: SchemaRegistry,
        default_mode: ValidationMode = ValidationMode.COERCE,
        history_limit: int = 1000
    ):
        self.registry = registry
        self.default_mode = default_mode
        self.history = deque(maxlen=history_limit)

    def validate(
        self,
        data: Dict[str, Any],
        schema_name: str,
        mode: ValidationMode = None,
        model_class: Optional[Type[BaseModel]] = None
    ) -> Dict[str, Any]:
        mode = mode or self.default_mode
        model_class = model_class or self.registry.get(schema_name)
        if not model_class:
            return {"success": False, "error": f"No schema '{schema_name}' registered"}

        try:
            model_class.model_rebuild(force=True)
        except Exception:
            pass

        start = datetime.now()
        try:
            if mode == ValidationMode.STRICT:
                validated = model_class(**data)
            elif mode == ValidationMode.COERCE:
                validated = model_class.model_validate(data)
            elif mode == ValidationMode.PARTIAL:
                filtered = {k: v for k, v in data.items() if k in model_class.model_fields}
                validated = model_class(**filtered)
            elif mode == ValidationMode.LENIENT:
                valid, _ = self._lenient(data, model_class)
                validated = model_class.model_validate(valid)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            result = {
                "success": True,
                "data": validated.model_dump(),
                "mode": mode.value,
                "errors": []
            }

        except ValidationError as e:
            result = {"success": False, "data": {}, "mode": mode.value, "errors": e.errors()}

        result.update({
            "schema": schema_name,
            "duration_ms": (datetime.now() - start).total_seconds() * 1000,
        })
        self.history.append(result)
        return result

    def validate_batch(
        self,
        data_list: List[Dict[str, Any]],
        schema_name: str,
        mode: ValidationMode = None,
        model_class: Optional[Type[BaseModel]] = None
    ):
        return [self.validate(item, schema_name, mode, model_class) for item in data_list]

    def stats(self) -> Dict[str, Any]:
        total = len(self.history)
        if not total:
            return {"message": "No validations yet"}
        successes = sum(1 for h in self.history if h["success"])
        schema_counts = Counter(h["schema"] for h in self.history)
        return {
            "total": total,
            "success_rate": successes / total,
            "schemas": dict(schema_counts),
        }

    def _lenient(self, data: Dict[str, Any], model_class: Type[BaseModel]):
        valid, errors = {}, []
        for k, v in data.items():
            if k not in model_class.model_fields:
                continue
            try:
                temp = model_class(**{k: v})
                valid[k] = getattr(temp, k)
            except Exception as e:
                errors.append({"field": k, "error": str(e)})
        return valid, errors


# -------------------------
# Schema Loader
# -------------------------
class SchemaLoader:
    """Discovers schemas on filesystem and registers them."""

    def __init__(self, registry: SchemaRegistry):
        self.registry = registry

    def auto_load(
        self,
        schemas_path: Optional[Path] = None,
        base_package: str = "ethos_mcp.schemas"
    ) -> Dict[str, Any]:
        results = {"loaded": {}, "failed": []}
        try:
            schemas_path = schemas_path or (Path(__file__).parent.parent.parent / "schemas")
            for py_file in schemas_path.rglob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                try:
                    module_path = str(py_file.relative_to(schemas_path.parent)).replace("/", ".").replace("\\", ".")
                    module = importlib.import_module(module_path)
                    found = []
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel:
                            schema_name = f"{py_file.stem}.{name}"
                            try:
                                obj.model_rebuild(force=True)
                            except Exception:
                                pass
                            self.registry.register(schema_name, obj)
                            found.append(name)
                    if found:
                        results["loaded"][module_path] = found
                except Exception as e:
                    results["failed"].append({"module": str(py_file), "error": str(e)})
        except Exception as e:
            results["failed"].append({"general_error": str(e)})
        return results
