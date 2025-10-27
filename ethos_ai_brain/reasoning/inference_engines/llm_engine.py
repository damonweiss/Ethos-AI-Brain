#!/usr/bin/env python3
"""
LLM Engine - Universal LLM interface using LiteLLM (Metadata-driven)

Implements BaseInferenceEngine for text-based models.
- Reads model behavior from metadata JSON (via MetadataStore)
- Executes completions via LiteLLM
- Allows full override of model metadata at runtime
- Normalizes outputs, usage, and cost reports
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import litellm
from litellm import completion
from pydantic import BaseModel

from .inference_engine_base import BaseInferenceEngine

logger = logging.getLogger(__name__)


# -------------------------
# Metadata Store
# -------------------------

class MetadataStore:
    """Abstract metadata provider for model descriptions."""
    def load_model(self, model_id: str) -> Dict[str, Any]:
        raise NotImplementedError


class FileMetadataStore(MetadataStore):
    """Load model metadata from a local JSON file with caching."""
    def __init__(self, metadata_file: Optional[Path] = None):
        self.metadata_file = metadata_file or (
            Path(__file__).parent.parent
            / "inference_model_manager"
            / "metadata"
            / "inference_model_metadata.json"
        )
        self._cache: Optional[Dict[str, Any]] = None

    def _load_all(self) -> Dict[str, Any]:
        if self._cache is None:
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata file {self.metadata_file}: {e}")
                self._cache = {}
        return self._cache

    def load_model(self, model_id: str) -> Dict[str, Any]:
        return self._load_all().get(model_id, {})


# -------------------------
# LLM Engine
# -------------------------

class LLMEngine(BaseInferenceEngine):
    """Universal LLM executor with LiteLLM + enriched metadata."""

    def __init__(self, model: str, metadata_store: Optional[MetadataStore] = None):
        super().__init__(name=f"llm_{model}", engine_type="llm")
        self.model = model
        self.metadata_store = metadata_store or FileMetadataStore()
        self.model_metadata = self.metadata_store.load_model(model)

        # Fallback defaults if no metadata found
        if not self.model_metadata:
            logger.warning(f"No enriched metadata found for model {model}. Using fallback defaults.")
            self.model_metadata = {
                "model_id": model,
                "provider": "unknown",
                "context": {"max_input_tokens": 4096, "max_output_tokens": 1024},
                "features": {
                    "json_mode": False,
                    "function_calling": False,
                    "streaming": True,
                },
                "economics": {
                    "input_cost_per_1k_tokens_usd": 0.0,
                    "output_cost_per_1k_tokens_usd": 0.0,
                },
            }

        litellm.set_verbose = False

    # -------------------------
    # Helpers
    # -------------------------

    def _supports(self, feature: str, metadata: Dict[str, Any]) -> bool:
        return bool(metadata.get("features", {}).get(feature, False))

    def _get_costs(self, metadata: Dict[str, Any]) -> tuple[float, float]:
        econ = metadata.get("economics", {})
        return (
            econ.get("input_cost_per_1k_tokens_usd", 0.0),
            econ.get("output_cost_per_1k_tokens_usd", 0.0),
        )

    def _calculate_cost(self, usage: Dict[str, Any], metadata: Dict[str, Any]) -> float:
        """Compute token cost if economics available."""
        in_cost, out_cost = self._get_costs(metadata)
        pt = (usage.get("prompt_tokens") or 0) if usage else 0
        ct = (usage.get("completion_tokens") or 0) if usage else 0
        return (pt / 1000) * in_cost + (ct / 1000) * out_cost

    def _normalize_schema(self, schema: Any) -> Dict[str, Any]:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_json_schema()
        if isinstance(schema, dict):
            return schema
        return {"result": "string"}  # fallback

    def _build_params(
        self, input_data: str, schema_dict: Dict[str, Any], metadata: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "messages": [
                {
                    "role": "system",
                    "content": f"Respond with JSON matching this schema: {json.dumps(schema_dict)}",
                },
                {"role": "user", "content": input_data},
            ],
        }
        if self._supports("json_mode", metadata):
            params["response_format"] = {"type": "json_object"}
        if self._supports("function_calling", metadata) and "functions" in kwargs:
            params["functions"] = kwargs["functions"]
        if self._supports("streaming", metadata) and kwargs.get("streaming", False):
            params["stream"] = True
        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]
        return params

    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Try structured JSON parse, fallback to raw text."""
        try:
            if hasattr(response.choices[0], "message"):
                return json.loads(response.choices[0].message.content)
            if hasattr(response.choices[0], "text"):
                return json.loads(response.choices[0].text)
        except Exception:
            if hasattr(response.choices[0], "message"):
                return {"response": response.choices[0].message.content}
            if hasattr(response.choices[0], "text"):
                return {"response": response.choices[0].text}
        return {"response": str(response)}

    # -------------------------
    # Public API
    # -------------------------

    def run(
        self,
        input_data: Any,
        schema: Any = None,
        model_metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run completion against the model.
        - input_data: user input (string required)
        - schema: optional JSON schema (dict or Pydantic model)
        - model_metadata: optional override for metadata (fallbacks to stored)
        """
        self._record_usage()

        if not isinstance(input_data, str):
            return {"success": False, "error": f"Unsupported input type: {type(input_data)}"}

        # ✅ allow override
        metadata = model_metadata or self.model_metadata

        try:
            schema_dict = self._normalize_schema(schema)
            params = self._build_params(input_data, schema_dict, metadata, **kwargs)

            response = completion(**params)
            result = self._parse_response(response)

            raw_usage = getattr(response, "usage", None) or {}
            cost = self._calculate_cost(raw_usage, metadata)

            return {
                "success": True,
                "result": self._normalize(result),   # ✅ from Base
                "engine": self.name,
                "model": self.model,
                "provider": metadata.get("provider", "unknown"),
                "cost": cost,
                "usage": self._normalize(raw_usage), # ✅ from Base
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "response_id": getattr(response, "id", None),
                },
            }

        except Exception as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}
