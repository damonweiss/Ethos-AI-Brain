#!/usr/bin/env python3
"""
Vision Engine - Multimodal engine using LiteLLM (Metadata-driven)
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

import litellm
from litellm import completion
from pydantic import BaseModel

# from ethos_mcp.mcp_managers.inference_engines.inference_engine_base import BaseInferenceEngine
# from ethos_mcp.mcp_managers.inference_engines.llm_engine import FileMetadataStore, MetadataStore
from .inference_engine_base import BaseInferenceEngine
from .llm_engine import FileMetadataStore, MetadataStore

logger = logging.getLogger(__name__)


class VisionEngine(BaseInferenceEngine):
    """Universal vision executor with LiteLLM + enriched metadata."""

    def __init__(self, model: str, metadata_store: Optional[MetadataStore] = None):
        super().__init__(name=f"vision_{model}", engine_type="vision")
        self.model = model
        self.metadata_store = metadata_store or FileMetadataStore()
        self.model_metadata = self.metadata_store.load_model(model) or {
            "model_id": model,
            "provider": "unknown",
            "modalities": {"text": True, "vision": True},
            "context": {"max_input_tokens": 4096, "max_output_tokens": 1024},
            "features": {"json_mode": False, "streaming": True},
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

    def _normalize_schema(self, schema: Any) -> Optional[Dict[str, Any]]:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_json_schema()
        if isinstance(schema, dict):
            return schema
        return None

    def _build_params(
        self,
        input_data: str,
        image_url: str,
        schema_dict: Optional[Dict[str, Any]],
        metadata: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": input_data},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }]

        if self._supports("json_mode", metadata) and schema_dict:
            messages.insert(0, {
                "role": "system",
                "content": f"You are a structured output engine. "
                           f"Return only valid JSON matching this schema: {json.dumps(schema_dict)}"
            })

        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 500),
        }
        if self._supports("json_mode", metadata) and schema_dict:
            params["response_format"] = {"type": "json_object"}
        if self._supports("streaming", metadata) and kwargs.get("streaming", False):
            params["stream"] = True
        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]
        return params

    def _parse_response(self, response: Any) -> Dict[str, Any]:
        try:
            choice = response.choices[0]
            if hasattr(choice, "message"):
                content = getattr(choice.message, "content", "")
                if isinstance(content, list):  # sometimes list-of-dicts
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            parts.append(item["text"])
                        elif isinstance(item, str):
                            parts.append(item)
                    content = " ".join(parts).strip()
                return {"response": content}
            if hasattr(choice, "text"):
                return {"response": choice.text}
            return {"response": str(choice)}
        except Exception:
            return {"response": str(response)}

    def _calculate_cost(self, usage: Dict[str, Any], metadata: Dict[str, Any]) -> float:
        econ = metadata.get("economics", {})
        in_cost = econ.get("input_cost_per_1k_tokens_usd", 0.0)
        out_cost = econ.get("output_cost_per_1k_tokens_usd", 0.0)
        pt = (usage.get("prompt_tokens") or 0) if usage else 0
        ct = (usage.get("completion_tokens") or 0) if usage else 0
        return (pt / 1000) * in_cost + (ct / 1000) * out_cost

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
        self._record_usage()

        metadata = model_metadata or self.model_metadata

        if not isinstance(input_data, str):
            return {"success": False, "error": f"Unsupported input type: {type(input_data)}"}

        image_url = kwargs.pop("image_url", None)
        if not image_url:
            return {"success": False, "error": "VisionEngine requires an image_url"}

        try:
            schema_dict = self._normalize_schema(schema)
            params = self._build_params(input_data, image_url, schema_dict, metadata, **kwargs)

            response = completion(**params)
            result = self._parse_response(response)

            raw_usage = getattr(response, "usage", None) or {}
            cost = self._calculate_cost(raw_usage, metadata)

            return {
                "success": True,
                "result": self._normalize(result),
                "engine": self.name,
                "model": self.model,
                "provider": metadata.get("provider", "unknown"),
                "cost": cost,
                "usage": self._normalize(raw_usage),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "response_id": getattr(response, "id", None),
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}
