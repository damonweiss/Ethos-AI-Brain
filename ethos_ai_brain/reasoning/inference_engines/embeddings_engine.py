import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import litellm 
from litellm import embedding
# from ethos_mcp.mcp_managers.inference_engines.inference_engine_base import BaseInferenceEngine
# from ethos_mcp.mcp_managers.inference_engines.llm_engine import FileMetadataStore, MetadataStore
from .inference_engine_base import BaseInferenceEngine
from .llm_engine import FileMetadataStore, MetadataStore

class EmbeddingsEngine(BaseInferenceEngine):
    """Universal embeddings engine using LiteLLM + enriched metadata JSON."""

    def __init__(self, model: str = "text-embedding-ada-002", metadata_store: Optional[MetadataStore] = None):
        super().__init__(name=f"embeddings_{model}", engine_type="embeddings")
        self.model = model
        self.metadata_store = metadata_store or FileMetadataStore()
        self.model_metadata = self.metadata_store.load_model(model) or {
            "model_id": model,
            "provider": "unknown",
            "features": {"embeddings": True},
            "economics": {"input_cost_per_1k_tokens_usd": 0.0},
        }
        litellm.set_verbose = False

    def _calculate_cost(self, usage: Dict[str, Any], metadata: Dict[str, Any]) -> float:
        econ = metadata.get("economics", {})
        in_cost = econ.get("input_cost_per_1k_tokens_usd", 0.0)
        tokens = (usage.get("total_tokens") or usage.get("prompt_tokens") or 0) if usage else 0
        return (tokens / 1000) * in_cost

    def run(self, input_data: Any, schema: Any = None,
            model_metadata: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        self._record_usage()

        if not isinstance(input_data, str):
            return {"success": False, "error": f"Unsupported input type: {type(input_data)}"}

        metadata = model_metadata or self.model_metadata

        try:
            response = embedding(model=self.model, input=input_data)

            vector, usage, response_id = None, {}, None
            if isinstance(response, dict):
                data = response.get("data", [])
                if data and "embedding" in data[0]:
                    vector = data[0]["embedding"]
                usage = response.get("usage", {})
                response_id = response.get("id")
            else:
                if hasattr(response, "data") and response.data:
                    first = response.data[0]
                    vector = getattr(first, "embedding", None) if not isinstance(first, dict) else first.get("embedding")
                usage = getattr(response, "usage", {}) or {}
                response_id = getattr(response, "id", None)

            if vector is None:
                return {"success": False, "error": "No embedding returned", "raw_response": str(response)}

            cost = self._calculate_cost(usage, metadata)

            return {
                "success": True,
                "result": vector,
                "dim": len(vector),
                "engine": self.name,
                "model": self.model,
                "provider": metadata.get("provider", "unknown"),
                "cost": cost,
                "usage": self._normalize(usage),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "response_id": response_id,
                },
            }

        except Exception as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}
