#!/usr/bin/env python3
"""
InferenceModelManager - Model Discovery and Metadata Management

Manages model discovery, provider grouping, and metadata caching.
Exposes helper classes (selector + enricher) via composition.
"""

from typing import Dict, Any, List, Optional
import json
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# Import helpers
from .inference_model_selector import InferenceModelSelector
from .inference_model_enricher import InferenceModelEnricher


class InferenceModelManager:
    def __init__(self, auto_discover: bool = True):
        self.available_models: Dict[str, Dict[str, Any]] = {}
        self.provider_models: Dict[str, List[str]] = {}
        self.model_costs: Dict[str, Dict[str, Any]] = {}

        # Composition helpers
        self.selector = InferenceModelSelector(self)
        self.enricher = InferenceModelEnricher(self)

        if auto_discover:
            self.discover_models()

    # ---------------------------
    # Model Discovery
    # ---------------------------

    def discover_models(self) -> Dict[str, Any]:
        """Discover models from LiteLLM."""
        try:
            import litellm
            all_models = getattr(litellm, "model_list", [])
            self.provider_models = self._group_models_by_provider(all_models)
            for model in all_models:
                self.available_models[model] = {"name": model}
            return {
                "total_models": len(all_models),
                "providers": list(self.provider_models.keys()),
            }
        except Exception as e:
            return {"error": f"Failed to discover models: {e}"}

    # ---------------------------
    # Info Getters
    # ---------------------------

    def get_models_by_provider(self, provider: str) -> List[str]:
        return self.provider_models.get(provider.lower(), [])

    def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        if model not in self.available_models:
            return None
        return {**self.available_models[model], **self.get_model_cost_info(model)}

    def get_model_cost_info(self, model: str) -> Dict[str, Any]:
        if model in self.model_costs:
            return self.model_costs[model]

        def _get_cost_info():
            import litellm
            if hasattr(litellm, "get_model_info"):
                return litellm.get_model_info(model)
            elif hasattr(litellm, "model_cost"):
                return litellm.model_cost.get(model, {})
            return {}

        cost_info = self._suppress_output(_get_cost_info)
        if "error" not in cost_info:
            self.model_costs[model] = cost_info
        return cost_info

    def get_available_providers(self) -> List[str]:
        return list(self.provider_models.keys())

    def get_provider_summary(self, provider: str) -> Dict[str, Any]:
        models = self.get_models_by_provider(provider)
        return {
            "provider": provider,
            "total_models": len(models),
            "sample_models": models[:3],
        }

    # ---------------------------
    # Internals
    # ---------------------------

    def _group_models_by_provider(self, all_models: List[str]) -> Dict[str, List[str]]:
        patterns = {
            "openai": ["gpt-", "davinci", "/openai/"],
            "anthropic": ["claude", "/anthropic/"],
            "google": ["gemini", "/google/"],
            "meta": ["llama", "/meta/"],
            "mistral": ["mistral", "mixtral"],
        }
        grouped = {p: [] for p in patterns}
        for m in all_models:
            for provider, pats in patterns.items():
                if any(pat in m.lower() for pat in pats):
                    grouped[provider].append(m)
        return {k: v for k, v in grouped.items() if v}

    def _suppress_output(self, func, *args, **kwargs):
        buf_out, buf_err = StringIO(), StringIO()
        try:
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                return func(*args, **kwargs)
        except Exception as e:
            return {"error": str(e)}
