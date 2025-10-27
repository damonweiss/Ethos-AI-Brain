#!/usr/bin/env python3
"""
InferenceModelSelector - Task-based model selection

Uses InferenceModelManager data + optional curated set to select models.
"""

from typing import Dict, Any, Optional, List
import random


class InferenceModelSelector:
    def __init__(self, manager, curated: bool = False):
        """
        Args:
            manager: InferenceModelManager instance
            curated: If True, restrict selection to curated models only
        """
        self.manager = manager
        self.curated = curated
        self.default_model = "gpt-4o-mini"

        # If curated mode is enabled, try to load curated list
        self.curated_models: List[str] = []
        if self.curated:
            self.curated_models = self._load_curated_models()

    def select_model(
        self,
        task: str,
        provider: Optional[str] = None,
        override: Optional[str] = None,
        require: Optional[List[str]] = None,
        cheapest: bool = False,
    ) -> str:
        """
        Pick the best model given constraints.

        Args:
            task: Natural language description (e.g., "summarization")
            provider: Restrict to provider
            override: User-specified model (always wins)
            require: Capabilities that must be present (e.g., ["vision", "json_mode"])
            cheapest: Pick the cheapest valid model

        Returns:
            Selected model name
        """
        if override:
            return override

        # Step 1: Candidate pool
        if self.curated and self.curated_models:
            candidates = self.curated_models
        else:
            candidates = (
                self.manager.get_models_by_provider(provider)
                if provider else list(self.manager.available_models.keys())
            )

        # Step 2: Filter by capability requirements
        if require:
            candidates = [
                m for m in candidates
                if all(self._supports_capability(m, cap) for cap in require)
            ]

        if not candidates:
            return self.default_model

        # Step 3: Cheapest
        if cheapest:
            costs = [(m, self.manager.get_model_cost_info(m)) for m in candidates]
            costs = [(m, c.get("input_cost_per_token", 0) + c.get("output_cost_per_token", 0)) for m, c in costs]
            return min(costs, key=lambda x: x[1])[0]

        # Step 4: Task heuristic
        task_lower = task.lower()
        if "summarize" in task_lower:
            preferred = [m for m in candidates if "gpt" in m or "claude" in m]
            if preferred:
                return preferred[0]

        # Default: random pick
        return random.choice(candidates)

    def _supports_capability(self, model: str, capability: str) -> bool:
        info = self.manager.get_model_info(model)
        if not info:
            return False
        if capability == "vision":
            return info.get("supports_vision", False)
        if capability == "function_calling":
            return info.get("supports_function_calling", False)
        if capability == "json_mode":
            return info.get("supports_response_schema", False)
        return False

    def _load_curated_models(self) -> List[str]:
        """Load curated model list from metadata/curated_llm_models.py."""
        try:
            from pathlib import Path
            metadata_dir = Path(__file__).parent / "metadata"
            curated_file = metadata_dir / "curated_llm_models.py"
            if not curated_file.exists():
                return []
            local_vars = {}
            exec(curated_file.read_text(), {}, local_vars)
            return local_vars.get("curated_llm_models", [])
        except Exception as e:
            print(f"[WARN] Failed to load curated models: {e}")
            return []
