#!/usr/bin/env python3
"""
BaseInferenceEngine - Abstract contract for all inference engines.

Each engine subclass:
  - Knows how to run a request for its modality (LLM, Vision, Embeddings, etc.).
  - Implements `.run(model_metadata, input_data, schema, **kwargs)`.

Responsibilities:
  - Record usage stats
  - Normalize outputs to JSON-safe dicts
  - Provide registry metadata (priority, cost tier, capabilities, use cases)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime


class BaseInferenceEngine(ABC):
    """Abstract base class for all inference engines."""

    def __init__(self, name: str, engine_type: str = "unknown") -> None:
        self.name: str = name
        self.engine_type: str = engine_type
        self.usage_count: int = 0
        self.last_used: Optional[str] = None

    # ------------------------------------------------------------------
    # Core contract
    # ------------------------------------------------------------------

    @abstractmethod
    def run(
        self,
        model_metadata: Dict[str, Any],
        input_data: Any,
        schema: Any = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a single inference request.

        Args:
            model_metadata: Metadata dict describing the model
            input_data: Primary input to the engine (text, image, etc.)
            schema: Optional schema (dict or Pydantic model) for validation
            **kwargs: Engine-specific execution params

        Returns:
            Dict with standardized keys:
                - success: bool
                - result: Any (normalized output)
                - usage: dict (tokens, API calls, etc.)
                - cost: float (USD or 0.0 if unknown)
                - metadata: dict (timestamps, response_id, etc.)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Usage + cost helpers
    # ------------------------------------------------------------------

    def _record_usage(self) -> None:
        """Increment usage counter and update last-used timestamp."""
        self.usage_count += 1
        self.last_used = datetime.now().isoformat()

    def _calculate_cost(self, usage: Dict[str, Any], model_metadata: Dict[str, Any]) -> float:
        """
        Estimate cost of a run given usage and model metadata.
        Default returns 0.0 — engines with economics must override.
        """
        return 0.0

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _normalize(self, obj: Any) -> Any:
        """Recursively normalize provider responses to JSON-safe structures."""
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {k: self._normalize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._normalize(v) for v in obj]
        if hasattr(obj, "model_dump"):  # Pydantic v2 BaseModel
            return self._normalize(obj.model_dump())
        if hasattr(obj, "dict"):  # Pydantic v1 fallback
            return self._normalize(obj.dict())
        if hasattr(obj, "__dict__"):
            return self._normalize(vars(obj))
        return str(obj)

    # ------------------------------------------------------------------
    # Availability + registry metadata
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Override if the engine requires runtime checks (API keys, GPU, etc.)."""
        return True

    def get_info(self) -> Dict[str, Any]:
        """Return registry metadata about this engine implementation."""
        return {
            "name": self.name,
            "type": self.engine_type,
            "priority": self.get_priority(),
            "cost_tier": self.get_cost_tier(),
            "capabilities": self.get_capabilities(),
            "use_cases": self.get_use_cases(),
            "usage_count": self.usage_count,
            "last_used": self.last_used,
            "available": self.is_available(),
        }

    # ------------------------------------------------------------------
    # Override points for subclasses
    # ------------------------------------------------------------------

    def get_priority(self) -> int:
        """Relative priority (0–100). Lower = preferred. Default = 50."""
        return 50

    def get_cost_tier(self) -> str:
        """Rough classification: low / medium / high. Default = medium."""
        return "medium"

    def get_capabilities(self) -> List[str]:
        """List of capability tags for routing (e.g., 'chat', 'vision', 'embeddings')."""
        return []

    def get_use_cases(self) -> List[str]:
        """High-level use cases supported by this engine."""
        return []
