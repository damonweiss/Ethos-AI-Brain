#!/usr/bin/env python3
"""
InferenceEnginesManager - Engine Registry and Lazy Loading (Clean)

- Registry of engine types
- Lazy loading of engine classes
- Multiple instance support (engine_type + instance_key)
- Keyword-based engine selection
- Stats and cache management
"""

import importlib
import logging
from typing import Dict, Any, List, Optional, Type, Tuple
from datetime import datetime

# from ethos_mcp.mcp_managers.inference_engines.inference_engine_base import BaseInferenceEngine
from ..inference_engines.inference_engine_base import BaseInferenceEngine

logger = logging.getLogger(__name__)


class InferenceEnginesManager:
    """Registry and factory for inference engines."""

    def __init__(self):
        # Engine type → config
        self._registry: Dict[str, Dict[str, Any]] = {
            "llm": {
                "module": "ethos_mcp.mcp_managers.inference_engines.llm_engine",
                "class": "LLMEngine",
                "keywords": {
                    "summarize": 2,
                    "analyze": 2,
                    "generate": 1,
                    "translate": 1,
                    "explain": 2,
                    "reasoning": 2,
                },
                "description": "Large Language Model engine for text and reasoning tasks",
                "registered_at": datetime.now().isoformat(),
            },
        }

        # Caches
        self._loaded_classes: Dict[str, Type[BaseInferenceEngine]] = {}
        self._instances: Dict[Tuple[str, str], BaseInferenceEngine] = {}

    # ---------------------------
    # Registration / Discovery
    # ---------------------------

    def register(
        self,
        engine_type: str,
        module_path: str,
        class_name: str,
        keywords: Optional[Dict[str, int]] = None,
        description: str = "",
    ) -> None:
        """Register a new engine type."""
        self._registry[engine_type] = {
            "module": module_path,
            "class": class_name,
            "keywords": keywords or {},
            "description": description,
            "registered_at": datetime.now().isoformat(),
        }
        logger.info(f"[EngineManager] Registered engine: {engine_type}")

    def types(self) -> List[str]:
        """Return available engine types."""
        return list(self._registry.keys())

    def info(self, engine_type: str) -> Optional[Dict[str, Any]]:
        """Get registry info for an engine type."""
        config = self._registry.get(engine_type)
        if not config:
            return None
        return {
            **config,
            "engine_type": engine_type,
            "loaded": engine_type in self._loaded_classes,
            "instances": [k[1] for k in self._instances.keys() if k[0] == engine_type],
        }

    # ---------------------------
    # Engine Loading
    # ---------------------------

    def _load_class(self, engine_type: str) -> Optional[Type[BaseInferenceEngine]]:
        """Lazy load an engine class."""
        if engine_type in self._loaded_classes:
            return self._loaded_classes[engine_type]

        config = self._registry.get(engine_type)
        if not config:
            raise ValueError(f"Unknown engine type: {engine_type}")

        module = importlib.import_module(config["module"])
        engine_class = getattr(module, config["class"])
        self._loaded_classes[engine_type] = engine_class
        return engine_class

    def create(self, engine_type: str, instance_key: str, **kwargs) -> BaseInferenceEngine:
        """Create a new engine instance and cache it."""
        engine_class = self._load_class(engine_type)

        if engine_type == "llm" and "model" not in kwargs:
            raise ValueError("LLM engine requires a 'model' argument")

        instance = engine_class(**kwargs)
        self._instances[(engine_type, instance_key)] = instance
        return instance

    def get(self, engine_type: str, instance_key: str) -> Optional[BaseInferenceEngine]:
        """Get an existing engine instance."""
        return self._instances.get((engine_type, instance_key))

    def get_or_create(self, engine_type: str, instance_key: str, **kwargs) -> BaseInferenceEngine:
        """Get an existing instance or create a new one."""
        return self.get(engine_type, instance_key) or self.create(engine_type, instance_key, **kwargs)

    # ---------------------------
    # Engine Selection
    # ---------------------------

    def select(self, text: str) -> str:
        """Select the best engine type based on keyword scoring."""
        text_lower = text.lower()
        scores = {}

        for engine_type, config in self._registry.items():
            keywords = config.get("keywords", {})
            score = sum(weight for kw, weight in keywords.items() if kw in text_lower)
            scores[engine_type] = score

        if not scores:
            return "llm"

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "llm"

    # ---------------------------
    # Stats / Cache Management
    # ---------------------------

    def stats(self) -> Dict[str, Any]:
        """Return engine and instance stats."""
        stats = {
            "registered": len(self._registry),
            "loaded": list(self._loaded_classes.keys()),
            "instances": {f"{et}:{ik}": inst.get_info() for (et, ik), inst in self._instances.items()},
            "timestamp": datetime.now().isoformat(),
        }
        return stats

    def clear(self) -> None:
        """Clear cached classes and instances."""
        self._loaded_classes.clear()
        self._instances.clear()
