"""
Inference Manager - Inference Engine Registry and Management

This manager handles inference engine registration and execution.
Supports LLMs (OpenAI, Claude, Cohere), ML models, NLP engines, and custom inference methods.
"""

from .inference_model_manager import InferenceModelManager
from ..inference_engines import BaseInferenceEngine

__all__ = ['InferenceManager', 'BaseInferenceEngine']
