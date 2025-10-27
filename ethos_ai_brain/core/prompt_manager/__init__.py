"""
Prompt Manager - Pure Prompt Registry and Template Management

This manager handles prompt template storage, registration, and retrieval.
It does NOT execute prompts - that's the role of PromptRoutingManager.
"""

from .prompt_manager import PromptManager

__all__ = ['PromptManager']
