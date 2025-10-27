#!/usr/bin/env python3
"""
Model Schemas - Pydantic models for enriched model metadata
"""

from typing import List
from pydantic import BaseModel, Field, ConfigDict


class ModelContext(BaseModel):
    """Model context window information."""
    max_input_tokens: int = Field(description="Maximum input tokens")
    max_output_tokens: int = Field(description="Maximum output tokens")


class ModelModalities(BaseModel):
    """Model modality support."""
    text: bool = Field(default=True, description="Supports text input/output")
    vision: bool = Field(default=False, description="Supports image/vision input")
    audio: bool = Field(default=False, description="Supports audio input/output")
    
    model_config = ConfigDict(extra="allow")


class ModelFeatures(BaseModel):
    """Model feature support."""
    function_calling: bool = Field(default=False, description="Supports function calling")
    json_mode: bool = Field(default=False, description="Supports JSON mode")
    streaming: bool = Field(default=True, description="Supports streaming responses")
    tool_use: bool = Field(default=False, description="Supports tool use")
    
    model_config = ConfigDict(extra="allow")


class ModelPerformance(BaseModel):
    """Model performance characteristics."""
    latency_class: str = Field(description="Latency class: very_low, low, medium, high")
    throughput_class: str = Field(description="Throughput class: low, medium, high")
    stability_class: str = Field(description="Stability class: low, medium, high")


class ModelEconomics(BaseModel):
    """Model economics and pricing."""
    input_cost_per_1k_tokens_usd: float = Field(description="Input cost per 1k tokens in USD")
    output_cost_per_1k_tokens_usd: float = Field(description="Output cost per 1k tokens in USD")
    currency: str = Field(default="USD", description="Currency")
    effective_date: str = Field(description="Effective date (YYYY-MM-DD)")


class TaskFit(BaseModel):
    """Task suitability ratings."""
    reasoning: str = Field(description="Reasoning capability: low, medium, high")
    summarization: str = Field(description="Summarization capability: low, medium, high")
    chat: str = Field(description="Chat capability: low, medium, high")
    coding: str = Field(description="Coding capability: low, medium, high")
    multimodal_analysis: str = Field(description="Multimodal analysis capability: low, medium, high")


class DeploymentFit(BaseModel):
    """Deployment suitability ratings."""
    batch: str = Field(description="Batch processing fit: low, medium, high")
    real_time: str = Field(description="Real-time processing fit: low, medium, high")
    long_context: str = Field(description="Long context processing fit: low, medium, high")


class ModelSuitability(BaseModel):
    """Model suitability for different use cases."""
    task_fit: TaskFit = Field(description="Task-specific suitability")
    deployment_fit: DeploymentFit = Field(description="Deployment-specific suitability")


class EnrichedModelMetadata(BaseModel):
    """Enriched model metadata with structured format."""
    model_id: str = Field(description="Model identifier")
    provider: str = Field(description="Model provider (e.g., openai, anthropic)")
    version: str = Field(description="Model version or release date")
    context: ModelContext = Field(description="Context window information")
    modalities: ModelModalities = Field(description="Supported modalities")
    features: ModelFeatures = Field(description="Supported features")
    performance: ModelPerformance = Field(description="Performance characteristics")
    economics: ModelEconomics = Field(description="Pricing and economics")
    suitability: ModelSuitability = Field(description="Suitability for different use cases")
    limitations: List[str] = Field(description="Known limitations or considerations")
