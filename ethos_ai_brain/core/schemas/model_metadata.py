#!/usr/bin/env python3
"""
Model Metadata Schema - Pydantic models for consistent model metadata structure.

Used by update_metadata scripts to ensure consistent data representation
and AI-enhanced model analysis.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator


class ModelPricing(BaseModel):
    """Pricing information for a model."""
    input_cost_per_1k_tokens: float = Field(..., description="Cost per 1K input tokens in USD")
    output_cost_per_1k_tokens: float = Field(..., description="Cost per 1K output tokens in USD")
    currency: str = Field(default="USD", description="Currency for pricing")
    effective_date: str = Field(..., description="Date when pricing became effective (YYYY-MM-DD)")
    source: str = Field(..., description="Source of pricing data (api, website, etc.)")


class ModelCapabilityScores(BaseModel):
    """Capability scores for different task categories (0-100 scale)."""
    creative_writing: int = Field(..., ge=0, le=100, description="Score for creative writing tasks")
    analytical_reasoning: int = Field(..., ge=0, le=100, description="Score for analytical and logical reasoning")
    coding_programming: int = Field(..., ge=0, le=100, description="Score for coding and programming tasks")
    mathematical_computation: int = Field(..., ge=0, le=100, description="Score for mathematical problem solving")
    language_translation: int = Field(..., ge=0, le=100, description="Score for language translation tasks")
    summarization: int = Field(..., ge=0, le=100, description="Score for text summarization")
    classification: int = Field(..., ge=0, le=100, description="Score for classification and categorization")
    question_answering: int = Field(..., ge=0, le=100, description="Score for question answering")
    conversation: int = Field(..., ge=0, le=100, description="Score for conversational interactions")
    instruction_following: int = Field(..., ge=0, le=100, description="Score for following complex instructions")


class ModelPerformance(BaseModel):
    """Performance characteristics of a model."""
    speed: str = Field(..., description="Speed tier: very_fast, fast, medium, slow")
    quality: str = Field(..., description="Quality tier: excellent, very_high, high, good, fair")
    reliability: str = Field(..., description="Reliability tier: very_high, high, medium, low")
    consistency: str = Field(..., description="Output consistency: very_high, high, medium, low")


class ModelTechnicalSpecs(BaseModel):
    """Technical specifications of a model."""
    context_window: int = Field(..., description="Maximum context window size in tokens")
    max_output_tokens: int = Field(..., description="Maximum output tokens per request")
    supports_json_mode: bool = Field(default=False, description="Whether model supports JSON output mode")
    supports_function_calling: bool = Field(default=False, description="Whether model supports function calling")
    supports_streaming: bool = Field(default=True, description="Whether model supports streaming responses")
    supports_vision: bool = Field(default=False, description="Whether model can process images")
    training_cutoff: Optional[str] = Field(None, description="Knowledge cutoff date (YYYY-MM-DD)")


class ModelMetadata(BaseModel):
    """Complete metadata for a single model."""
    # Basic Information
    model_id: str = Field(..., description="Official model identifier")
    display_name: str = Field(..., description="Human-readable model name")
    description: str = Field(..., description="Detailed description of the model")
    provider: str = Field(..., description="Model provider (OpenAI, Anthropic, etc.)")
    
    # Capabilities and Scoring
    capability_scores: ModelCapabilityScores = Field(..., description="Scores for different capability categories")
    primary_use_cases: List[str] = Field(..., description="Primary recommended use cases")
    strengths: List[str] = Field(..., description="Key strengths of this model")
    limitations: List[str] = Field(..., description="Known limitations or weaknesses")
    
    # Technical Specifications
    technical_specs: ModelTechnicalSpecs = Field(..., description="Technical specifications")
    performance: ModelPerformance = Field(..., description="Performance characteristics")
    
    # Pricing and Availability
    pricing: ModelPricing = Field(..., description="Pricing information")
    availability: str = Field(..., description="Availability status: available, limited, deprecated")
    
    # Metadata
    priority: int = Field(..., ge=0, le=100, description="Overall priority score for selection")
    last_updated: str = Field(..., description="Last update timestamp (ISO format)")
    data_sources: List[str] = Field(..., description="Sources of data for this model")
    
    @validator('priority')
    def validate_priority(cls, v, values):
        """Calculate priority based on capability scores and other factors."""
        if 'capability_scores' in values:
            scores = values['capability_scores']
            # Average of top capabilities as base priority
            avg_score = sum([
                scores.creative_writing,
                scores.analytical_reasoning,
                scores.coding_programming,
                scores.mathematical_computation
            ]) / 4
            return min(100, max(0, int(avg_score)))
        return v


class EngineMetadata(BaseModel):
    """Complete metadata for an inference engine."""
    engine_info: Dict[str, Any] = Field(..., description="Engine information")
    models: Dict[str, ModelMetadata] = Field(..., description="Dictionary of model metadata")
    usage_statistics: Dict[str, Any] = Field(default_factory=dict, description="Usage statistics")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="Engine preferences")
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class ModelAnalysisPrompt(BaseModel):
    """Prompt template for AI-enhanced model analysis."""
    
    @staticmethod
    def create_analysis_prompt(raw_model_data: Dict[str, Any], provider: str) -> str:
        """Create a prompt for AI analysis of model data."""
        return f"""
You are an AI model analysis expert. Analyze the following model data and provide comprehensive, accurate metadata.

PROVIDER: {provider}
RAW MODEL DATA: {raw_model_data}

Please analyze this model and provide detailed metadata following these guidelines:

1. CAPABILITY SCORING (0-100 scale):
   - Score each capability based on the model's actual strengths
   - Consider model size, training, and known performance benchmarks
   - Be realistic - not all models excel at everything
   - Use these categories:
     * creative_writing: Fiction, poetry, creative content
     * analytical_reasoning: Logic, analysis, complex reasoning
     * coding_programming: Code generation, debugging, technical tasks
     * mathematical_computation: Math problems, calculations, quantitative analysis
     * language_translation: Multi-language tasks
     * summarization: Text summarization and extraction
     * classification: Categorization and labeling tasks
     * question_answering: Direct Q&A and information retrieval
     * conversation: Natural dialogue and chat
     * instruction_following: Following complex, multi-step instructions

2. USE CASES & STRENGTHS:
   - Identify 3-5 primary use cases where this model excels
   - List 3-5 key strengths
   - Note 2-3 limitations or weaknesses

3. PERFORMANCE ASSESSMENT:
   - Speed: very_fast, fast, medium, slow (based on model size/complexity)
   - Quality: excellent, very_high, high, good, fair
   - Reliability: very_high, high, medium, low
   - Consistency: very_high, high, medium, low

4. PRIORITY CALCULATION:
   - Overall priority (0-100) considering capability, cost, and utility
   - Higher scores for more versatile, cost-effective models

Provide your analysis as a JSON object matching the ModelMetadata schema.
Be thorough but concise. Focus on practical, actionable insights.
"""


# Scoring rubrics for consistency
CAPABILITY_SCORING_RUBRIC = {
    "creative_writing": {
        90-100: "Exceptional creative writing, storytelling, and artistic expression",
        80-89: "Very strong creative abilities with nuanced style",
        70-79: "Good creative writing with some limitations",
        60-69: "Basic creative abilities, functional but not exceptional",
        50-59: "Limited creative writing capabilities",
        0-49: "Poor or no creative writing abilities"
    },
    "analytical_reasoning": {
        90-100: "Advanced logical reasoning, complex problem solving",
        80-89: "Strong analytical capabilities with good reasoning",
        70-79: "Good analytical skills for most tasks",
        60-69: "Basic analytical reasoning, handles simple problems",
        50-59: "Limited analytical capabilities",
        0-49: "Poor reasoning and analytical skills"
    },
    "coding_programming": {
        90-100: "Expert-level code generation, debugging, architecture",
        80-89: "Strong programming across multiple languages",
        70-79: "Good coding abilities for common tasks",
        60-69: "Basic programming, simple scripts and functions",
        50-59: "Limited coding capabilities",
        0-49: "Poor or no programming abilities"
    }
}
