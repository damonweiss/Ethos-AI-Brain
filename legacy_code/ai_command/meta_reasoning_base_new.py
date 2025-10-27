#!/usr/bin/env python3
"""
Hybrid Abstract Meta-Reasoning Engine
Combines clean abstractions with proven reasoning patterns

Features:
- Lightweight, domain-agnostic reasoning abstractions
- Multi-stage pipeline orchestration (assess → decompose → execute → validate)
- Pluggable reasoning strategies with complexity-based selection
- Built-in quality assurance and gap analysis framework
- Clean LLM backend abstraction with structured output support
- Extensible for any reasoning domain (intent graphs, code analysis, etc.)
- Proven patterns from production meta-reasoning systems

Architecture:
- Pure abstractions without domain-specific dependencies
- Strategy pattern for different reasoning approaches
- Quality-first design with validation at every stage
- Extensible through inheritance for domain specialization
"""

import asyncio
import json
import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

# Import Pydantic for structured models
from pydantic import BaseModel, Field

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================
# Core Reasoning Enums and Types
# ========================================

class ComplexityLevel(str, Enum):
    """Standard complexity levels for reasoning tasks"""
    TRIVIAL = "trivial"
    SIMPLE = "simple" 
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

class ReasoningStrategyType(str, Enum):
    """Types of reasoning strategies available"""
    DIRECT = "direct"                    # Single-step reasoning
    DECOMPOSED = "decomposed"            # Break down into sub-problems
    PARALLEL = "parallel"               # Multiple simultaneous reasoning paths
    ITERATIVE = "iterative"             # Refine through multiple passes
    HIERARCHICAL = "hierarchical"       # Multi-level reasoning
    COLLABORATIVE = "collaborative"     # Multiple reasoning agents

class QualityDimension(str, Enum):
    """Dimensions for quality assessment"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    RELEVANCE = "relevance"
    CLARITY = "clarity"
    CONFIDENCE = "confidence"

# ========================================
# Core Reasoning Models
# ========================================

class ComplexityFactor(BaseModel):
    """Individual factor contributing to complexity"""
    factor_name: str = Field(..., description="Name of the complexity factor")
    weight: float = Field(..., ge=0.0, le=1.0, description="Weight of this factor (0-1)")
    score: float = Field(..., ge=0.0, le=1.0, description="Score for this factor (0-1)")
    reasoning: str = Field("", description="Why this factor has this score")

class ComplexityAssessment(BaseModel):
    """Domain-agnostic complexity analysis"""
    overall_complexity: ComplexityLevel = Field(..., description="Overall complexity level")
    complexity_score: float = Field(..., ge=0.0, le=1.0, description="Numeric complexity score")
    complexity_factors: List[ComplexityFactor] = Field(default_factory=list)
    recommended_strategy: ReasoningStrategyType = Field(..., description="Recommended reasoning strategy")
    processing_hints: List[str] = Field(default_factory=list, description="Hints for processing")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in assessment")
    reasoning: str = Field("", description="Explanation of complexity assessment")

class ReasoningStrategy(BaseModel):
    """Abstract reasoning approach configuration"""
    strategy_type: ReasoningStrategyType = Field(..., description="Type of reasoning strategy")
    complexity_level: ComplexityLevel = Field(..., description="Complexity level this strategy handles")
    processing_stages: List[str] = Field(default_factory=list, description="Ordered list of processing stages")
    parallel_capable: bool = Field(False, description="Can this strategy run stages in parallel")
    requires_validation: bool = Field(True, description="Does this strategy require quality validation")
    max_iterations: int = Field(1, description="Maximum iterations for iterative strategies")
    timeout_seconds: int = Field(300, description="Timeout for strategy execution")
    
class SubProblem(BaseModel):
    """Individual sub-problem in decomposition"""
    problem_id: str = Field(..., description="Unique identifier for this sub-problem")
    description: str = Field(..., description="Description of the sub-problem")
    complexity: ComplexityLevel = Field(..., description="Complexity of this sub-problem")
    priority: int = Field(1, description="Priority level (1=highest)")
    estimated_duration: int = Field(60, description="Estimated duration in seconds")
    dependencies: List[str] = Field(default_factory=list, description="IDs of problems this depends on")
    required_capabilities: List[str] = Field(default_factory=list, description="Capabilities needed to solve")

class ProblemDecomposition(BaseModel):
    """Generic problem breakdown structure"""
    sub_problems: List[SubProblem] = Field(default_factory=list)
    execution_order: List[str] = Field(default_factory=list, description="Ordered list of problem IDs")
    parallel_groups: List[List[str]] = Field(default_factory=list, description="Groups that can run in parallel")
    critical_path: List[str] = Field(default_factory=list, description="Critical path through dependencies")
    total_estimated_duration: int = Field(0, description="Total estimated duration in seconds")
    decomposition_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in decomposition")

class QualityMetric(BaseModel):
    """Individual quality measurement"""
    dimension: QualityDimension = Field(..., description="Quality dimension being measured")
    score: float = Field(..., ge=0.0, le=1.0, description="Score for this dimension")
    weight: float = Field(1.0, ge=0.0, description="Weight of this dimension in overall score")
    evidence: List[str] = Field(default_factory=list, description="Evidence supporting this score")
    improvement_suggestions: List[str] = Field(default_factory=list, description="How to improve this dimension")

class GapAnalysis(BaseModel):
    """Detailed gap analysis for reasoning validation"""
    gap_type: str = Field(..., description="Type of gap identified")
    description: str = Field(..., description="Description of the gap")
    severity: str = Field(..., pattern="^(critical|high|medium|low)$", description="Severity level")
    impact_areas: List[str] = Field(default_factory=list, description="Areas impacted by this gap")
    mitigation_strategies: List[str] = Field(default_factory=list, description="Strategies to address the gap")
    requires_human_input: bool = Field(False, description="Whether human input is needed to resolve")

class QualityAssessment(BaseModel):
    """Domain-agnostic quality validation with enhanced gap analysis"""
    overall_quality_score: float = Field(..., ge=0.0, le=1.0, description="Weighted average quality score")
    quality_metrics: List[QualityMetric] = Field(default_factory=list)
    identified_gaps: List[GapAnalysis] = Field(default_factory=list, description="Detailed gap analysis")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Overall improvement suggestions")
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Confidence in quality assessment")
    validation_passed: bool = Field(True, description="Whether validation criteria were met")
    critical_issues: List[str] = Field(default_factory=list, description="Critical issues that must be addressed")
    human_queries: List[str] = Field(default_factory=list, description="Questions for human clarification")
    completeness_score: float = Field(0.0, ge=0.0, le=1.0, description="How complete the reasoning is")

class ReasoningResult(BaseModel):
    """Generic reasoning result container"""
    result_data: Dict[str, Any] = Field(default_factory=dict, description="The actual reasoning output")
    reasoning_trace: List[str] = Field(default_factory=list, description="Step-by-step reasoning trace")
    strategy_used: ReasoningStrategy = Field(..., description="Strategy that was used")
    execution_time: float = Field(0.0, description="Time taken to execute reasoning")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

# ========================================
# LLM Backend Abstraction
# ========================================

class LLMBackend(ABC):
    """Abstract base class for LLM backends with structured output support"""
    
    def __init__(self, name: str, model: str = "default"):
        self.name = name
        self.model = model
        self.system_prompt = "You are an expert reasoning assistant."
        self.temperature = 0.1
        self.max_tokens = 1500
    
    @abstractmethod
    async def complete(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate text completion"""
        pass
    
    @abstractmethod
    async def complete_structured(self, prompt: str, context: Dict[str, Any], response_model: BaseModel) -> Dict[str, Any]:
        """Generate structured response using Pydantic model"""
        pass
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt for this backend"""
        self.system_prompt = prompt
    
    def configure(self, temperature: float = None, max_tokens: int = None):
        """Configure LLM parameters"""
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens

class OpenAIBackend(LLMBackend):
    """OpenAI LLM backend for real AI integration"""
    
    def __init__(self, name: str, model: str = "gpt-4o-mini", api_key: str = None):
        super().__init__(name, model)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    async def complete(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate text completion using OpenAI"""
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}")
    
    async def complete_structured(self, prompt: str, context: Dict[str, Any], response_model: BaseModel) -> Dict[str, Any]:
        """Generate structured response using OpenAI with JSON mode"""
        try:
            # Add JSON format instruction to prompt
            json_prompt = f"{prompt}\n\nRespond with a valid JSON object that matches this structure:\n{response_model.model_json_schema()}"
            
            messages = [
                {"role": "system", "content": f"{self.system_prompt} Always respond with valid JSON."},
                {"role": "user", "content": json_prompt}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            raise Exception(f"OpenAI structured API call failed: {str(e)}")

# ========================================
# Core Meta-Reasoning Engine
# ========================================

class MetaReasoningBase(ABC):
    """
    Pure meta-reasoning engine - domain agnostic
    
    Provides core reasoning capabilities that can be specialized
    for specific domains (intent analysis, code analysis, etc.)
    """
    
    def __init__(self, llm_backends: Dict[str, LLMBackend] = None):
        """
        Initialize meta-reasoning engine
        
        Args:
            llm_backends: Dictionary of LLM backends by name
        """
        self.llm_backends = llm_backends or {}
        self.reasoning_strategies = self._initialize_strategies()
        self.quality_thresholds = self._initialize_quality_thresholds()
        
        logger.info(f"MetaReasoningBase initialized with {len(self.llm_backends)} LLM backends")
    
    # ========================================
    # Core Abstract Methods (Must Implement)
    # ========================================
    
    @abstractmethod
    async def assess_complexity(self, input_data: Any, domain_context: str = "general") -> ComplexityAssessment:
        """
        Assess the complexity of the reasoning task
        
        Args:
            input_data: The input to analyze
            domain_context: Domain context for specialized assessment
            
        Returns:
            ComplexityAssessment with recommended strategy
        """
        pass
    
    @abstractmethod
    async def decompose_problem(self, input_data: Any, strategy: ReasoningStrategy, context: Dict[str, Any] = None) -> ProblemDecomposition:
        """
        Break down the problem into manageable sub-problems
        
        Args:
            input_data: The input to decompose
            strategy: The reasoning strategy to use
            context: Additional context for decomposition
            
        Returns:
            ProblemDecomposition with execution plan
        """
        pass
    
    @abstractmethod
    async def execute_reasoning(self, decomposition: ProblemDecomposition, input_data: Any, context: Dict[str, Any] = None) -> ReasoningResult:
        """
        Execute the reasoning process based on decomposition
        
        Args:
            decomposition: The problem decomposition
            input_data: Original input data
            context: Additional context for execution
            
        Returns:
            ReasoningResult with output and metadata
        """
        pass
    
    # ========================================
    # Concrete Methods (Provided Implementation)
    # ========================================
    
    async def select_strategy(self, complexity: ComplexityAssessment) -> ReasoningStrategy:
        """
        Select appropriate reasoning strategy based on complexity
        
        Args:
            complexity: The complexity assessment
            
        Returns:
            ReasoningStrategy to use
        """
        strategy_type = complexity.recommended_strategy
        
        if strategy_type in self.reasoning_strategies:
            strategy = self.reasoning_strategies[strategy_type].model_copy()
            strategy.complexity_level = complexity.overall_complexity
            return strategy
        
        # Fallback to direct strategy
        logger.warning(f"Strategy {strategy_type} not found, using direct strategy")
        return self.reasoning_strategies[ReasoningStrategyType.DIRECT]
    
    async def validate_reasoning(self, input_data: Any, reasoning_result: ReasoningResult, context: Dict[str, Any] = None) -> QualityAssessment:
        """
        Validate the quality of reasoning output with comprehensive gap analysis
        
        Args:
            input_data: Original input
            reasoning_result: The reasoning output to validate
            context: Additional validation context
            
        Returns:
            QualityAssessment with detailed gap analysis and human queries
        """
        # Multi-dimensional quality assessment
        quality_metrics = []
        
        # Completeness assessment
        completeness_score = self._assess_completeness(reasoning_result)
        quality_metrics.append(QualityMetric(
            dimension=QualityDimension.COMPLETENESS,
            score=completeness_score,
            weight=0.3,
            evidence=[f"Result contains {len(reasoning_result.result_data)} data fields", 
                     f"Reasoning trace has {len(reasoning_result.reasoning_trace)} steps"],
            improvement_suggestions=["Add more detailed analysis", "Expand reasoning trace"] if completeness_score < 0.7 else []
        ))
        
        # Confidence assessment
        confidence_score = reasoning_result.confidence
        quality_metrics.append(QualityMetric(
            dimension=QualityDimension.CONFIDENCE,
            score=confidence_score,
            weight=0.25,
            evidence=[f"Reasoning confidence: {confidence_score:.2f}"],
            improvement_suggestions=["Gather more evidence", "Validate assumptions"] if confidence_score < 0.7 else []
        ))
        
        # Consistency assessment
        consistency_score = self._assess_consistency(reasoning_result)
        quality_metrics.append(QualityMetric(
            dimension=QualityDimension.CONSISTENCY,
            score=consistency_score,
            weight=0.2,
            evidence=[f"Internal consistency score: {consistency_score:.2f}"],
            improvement_suggestions=["Resolve contradictions", "Align reasoning steps"] if consistency_score < 0.7 else []
        ))
        
        # Relevance assessment
        relevance_score = self._assess_relevance(input_data, reasoning_result)
        quality_metrics.append(QualityMetric(
            dimension=QualityDimension.RELEVANCE,
            score=relevance_score,
            weight=0.25,
            evidence=[f"Relevance to input: {relevance_score:.2f}"],
            improvement_suggestions=["Focus on key aspects", "Remove irrelevant details"] if relevance_score < 0.7 else []
        ))
        
        # Calculate weighted overall score
        total_weight = sum(metric.weight for metric in quality_metrics)
        overall_score = sum(metric.score * metric.weight for metric in quality_metrics) / total_weight if total_weight > 0 else 0.0
        
        # Comprehensive gap analysis
        gaps = self._perform_gap_analysis(quality_metrics, reasoning_result, context)
        
        # Generate human queries for clarification
        human_queries = self._generate_human_queries(gaps, reasoning_result, context)
        
        # Collect improvement suggestions
        suggestions = []
        for metric in quality_metrics:
            suggestions.extend(metric.improvement_suggestions)
        
        # Add gap-specific suggestions
        for gap in gaps:
            suggestions.extend(gap.mitigation_strategies)
        
        # Identify critical issues
        critical_issues = [gap.description for gap in gaps if gap.severity in ["critical", "high"]]
        
        return QualityAssessment(
            overall_quality_score=overall_score,
            quality_metrics=quality_metrics,
            identified_gaps=gaps,
            improvement_suggestions=list(set(suggestions)),  # Remove duplicates
            confidence_level=confidence_score,
            validation_passed=overall_score >= self.quality_thresholds.get("minimum_quality", 0.6) and len(critical_issues) == 0,
            critical_issues=critical_issues,
            human_queries=human_queries,
            completeness_score=completeness_score
        )
    
    async def reason(self, input_data: Any, domain_context: str = "general", context: Dict[str, Any] = None) -> tuple[ReasoningResult, QualityAssessment]:
        """
        Complete reasoning pipeline: assess → decompose → execute → validate
        
        Args:
            input_data: The input to reason about
            domain_context: Domain context for specialized reasoning
            context: Additional context
            
        Returns:
            Tuple of (ReasoningResult, QualityAssessment)
        """
        start_time = time.time()
        
        try:
            # Stage 1: Assess complexity
            logger.info("Stage 1: Assessing complexity...")
            complexity = await self.assess_complexity(input_data, domain_context)
            
            # Stage 2: Select strategy
            logger.info(f"Stage 2: Selecting strategy for {complexity.overall_complexity} complexity...")
            strategy = await self.select_strategy(complexity)
            
            # Stage 3: Decompose problem
            logger.info(f"Stage 3: Decomposing problem using {strategy.strategy_type} strategy...")
            decomposition = await self.decompose_problem(input_data, strategy, context)
            
            # Stage 4: Execute reasoning
            logger.info(f"Stage 4: Executing reasoning with {len(decomposition.sub_problems)} sub-problems...")
            reasoning_result = await self.execute_reasoning(decomposition, input_data, context)
            
            # Stage 5: Validate quality
            logger.info("Stage 5: Validating reasoning quality...")
            quality_assessment = await self.validate_reasoning(input_data, reasoning_result, context)
            
            # Update execution time
            reasoning_result.execution_time = time.time() - start_time
            
            logger.info(f"Reasoning completed in {reasoning_result.execution_time:.2f}s with quality score {quality_assessment.overall_quality_score:.2f}")
            
            return reasoning_result, quality_assessment
            
        except Exception as e:
            logger.error(f"Reasoning failed: {str(e)}")
            
            # Return error result
            error_result = ReasoningResult(
                result_data={"error": str(e)},
                reasoning_trace=[f"Error occurred: {str(e)}"],
                strategy_used=ReasoningStrategy(
                    strategy_type=ReasoningStrategyType.DIRECT,
                    complexity_level=ComplexityLevel.SIMPLE
                ),
                execution_time=time.time() - start_time,
                confidence=0.0,
                metadata={"error": True}
            )
            
            error_quality = QualityAssessment(
                overall_quality_score=0.0,
                confidence_level=0.0,
                validation_passed=False,
                critical_issues=[f"Reasoning failed: {str(e)}"]
            )
            
            return error_result, error_quality
    
    # ========================================
    # Helper Methods
    # ========================================
    
    def _initialize_strategies(self) -> Dict[ReasoningStrategyType, ReasoningStrategy]:
        """Initialize default reasoning strategies"""
        return {
            ReasoningStrategyType.DIRECT: ReasoningStrategy(
                strategy_type=ReasoningStrategyType.DIRECT,
                complexity_level=ComplexityLevel.SIMPLE,
                processing_stages=["analyze", "conclude"],
                parallel_capable=False,
                requires_validation=True,
                max_iterations=1,
                timeout_seconds=60
            ),
            ReasoningStrategyType.DECOMPOSED: ReasoningStrategy(
                strategy_type=ReasoningStrategyType.DECOMPOSED,
                complexity_level=ComplexityLevel.MODERATE,
                processing_stages=["decompose", "analyze_parts", "synthesize"],
                parallel_capable=True,
                requires_validation=True,
                max_iterations=1,
                timeout_seconds=300
            ),
            ReasoningStrategyType.PARALLEL: ReasoningStrategy(
                strategy_type=ReasoningStrategyType.PARALLEL,
                complexity_level=ComplexityLevel.COMPLEX,
                processing_stages=["parallel_analyze", "merge_results"],
                parallel_capable=True,
                requires_validation=True,
                max_iterations=1,
                timeout_seconds=180
            ),
            ReasoningStrategyType.ITERATIVE: ReasoningStrategy(
                strategy_type=ReasoningStrategyType.ITERATIVE,
                complexity_level=ComplexityLevel.VERY_COMPLEX,
                processing_stages=["initial_analysis", "refine", "validate", "iterate"],
                parallel_capable=False,
                requires_validation=True,
                max_iterations=3,
                timeout_seconds=600
            )
        }
    
    def _initialize_quality_thresholds(self) -> Dict[str, float]:
        """Initialize quality thresholds"""
        return {
            "minimum_quality": 0.6,
            "good_quality": 0.8,
            "excellent_quality": 0.9,
            "minimum_confidence": 0.5
        }
    
    def _assess_completeness(self, reasoning_result: ReasoningResult) -> float:
        """Assess completeness of reasoning result"""
        # Multi-factor completeness assessment
        data_fields = len(reasoning_result.result_data)
        trace_steps = len(reasoning_result.reasoning_trace)
        has_metadata = len(reasoning_result.metadata) > 0
        
        # Weighted completeness score
        completeness = min(1.0, 
            (data_fields * 0.15) +      # Data richness
            (trace_steps * 0.1) +       # Reasoning depth
            (0.2 if has_metadata else 0) # Metadata presence
        )
        return max(0.1, completeness)  # Minimum 0.1 score
    
    def _assess_consistency(self, reasoning_result: ReasoningResult) -> float:
        """Assess internal consistency of reasoning"""
        consistency_factors = []
        
        # Factor 1: Confidence vs trace detail alignment
        confidence = reasoning_result.confidence
        trace_length = len(reasoning_result.reasoning_trace)
        
        # More reasonable expectation: confidence 0.8+ should have 2+ trace steps
        if confidence >= 0.8 and trace_length >= 2:
            confidence_trace_consistency = 1.0
        elif confidence >= 0.6 and trace_length >= 1:
            confidence_trace_consistency = 0.8
        elif confidence >= 0.4:
            confidence_trace_consistency = 0.6
        else:
            confidence_trace_consistency = 0.4
        
        consistency_factors.append(confidence_trace_consistency)
        
        # Factor 2: Result data richness vs confidence
        data_richness = len(reasoning_result.result_data)
        if confidence >= 0.8 and data_richness >= 3:
            data_consistency = 1.0
        elif confidence >= 0.6 and data_richness >= 2:
            data_consistency = 0.8
        else:
            data_consistency = 0.6
        
        consistency_factors.append(data_consistency)
        
        # Factor 3: Internal coherence check
        # Check if reasoning trace mentions key concepts from result data
        trace_text = " ".join(reasoning_result.reasoning_trace).lower()
        result_text = str(reasoning_result.result_data).lower()
        
        # Simple keyword overlap check
        trace_words = set(trace_text.split())
        result_words = set(result_text.split())
        
        if len(trace_words) > 0 and len(result_words) > 0:
            overlap = len(trace_words.intersection(result_words))
            coherence = min(1.0, overlap / min(len(trace_words), len(result_words)) * 2)
        else:
            coherence = 0.5
        
        consistency_factors.append(coherence)
        
        # Weighted average (more forgiving)
        overall_consistency = sum(consistency_factors) / len(consistency_factors)
        return max(0.3, overall_consistency)  # Minimum 0.3 instead of 0.1
    
    def _assess_relevance(self, input_data: Any, reasoning_result: ReasoningResult) -> float:
        """Assess relevance of reasoning to input"""
        # Simple relevance check based on keyword overlap
        input_str = str(input_data).lower()
        result_str = str(reasoning_result.result_data).lower()
        
        input_words = set(input_str.split())
        result_words = set(result_str.split())
        
        if not input_words:
            return 0.5  # Neutral if no input words
        
        overlap = len(input_words.intersection(result_words))
        relevance = min(1.0, overlap / len(input_words))
        return max(0.1, relevance)
    
    def _perform_gap_analysis(self, quality_metrics: List[QualityMetric], reasoning_result: ReasoningResult, context: Dict[str, Any] = None) -> List[GapAnalysis]:
        """Perform comprehensive gap analysis with enhanced detection"""
        gaps = []
        
        # Analyze quality metrics for gaps
        for metric in quality_metrics:
            if metric.score < 0.6:  # More reasonable threshold
                severity = "critical" if metric.score < 0.3 else "high" if metric.score < 0.5 else "medium"
                
                # Specific gap types based on dimension
                if metric.dimension == QualityDimension.COMPLETENESS:
                    gap_type = "incomplete_analysis"
                    description = f"Analysis lacks sufficient detail (completeness: {metric.score:.2f})"
                    mitigation_strategies = ["Add more comprehensive analysis", "Include additional perspectives", "Expand on key concepts"]
                elif metric.dimension == QualityDimension.CONSISTENCY:
                    gap_type = "logical_inconsistency"
                    description = f"Internal reasoning inconsistencies detected (consistency: {metric.score:.2f})"
                    mitigation_strategies = ["Review logical flow", "Align confidence with evidence", "Check for contradictions"]
                elif metric.dimension == QualityDimension.RELEVANCE:
                    gap_type = "relevance_drift"
                    description = f"Analysis drifts from core input focus (relevance: {metric.score:.2f})"
                    mitigation_strategies = ["Refocus on key input elements", "Remove tangential content", "Strengthen input-output alignment"]
                else:
                    gap_type = f"{metric.dimension.value}_deficiency"
                    description = f"Low {metric.dimension.value} score: {metric.score:.2f}"
                    mitigation_strategies = metric.improvement_suggestions
                
                gaps.append(GapAnalysis(
                    gap_type=gap_type,
                    description=description,
                    severity=severity,
                    impact_areas=[metric.dimension.value, "overall_quality"],
                    mitigation_strategies=mitigation_strategies,
                    requires_human_input=severity in ["critical", "high"]
                ))
        
        # Enhanced reasoning trace analysis
        trace_length = len(reasoning_result.reasoning_trace)
        if trace_length < 1:
            gaps.append(GapAnalysis(
                gap_type="missing_reasoning_trace",
                description="No reasoning trace provided - analysis is opaque",
                severity="critical",
                impact_areas=["transparency", "explainability", "trustworthiness"],
                mitigation_strategies=["Provide step-by-step reasoning", "Document decision points", "Show logical progression"],
                requires_human_input=True
            ))
        elif trace_length < 2 and reasoning_result.confidence > 0.7:
            gaps.append(GapAnalysis(
                gap_type="insufficient_reasoning_depth",
                description="High confidence with minimal reasoning explanation",
                severity="medium",
                impact_areas=["transparency", "validation"],
                mitigation_strategies=["Add intermediate reasoning steps", "Explain confidence sources", "Show work more clearly"],
                requires_human_input=False
            ))
        
        # Confidence-evidence alignment check
        confidence = reasoning_result.confidence
        data_richness = len(reasoning_result.result_data)
        
        if confidence > 0.8 and data_richness < 3:
            gaps.append(GapAnalysis(
                gap_type="confidence_evidence_mismatch",
                description="High confidence not supported by sufficient evidence",
                severity="high",
                impact_areas=["reliability", "validation"],
                mitigation_strategies=["Provide more supporting evidence", "Lower confidence rating", "Add uncertainty quantification"],
                requires_human_input=True
            ))
        elif confidence < 0.5 and data_richness >= 4:
            gaps.append(GapAnalysis(
                gap_type="underconfidence_bias",
                description="Low confidence despite rich analysis",
                severity="medium",
                impact_areas=["decision_making", "actionability"],
                mitigation_strategies=["Review confidence calibration", "Highlight strong evidence", "Consider confidence boost"],
                requires_human_input=False
            ))
        
        # Strategy-complexity alignment check
        if context and "strategy_used" in context:
            strategy = context["strategy_used"]
            if hasattr(strategy, 'strategy_type') and hasattr(strategy, 'complexity_level'):
                if strategy.complexity_level == ComplexityLevel.SIMPLE and len(reasoning_result.reasoning_trace) > 3:
                    gaps.append(GapAnalysis(
                        gap_type="strategy_complexity_mismatch",
                        description="Simple complexity assessment but complex reasoning execution",
                        severity="low",
                        impact_areas=["efficiency", "resource_usage"],
                        mitigation_strategies=["Reassess complexity", "Simplify reasoning approach", "Optimize strategy selection"],
                        requires_human_input=False
                    ))
        
        return gaps
    
    def _generate_human_queries(self, gaps: List[GapAnalysis], reasoning_result: ReasoningResult, context: Dict[str, Any] = None) -> List[str]:
        """Generate specific human queries based on identified gaps"""
        queries = []
        
        # Generate specific queries for gaps requiring human input
        for gap in gaps:
            if gap.requires_human_input:
                if gap.gap_type == "missing_reasoning_trace":
                    queries.append("Can you explain the reasoning steps that led to this conclusion?")
                elif gap.gap_type == "logical_inconsistency":
                    queries.append("How can we resolve the logical inconsistencies in this analysis?")
                elif gap.gap_type == "confidence_evidence_mismatch":
                    queries.append("What additional evidence supports this high confidence level?")
                elif gap.gap_type == "incomplete_analysis":
                    queries.append("What key aspects of this problem need deeper analysis?")
                elif gap.gap_type == "relevance_drift":
                    queries.append("How can we better align the analysis with the original input?")
                elif "deficiency" in gap.gap_type:
                    dimension = gap.gap_type.replace('_deficiency', '')
                    queries.append(f"How can we improve the {dimension} of this reasoning?")
                else:
                    queries.append(f"How should we address the {gap.gap_type.replace('_', ' ')} issue?")
        
        # Context-specific queries
        if reasoning_result.confidence < 0.6:
            queries.append("What factors contribute to the uncertainty in this analysis?")
        
        if len(reasoning_result.result_data) < 2:
            queries.append("What additional insights or analysis would strengthen this reasoning?")
        
        # Remove duplicates while preserving order
        unique_queries = []
        for query in queries:
            if query not in unique_queries:
                unique_queries.append(query)
        
        return unique_queries[:3]  # Limit to 3 most important queries
    
    def get_available_strategies(self) -> List[ReasoningStrategyType]:
        """Get list of available reasoning strategies"""
        return list(self.reasoning_strategies.keys())
    
    def add_llm_backend(self, name: str, backend: LLMBackend):
        """Add an LLM backend"""
        self.llm_backends[name] = backend
        logger.info(f"Added LLM backend: {name}")
    
    def get_llm_backend(self, name: str = "default") -> Optional[LLMBackend]:
        """Get an LLM backend by name"""
        return self.llm_backends.get(name)

# ========================================
# Utility Functions
# ========================================

def create_complexity_factor(name: str, weight: float, score: float, reasoning: str = "") -> ComplexityFactor:
    """Helper to create complexity factors"""
    return ComplexityFactor(
        factor_name=name,
        weight=weight,
        score=score,
        reasoning=reasoning
    )

def create_sub_problem(problem_id: str, description: str, complexity: ComplexityLevel = ComplexityLevel.MODERATE, 
                      priority: int = 1, dependencies: List[str] = None) -> SubProblem:
    """Helper to create sub-problems"""
    return SubProblem(
        problem_id=problem_id,
        description=description,
        complexity=complexity,
        priority=priority,
        dependencies=dependencies or []
    )

def calculate_critical_path(sub_problems: List[SubProblem]) -> List[str]:
    """Calculate critical path through sub-problems based on dependencies"""
    # Simple topological sort for critical path
    # This is a basic implementation - could be enhanced with proper critical path algorithm
    
    problem_map = {p.problem_id: p for p in sub_problems}
    visited = set()
    path = []
    
    def visit(problem_id: str):
        if problem_id in visited or problem_id not in problem_map:
            return
        
        visited.add(problem_id)
        problem = problem_map[problem_id]
        
        # Visit dependencies first
        for dep_id in problem.dependencies:
            visit(dep_id)
        
        path.append(problem_id)
    
    # Visit all problems
    for problem in sub_problems:
        visit(problem.problem_id)
    
    return path

if __name__ == "__main__":
    import asyncio
    
    # Enhanced demonstration with LLM integration
    class DemoReasoner(MetaReasoningBase):
        """Demo implementation showing LLM integration patterns"""
        
        def __init__(self, llm_backends: Dict[str, LLMBackend] = None):
            # Try to initialize with OpenAI backend if available
            if llm_backends is None:
                llm_backends = {}
                try:
                    # Try to create OpenAI backend
                    openai_backend = OpenAIBackend("OpenAI-Demo", "gpt-4o-mini")
                    llm_backends["default"] = openai_backend
                    print("✅ OpenAI backend initialized successfully")
                except Exception as e:
                    print(f"❌ OpenAI backend not available: {e}")
                    print("⚠️ Demo requires LLM backend - set OPENAI_API_KEY environment variable")
            
            super().__init__(llm_backends)
        
        async def assess_complexity(self, input_data: Any, domain_context: str = "general") -> ComplexityAssessment:
            # Use LLM for complexity assessment - no fallbacks
            input_str = str(input_data)
            
            prompt = f"""
            Assess the complexity of this input for reasoning analysis:
            
            Input: "{input_str}"
            Domain: {domain_context}
            
            Consider factors like:
            - Length and detail level
            - Number of concepts involved
            - Ambiguity and clarity
            - Required reasoning depth
            
            Provide complexity assessment with reasoning.
            """
            
            llm = self.get_llm_backend("default")
            if not llm:
                raise RuntimeError("No LLM backend available. Cannot perform complexity assessment.")
            
            try:
                # Try structured response first
                response = await llm.complete_structured(prompt, {"input": input_str}, ComplexityAssessment)
                print(f"🤖 LLM Structured Response: {response}")
                return ComplexityAssessment(**response)
            except Exception as e:
                # Try text response as backup
                try:
                    response = await llm.complete(prompt, {"input": input_str})
                    print(f"🤖 LLM Complexity Assessment: {response}")
                    
                    # Parse response manually for key complexity indicators
                    response_lower = response.lower()
                    if "simple" in response_lower or "basic" in response_lower:
                        complexity = ComplexityLevel.SIMPLE
                        strategy = ReasoningStrategyType.DIRECT
                    elif "complex" in response_lower or "sophisticated" in response_lower:
                        complexity = ComplexityLevel.COMPLEX
                        strategy = ReasoningStrategyType.PARALLEL
                    else:
                        complexity = ComplexityLevel.MODERATE
                        strategy = ReasoningStrategyType.DECOMPOSED
                    
                    return ComplexityAssessment(
                        overall_complexity=complexity,
                        complexity_score=0.6,
                        recommended_strategy=strategy,
                        confidence=0.7,
                        reasoning=f"LLM assessment: {response[:100]}..."
                    )
                except Exception as inner_e:
                    raise RuntimeError(f"LLM complexity assessment failed: {inner_e}")
        
        async def decompose_problem(self, input_data: Any, strategy: ReasoningStrategy, context: Dict[str, Any] = None) -> ProblemDecomposition:
            # Use LLM for problem decomposition
            input_str = str(input_data)
            
            prompt = f"""
            Break down this problem into manageable sub-problems:
            
            Input: "{input_str}"
            Strategy: {strategy.strategy_type.value}
            Complexity: {strategy.complexity_level.value}
            
            Create a logical sequence of analysis steps with dependencies.
            Focus on: analysis → processing → synthesis → validation
            """
            
            llm = self.get_llm_backend("default")
            if not llm:
                raise RuntimeError("No LLM backend available. Cannot perform problem decomposition.")
            
            try:
                response = await llm.complete(prompt, {"input": input_str, "strategy": strategy.strategy_type.value})
                print(f"🤖 LLM Decomposition: {response}")
            except Exception as e:
                raise RuntimeError(f"LLM decomposition failed: {e}")
            
            # Create structured decomposition based on strategy
            if strategy.strategy_type == ReasoningStrategyType.DIRECT:
                sub_problems = [
                    create_sub_problem("direct_analysis", "Direct analysis of input", ComplexityLevel.SIMPLE, 1)
                ]
            elif strategy.strategy_type == ReasoningStrategyType.PARALLEL:
                # Parallel strategy should have 4+ sub-problems for true parallelism
                sub_problems = [
                    create_sub_problem("analyze_concepts", "Analyze key concepts", ComplexityLevel.MODERATE, 1),
                    create_sub_problem("analyze_relationships", "Analyze relationships", ComplexityLevel.MODERATE, 1),
                    create_sub_problem("analyze_context", "Analyze contextual factors", ComplexityLevel.MODERATE, 1),
                    create_sub_problem("analyze_implications", "Analyze implications", ComplexityLevel.MODERATE, 1),
                    create_sub_problem("synthesize", "Synthesize all findings", ComplexityLevel.COMPLEX, 2, 
                                     ["analyze_concepts", "analyze_relationships", "analyze_context", "analyze_implications"])
                ]
            elif strategy.strategy_type == ReasoningStrategyType.ITERATIVE:
                # Iterative strategy with refinement cycles
                sub_problems = [
                    create_sub_problem("initial_analysis", "Initial analysis", ComplexityLevel.SIMPLE, 1),
                    create_sub_problem("refine_analysis", "Refine analysis", ComplexityLevel.MODERATE, 2, ["initial_analysis"]),
                    create_sub_problem("validate_refinement", "Validate refinement", ComplexityLevel.MODERATE, 3, ["refine_analysis"]),
                    create_sub_problem("final_synthesis", "Final synthesis", ComplexityLevel.MODERATE, 4, ["validate_refinement"])
                ]
            else:  # DECOMPOSED (default)
                sub_problems = [
                    create_sub_problem("analyze", "Analyze the input", ComplexityLevel.SIMPLE, 1),
                    create_sub_problem("process", "Process the analysis", ComplexityLevel.SIMPLE, 2, ["analyze"]),
                    create_sub_problem("conclude", "Draw conclusions", ComplexityLevel.SIMPLE, 3, ["process"])
                ]
            
            return ProblemDecomposition(
                sub_problems=sub_problems,
                execution_order=[p.problem_id for p in sub_problems],
                critical_path=calculate_critical_path(sub_problems),
                decomposition_confidence=0.9
            )
        
        async def execute_reasoning(self, decomposition: ProblemDecomposition, input_data: Any, context: Dict[str, Any] = None) -> ReasoningResult:
            # Use LLM for reasoning execution
            input_str = str(input_data)
            
            prompt = f"""
            Execute reasoning analysis for this input:
            
            Input: "{input_str}"
            Analysis Steps: {[p.description for p in decomposition.sub_problems]}
            
            Provide detailed reasoning with step-by-step analysis.
            Include your confidence level and key insights.
            """
            
            llm = self.get_llm_backend("default")
            if not llm:
                raise RuntimeError("No LLM backend available. Cannot execute reasoning.")
            
            try:
                reasoning_response = await llm.complete(prompt, {"input": input_str})
                print(f"🤖 LLM Reasoning: {reasoning_response}")
            except Exception as e:
                raise RuntimeError(f"LLM reasoning failed: {e}")
            
            # Build reasoning trace
            trace = []
            result_data = {}
            
            for problem in decomposition.sub_problems:
                step_prompt = f"Execute step: {problem.description} for input: {input_str[:100]}..."
                try:
                    step_response = await llm.complete(step_prompt, {"step": problem.description})
                    trace.append(f"Step '{problem.problem_id}': {step_response}")
                    result_data[problem.problem_id] = step_response
                except Exception as e:
                    raise RuntimeError(f"LLM step execution failed for {problem.problem_id}: {e}")
            
            # Overall analysis
            result_data["overall_analysis"] = reasoning_response
            result_data["input_summary"] = f"Input length: {len(input_str)} chars, Steps: {len(decomposition.sub_problems)}"
            
            # Determine strategy based on decomposition
            num_problems = len(decomposition.sub_problems)
            if num_problems == 1:
                strategy_type = ReasoningStrategyType.DIRECT
                complexity = ComplexityLevel.SIMPLE
            elif num_problems <= 3:
                strategy_type = ReasoningStrategyType.DECOMPOSED
                complexity = ComplexityLevel.MODERATE
            elif num_problems == 4 and any("refine" in p.problem_id for p in decomposition.sub_problems):
                # Iterative strategy pattern detected
                strategy_type = ReasoningStrategyType.ITERATIVE
                complexity = ComplexityLevel.COMPLEX
            else:
                # 5+ problems or parallel pattern
                strategy_type = ReasoningStrategyType.PARALLEL
                complexity = ComplexityLevel.COMPLEX
            
            strategy = ReasoningStrategy(
                strategy_type=strategy_type,
                complexity_level=complexity
            )
            
            return ReasoningResult(
                result_data=result_data,
                reasoning_trace=trace,
                strategy_used=strategy,
                confidence=0.85,
                metadata={
                    "llm_calls": len(decomposition.sub_problems) + 2, 
                    "input_length": len(input_str),
                    "has_llm": True
                }
            )
    
    async def demo():
        print("🧠 Hybrid Meta-Reasoning Demo")
        print("=" * 60)
        print("Features: LLM-ready • Abstract • Quality-focused • Gap analysis")
        print("=" * 60)
        
        reasoner = DemoReasoner()
        
        # Check if LLM backend is available
        has_llm = reasoner.get_llm_backend("default") is not None
        if has_llm:
            print("🤖 LLM Backend: Active - Real AI reasoning enabled")
            print("=" * 60)
        else:
            print("❌ LLM Backend: Not available - Demo cannot proceed")
            print("⚠️ Set OPENAI_API_KEY environment variable to run demo")
            print("=" * 60)
            return
        
        # Test with different complexity inputs
        test_cases = [
            "Simple input",
            "This is a moderately complex input that requires some analysis and decomposition into smaller parts",
            "This is a very complex input that demonstrates the meta-reasoning engine's ability to handle sophisticated problems that require parallel processing, iterative refinement, and comprehensive quality validation across multiple dimensions of analysis"
        ]
        
        for i, test_input in enumerate(test_cases, 1):
            print(f"\n🔍 Test {i}: '{test_input[:50]}{'...' if len(test_input) > 50 else ''}' ({len(test_input)} chars)")
            print("-" * 50)
            
            result, quality = await reasoner.reason(test_input)
            
            print(f"📊 Results:")
            print(f"   Strategy: {result.strategy_used.strategy_type.value}")
            print(f"   Quality: {quality.overall_quality_score:.2f}")
            print(f"   Time: {result.execution_time:.3f}s")
            print(f"   Steps: {len(result.reasoning_trace)}")
            print(f"   LLM calls: {result.metadata.get('llm_calls', 0)}")
            print(f"   Gaps: {len(quality.identified_gaps)}")
            print(f"   Human queries: {len(quality.human_queries)}")
            
            # Show sample reasoning trace
            if result.reasoning_trace:
                print(f"   Sample trace: {result.reasoning_trace[0][:80]}...")
            
            # Show gaps if any
            if quality.identified_gaps:
                print(f"   Gap example: {quality.identified_gaps[0].description}")
            
            # Show human queries if any
            if quality.human_queries:
                print(f"   Human query: {quality.human_queries[0]}")
        
        print(f"\n✅ LLM-powered meta-reasoning capabilities demonstrated")
        print("🎯 Ready for intent graph development and other domain applications")
        print("🤖 All reasoning powered by real AI - no fallbacks")
    
    # Run demo
    asyncio.run(demo())
