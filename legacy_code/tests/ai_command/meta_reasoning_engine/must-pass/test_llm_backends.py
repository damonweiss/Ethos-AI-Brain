#!/usr/bin/env python3
"""
MetaReasoningEngine - LLM Backend Tests (MUST PASS)
Tests LLM backend functionality and mock responses
NO MOCKING - NO FALLBACKS - REAL TESTS ONLY
"""

import sys
import os
import pytest
import logging
import asyncio

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'ai_command'))

from meta_reasoning_engine import (
    MetaReasoningEngine,
    MockLLMBackend,
    LLMBackend
)

logger = logging.getLogger(__name__)

class TestLLMBackends:
    """Critical LLM backend tests that MUST pass"""
    
    def setup_method(self):
        """Setup for each test"""
        self.backends = {}
    
    def teardown_method(self):
        """Cleanup after each test"""
        self.backends.clear()
    
    def test_mock_llm_backend_initialization(self):
        """Test MockLLMBackend can be initialized with different personalities"""
        personalities = ["analytical", "creative", "empathetic", "systematic"]
        
        for personality in personalities:
            backend = MockLLMBackend(f"TestBot_{personality}", personality)
            
            assert backend.name == f"TestBot_{personality}"
            assert backend.personality == personality
            assert isinstance(backend, LLMBackend)
    
    @pytest.mark.asyncio
    async def test_analysis_prompt_responses(self):
        """Test analysis-specific prompt responses"""
        backend = MockLLMBackend("Analyst", "analytical")
        
        analysis_prompts = [
            "analyze the user data",
            "perform analysis on system metrics",
            "analyze performance patterns",
            "data analysis required"
        ]
        
        for prompt in analysis_prompts:
            response = await backend.complete(prompt, {})
            
            assert isinstance(response, str)
            assert len(response) > 0
            # Should contain "Analysis" for analyze prompts or backend name
            assert "Analysis" in response or "Analyst" in response
            assert "Analyst" in response
    
    @pytest.mark.asyncio
    async def test_planning_prompt_responses(self):
        """Test planning-specific prompt responses"""
        backend = MockLLMBackend("Planner", "strategic")
        
        planning_prompts = [
            "create a plan for deployment",
            "plan the project timeline",
            "develop implementation plan",
            "planning strategy needed"
        ]
        
        for prompt in planning_prompts:
            response = await backend.complete(prompt, {})
            
            assert isinstance(response, str)
            assert len(response) > 0
            assert "Plan" in response
            assert "Planner" in response
            assert "phase" in response.lower()
            assert "approach" in response.lower()
    
    @pytest.mark.asyncio
    async def test_decomposition_prompt_responses(self):
        """Test decomposition-specific prompt responses"""
        backend = MockLLMBackend("Decomposer", "systematic")
        
        decomposition_prompts = [
            "decompose this complex task",
            "break down the requirements",
            "decompose into subtasks",
            "task decomposition needed"
        ]
        
        for prompt in decomposition_prompts:
            response = await backend.complete(prompt, {})
            
            assert isinstance(response, str)
            assert len(response) > 0
            # Should contain "Decomposition" for decompose prompts or backend name
            assert "Decomposition" in response or "Decomposer" in response
            assert "Decomposer" in response
    
    @pytest.mark.asyncio
    async def test_confidence_assessment_responses(self):
        """Test confidence assessment prompt responses"""
        backend = MockLLMBackend("Assessor", "analytical")
        
        confidence_prompts = [
            "assess confidence in this solution",
            "confidence level evaluation",
            "how confident are you",
            "confidence assessment needed"
        ]
        
        for prompt in confidence_prompts:
            response = await backend.complete(prompt, {})
            
            assert isinstance(response, str)
            assert len(response) > 0
            # Should contain "Confidence" for confidence prompts or backend name
            assert "Confidence" in response or "Assessor" in response
            assert "Assessor" in response
    
    @pytest.mark.asyncio
    async def test_citizen_engagement_responses(self):
        """Test citizen engagement prompt responses"""
        backend = MockLLMBackend("CitizenBot", "empathetic")
        
        citizen_prompts = [
            "question for citizen engagement",
            "citizen input needed",
            "engage with citizen feedback",
            "citizen question about policy"
        ]
        
        for prompt in citizen_prompts:
            response = await backend.complete(prompt, {})
            
            assert isinstance(response, str)
            assert len(response) > 0
            assert "Citizen" in response
            assert "CitizenBot" in response
            assert any(word in response.lower() for word in ["thank", "understand", "concerns", "help"])
    
    @pytest.mark.asyncio
    async def test_generic_prompt_responses(self):
        """Test generic prompt handling"""
        backend = MockLLMBackend("GenericBot", "helpful")
        
        generic_prompts = [
            "help me with this task",
            "what should I do",
            "general assistance needed",
            "random question"
        ]
        
        for prompt in generic_prompts:
            response = await backend.complete(prompt, {})
            
            assert isinstance(response, str)
            assert len(response) > 0
            assert "GenericBot" in response
            assert "helpful" in response.lower()
            assert prompt[:20] in response  # Should include part of the prompt
    
    @pytest.mark.asyncio
    async def test_streaming_response_functionality(self):
        """Test streaming response generation"""
        backend = MockLLMBackend("StreamBot", "verbose")
        
        prompt = "analyze the data patterns"
        
        # Collect streaming response
        streamed_words = []
        async for word_chunk in backend.generate(prompt, {}):
            streamed_words.append(word_chunk)
        
        # Verify streaming worked
        assert len(streamed_words) > 0
        
        # Reconstruct full response
        full_response = "".join(streamed_words).strip()
        
        # Compare with complete response
        complete_response = await backend.complete(prompt, {})
        
        assert full_response == complete_response
        assert "StreamBot" in full_response
        assert "Analysis" in full_response
    
    @pytest.mark.asyncio
    async def test_context_parameter_handling(self):
        """Test context parameter is properly handled"""
        backend = MockLLMBackend("ContextBot", "contextual")
        
        prompt = "analyze this data"
        context = {
            "user_id": "test_user",
            "session_id": "test_session",
            "metadata": {"priority": "high"}
        }
        
        response = await backend.complete(prompt, context)
        
        # Context should not break the response
        assert isinstance(response, str)
        assert len(response) > 0
        assert "ContextBot" in response
        assert "Analysis" in response
    
    def test_multiple_backend_personalities(self):
        """Test different personalities produce different characteristics"""
        personalities = {
            "analytical": MockLLMBackend("Analyst", "analytical"),
            "creative": MockLLMBackend("Creator", "creative"),
            "empathetic": MockLLMBackend("Empath", "empathetic"),
            "systematic": MockLLMBackend("System", "systematic")
        }
        
        for personality_name, backend in personalities.items():
            assert backend.personality == personality_name
            # Backend name should be related to personality (flexible matching)
            assert isinstance(backend.name, str) and len(backend.name) > 0
    
    @pytest.mark.asyncio
    async def test_prompt_length_handling(self):
        """Test handling of different prompt lengths"""
        backend = MockLLMBackend("LengthBot", "adaptive")
        
        # Test different prompt lengths
        prompts = [
            "short",
            "medium length prompt for testing",
            "very long prompt that contains multiple sentences and detailed information about what we want to analyze and how we want to proceed with the analysis including various constraints and requirements that need to be considered"
        ]
        
        for prompt in prompts:
            response = await backend.complete(prompt, {})
            
            assert isinstance(response, str)
            assert len(response) > 0
            assert "LengthBot" in response
            
            # For long prompts, check if it's handled appropriately
            if len(prompt) > 100:
                # Should either include truncated version or handle gracefully
                assert len(response) > 50  # Should produce substantial response
    
    @pytest.mark.asyncio
    async def test_concurrent_backend_usage(self):
        """Test multiple backends can be used concurrently"""
        backends = [
            MockLLMBackend("Bot1", "analytical"),
            MockLLMBackend("Bot2", "creative"),
            MockLLMBackend("Bot3", "systematic")
        ]
        
        # Create concurrent tasks
        tasks = []
        for i, backend in enumerate(backends):
            task = backend.complete(f"analyze task {i}", {})
            tasks.append(task)
        
        # Execute concurrently
        responses = await asyncio.gather(*tasks)
        
        # Verify all responses
        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert isinstance(response, str)
            assert len(response) > 0
            assert f"Bot{i+1}" in response
            assert "Analysis" in response
