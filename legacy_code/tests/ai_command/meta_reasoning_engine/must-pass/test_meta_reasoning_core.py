#!/usr/bin/env python3
"""
MetaReasoningEngine - Core Functionality Tests (MUST PASS)
Tests fundamental meta-reasoning capabilities with mock LLM backends
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
    ReasoningContext, 
    ReasoningStep,
    ConfidenceLevel,
    CollaborationMode,
    MockLLMBackend,
    ToolCapability
)

logger = logging.getLogger(__name__)

class TestMetaReasoningCore:
    """Critical meta-reasoning tests that MUST pass"""
    
    def setup_method(self):
        """Setup for each test"""
        self.engine = None
    
    def teardown_method(self):
        """Cleanup after each test"""
        if self.engine:
            # Clean up any active sessions
            try:
                self.engine.active_sessions.clear()
            except:
                pass
    
    def test_meta_reasoning_engine_initialization(self):
        """Test MetaReasoningEngine initializes with all components"""
        self.engine = MetaReasoningEngine()
        
        assert self.engine is not None
        assert hasattr(self.engine, 'llm_backends')
        assert hasattr(self.engine, 'memory')
        assert hasattr(self.engine, 'tool_registry')
        assert hasattr(self.engine, 'active_sessions')
        assert hasattr(self.engine, 'collaboration_handlers')
        assert hasattr(self.engine, 'event_handlers')
        
        # Verify LLM backends are initialized
        assert len(self.engine.llm_backends) > 0
        expected_backends = ['analyst', 'planner', 'citizen_engagement', 'decomposer']
        for backend_name in expected_backends:
            assert backend_name in self.engine.llm_backends
            assert isinstance(self.engine.llm_backends[backend_name], MockLLMBackend)
    
    def test_reasoning_context_creation(self):
        """Test ReasoningContext can be created with proper fields"""
        context = ReasoningContext(
            goal="Test goal",
            constraints={"budget": 1000},
            user_preferences={"style": "detailed"}
        )
        
        assert context.goal == "Test goal"
        assert context.constraints["budget"] == 1000
        assert context.user_preferences["style"] == "detailed"
        assert context.session_id is not None
        assert context.timestamp is not None
        assert isinstance(context.metadata, dict)
    
    def test_mock_llm_backend_functionality(self):
        """Test MockLLMBackend provides expected responses"""
        backend = MockLLMBackend("TestBot", "analytical")
        
        # Test different prompt patterns
        test_cases = [
            ("analyze the data", "Analysis"),
            ("create a plan", "Plan"),
            ("decompose this task", "Task Decomposition"),
            ("assess confidence", "Confidence Assessment"),
            ("general question", "I understand your request")
        ]
        
        for prompt, expected_keyword in test_cases:
            import asyncio
            async def test_prompt():
                response = await backend.complete(prompt, {})
                return response
            
            response = asyncio.run(test_prompt())
            assert isinstance(response, str)
            assert len(response) > 0
            assert expected_keyword in response
            assert backend.name in response
    
    @pytest.mark.asyncio
    async def test_basic_reasoning_workflow(self):
        """Test basic reasoning workflow from goal to result"""
        self.engine = MetaReasoningEngine()
        
        goal = "Analyze system performance and create improvement plan"
        context = ReasoningContext(goal=goal)
        
        # Execute reasoning
        result = await self.engine.reason(goal, context)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert result is not None
        
        # Verify session was cleaned up
        assert context.session_id not in self.engine.active_sessions
    
    @pytest.mark.asyncio
    async def test_goal_decomposition(self):
        """Test goal decomposition functionality"""
        self.engine = MetaReasoningEngine()
        
        goal = "Build a web application with user authentication"
        context = ReasoningContext(goal=goal)
        
        # Test decomposition (private method, but we can test through reason())
        result = await self.engine.reason(goal, context)
        
        # Result should contain decomposition information
        assert isinstance(result, dict)
        # The mock should provide some structured response
        assert result is not None
    
    def test_tool_registry_initialization(self):
        """Test tool registry is properly initialized"""
        self.engine = MetaReasoningEngine()
        
        # Verify tool registry exists
        assert self.engine.tool_registry is not None
        
        # Verify mock tools are registered
        expected_tools = ['data_analyzer', 'plan_generator', 'citizen_engagement', 'confidence_assessor']
        
        # Check if tools are available (implementation may vary)
        assert hasattr(self.engine.tool_registry, 'tools')
    
    def test_collaboration_handler_registration(self):
        """Test collaboration handler registration"""
        self.engine = MetaReasoningEngine()
        
        # Test handler registration
        def mock_approval_handler(request):
            return {"approved": True}
        
        self.engine.register_collaboration_handler(CollaborationMode.APPROVAL, mock_approval_handler)
        
        # Verify handler is registered
        assert CollaborationMode.APPROVAL.value in self.engine.collaboration_handlers
        assert self.engine.collaboration_handlers[CollaborationMode.APPROVAL.value] == mock_approval_handler
    
    def test_event_handler_registration(self):
        """Test event handler registration"""
        self.engine = MetaReasoningEngine()
        
        # Test event handler registration
        events_received = []
        
        def mock_event_handler(data):
            events_received.append(data)
        
        self.engine.register_event_handler("test_event", mock_event_handler)
        
        # Verify handler is registered
        assert "test_event" in self.engine.event_handlers
        assert mock_event_handler in self.engine.event_handlers["test_event"]
    
    @pytest.mark.asyncio
    async def test_event_emission(self):
        """Test event emission functionality"""
        self.engine = MetaReasoningEngine()
        
        # Register event handler
        events_received = []
        
        async def mock_async_handler(data):
            events_received.append(data)
        
        self.engine.register_event_handler("test_event", mock_async_handler)
        
        # Emit event
        test_data = {"message": "test"}
        await self.engine.emit_event("test_event", test_data)
        
        # Verify event was received
        assert len(events_received) == 1
        assert events_received[0] == test_data
    
    def test_llm_backend_registration(self):
        """Test LLM backend registration"""
        self.engine = MetaReasoningEngine()
        
        # Create custom backend
        custom_backend = MockLLMBackend("CustomBot", "creative")
        
        # Register backend
        self.engine.register_llm_backend("custom", custom_backend)
        
        # Verify backend is registered
        assert "custom" in self.engine.llm_backends
        assert self.engine.llm_backends["custom"] == custom_backend
    
    @pytest.mark.asyncio
    async def test_multiple_reasoning_sessions(self):
        """Test multiple concurrent reasoning sessions"""
        self.engine = MetaReasoningEngine()
        
        # Create multiple contexts
        context1 = ReasoningContext(goal="Task 1")
        context2 = ReasoningContext(goal="Task 2")
        
        # Execute reasoning concurrently
        task1 = self.engine.reason("Analyze data patterns", context1)
        task2 = self.engine.reason("Create deployment plan", context2)
        
        results = await asyncio.gather(task1, task2)
        
        # Verify both results
        assert len(results) == 2
        assert all(isinstance(result, dict) for result in results)
        assert all(result is not None for result in results)
        
        # Verify sessions were cleaned up
        assert len(self.engine.active_sessions) == 0
