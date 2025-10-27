#!/usr/bin/env python3
"""
MetaReasoningEngine - Reasoning Workflows Tests (NICE TO PASS)
Tests advanced reasoning workflows and patterns
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
    ConfidenceLevel,
    CollaborationMode
)

logger = logging.getLogger(__name__)

class TestReasoningWorkflows:
    """Advanced reasoning workflow tests"""
    
    def setup_method(self):
        """Setup for each test"""
        self.engine = MetaReasoningEngine()
    
    def teardown_method(self):
        """Cleanup after each test"""
        if self.engine:
            self.engine.active_sessions.clear()
    
    @pytest.mark.asyncio
    async def test_complex_goal_reasoning(self):
        """Test reasoning with complex, multi-faceted goals"""
        complex_goals = [
            "Design and implement a secure, scalable web application with user authentication, real-time features, and comprehensive testing",
            "Analyze customer behavior patterns, identify improvement opportunities, and create actionable recommendations",
            "Develop a machine learning pipeline for fraud detection with explainable AI and regulatory compliance"
        ]
        
        for goal in complex_goals:
            context = ReasoningContext(
                goal=goal,
                constraints={"timeline": "3 months", "budget": 50000},
                user_preferences={"methodology": "agile", "quality": "high"}
            )
            
            result = await self.engine.reason(goal, context)
            
            assert isinstance(result, dict)
            assert result is not None
            # Complex goals should produce substantial responses
            assert len(str(result)) > 100
    
    @pytest.mark.asyncio
    async def test_reasoning_with_constraints(self):
        """Test reasoning behavior with various constraints"""
        goal = "Build a mobile application"
        
        constraint_scenarios = [
            {"budget": 10000, "timeline": "1 month"},
            {"budget": 100000, "timeline": "6 months"},
            {"team_size": 2, "technology": "React Native"},
            {"compliance": "GDPR", "security": "high"}
        ]
        
        for constraints in constraint_scenarios:
            context = ReasoningContext(goal=goal, constraints=constraints)
            result = await self.engine.reason(goal, context)
            
            assert isinstance(result, dict)
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_reasoning_with_user_preferences(self):
        """Test reasoning adaptation to user preferences"""
        goal = "Create a data analysis solution"
        
        preference_scenarios = [
            {"style": "detailed", "format": "technical"},
            {"style": "summary", "format": "executive"},
            {"approach": "conservative", "risk_tolerance": "low"},
            {"approach": "innovative", "risk_tolerance": "high"}
        ]
        
        for preferences in preference_scenarios:
            context = ReasoningContext(goal=goal, user_preferences=preferences)
            result = await self.engine.reason(goal, context)
            
            assert isinstance(result, dict)
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_iterative_reasoning_sessions(self):
        """Test multiple reasoning sessions building on each other"""
        base_goal = "Develop a customer management system"
        
        # First iteration - high level
        context1 = ReasoningContext(goal=base_goal)
        result1 = await self.engine.reason(base_goal, context1)
        
        # Second iteration - with results from first
        refined_goal = "Refine the customer management system architecture"
        context2 = ReasoningContext(
            goal=refined_goal,
            metadata={"previous_analysis": result1}
        )
        result2 = await self.engine.reason(refined_goal, context2)
        
        # Third iteration - implementation details
        detail_goal = "Create implementation plan for customer management system"
        context3 = ReasoningContext(
            goal=detail_goal,
            metadata={"architecture": result2, "initial_analysis": result1}
        )
        result3 = await self.engine.reason(detail_goal, context3)
        
        # All results should be valid
        results = [result1, result2, result3]
        for result in results:
            assert isinstance(result, dict)
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_domain_specific_reasoning(self):
        """Test reasoning across different domains"""
        domain_goals = [
            ("healthcare", "Design a patient monitoring system with HIPAA compliance"),
            ("finance", "Create a trading algorithm with risk management"),
            ("education", "Develop an adaptive learning platform"),
            ("logistics", "Optimize supply chain delivery routes"),
            ("security", "Implement zero-trust network architecture")
        ]
        
        for domain, goal in domain_goals:
            context = ReasoningContext(
                goal=goal,
                metadata={"domain": domain}
            )
            
            result = await self.engine.reason(goal, context)
            
            assert isinstance(result, dict)
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_reasoning_error_handling(self):
        """Test reasoning engine error handling"""
        # Test with problematic inputs
        problematic_scenarios = [
            ("", {}),  # Empty goal
            ("valid goal", None),  # None context (should create default)
            ("goal with unicode: 🚀🎯💡", {}),  # Unicode characters
        ]
        
        for goal, context_data in problematic_scenarios:
            try:
                if context_data is None:
                    result = await self.engine.reason(goal)
                else:
                    context = ReasoningContext(goal=goal, **context_data)
                    result = await self.engine.reason(goal, context)
                
                # Should either succeed or fail gracefully
                if result is not None:
                    assert isinstance(result, dict)
                    
            except Exception as e:
                # Errors should be meaningful
                assert isinstance(e, (ValueError, TypeError))
                assert len(str(e)) > 0
    
    @pytest.mark.asyncio
    async def test_reasoning_session_isolation(self):
        """Test that reasoning sessions are properly isolated"""
        # Start multiple concurrent sessions
        goals = [
            "Analyze market trends",
            "Design system architecture", 
            "Plan project timeline"
        ]
        
        contexts = [ReasoningContext(goal=goal) for goal in goals]
        
        # Start all sessions
        tasks = [self.engine.reason(goal, context) for goal, context in zip(goals, contexts)]
        
        # Verify sessions are tracked
        # Note: This test depends on timing, sessions might complete before we check
        
        # Complete all sessions
        results = await asyncio.gather(*tasks)
        
        # Verify all completed successfully
        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)
            assert result is not None
        
        # Verify sessions were cleaned up
        assert len(self.engine.active_sessions) == 0
    
    @pytest.mark.asyncio
    async def test_reasoning_with_collaboration_modes(self):
        """Test reasoning with different collaboration requirements"""
        goal = "Implement new security policy"
        
        collaboration_scenarios = [
            CollaborationMode.APPROVAL,
            CollaborationMode.INPUT,
            CollaborationMode.REVIEW,
            CollaborationMode.OVERRIDE
        ]
        
        for mode in collaboration_scenarios:
            context = ReasoningContext(
                goal=goal,
                metadata={"collaboration_mode": mode.value}
            )
            
            result = await self.engine.reason(goal, context)
            
            assert isinstance(result, dict)
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_reasoning_performance_characteristics(self):
        """Test reasoning performance with various loads"""
        import time
        
        # Test single reasoning performance
        start_time = time.time()
        result = await self.engine.reason("Simple analysis task")
        single_duration = time.time() - start_time
        
        assert isinstance(result, dict)
        assert single_duration < 5.0  # Should complete within 5 seconds
        
        # Test concurrent reasoning performance
        start_time = time.time()
        tasks = [
            self.engine.reason(f"Task {i}")
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)
        concurrent_duration = time.time() - start_time
        
        assert len(results) == 5
        assert all(isinstance(result, dict) for result in results)
        # Concurrent should not be much slower than 5x single
        assert concurrent_duration < single_duration * 10
    
    @pytest.mark.asyncio
    async def test_reasoning_memory_integration(self):
        """Test reasoning integration with memory system"""
        goal = "Analyze user behavior patterns"
        context = ReasoningContext(goal=goal)
        
        # Execute reasoning
        result = await self.engine.reason(goal, context)
        
        # Verify memory system is accessible
        assert self.engine.memory is not None
        assert hasattr(self.engine.memory, 'store')
        
        # Result should be valid
        assert isinstance(result, dict)
        assert result is not None
