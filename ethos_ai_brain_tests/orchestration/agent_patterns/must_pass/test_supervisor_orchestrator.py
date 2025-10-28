"""
Test SupervisorOrchestrator - Must Pass
Tests the Supervisor orchestration pattern for centralized agent coordination
Uses real code only - no mocks or fallbacks
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.orchestration.agent_patterns.orchestration_patterns import SupervisorOrchestrator
from ethos_ai_brain.orchestration.agent_patterns.orchestration_base import (
    OrchestrationPattern, AgentMessage, AgentCapability, AgentOrchestrationBase
)
from ethos_ai_brain.core.ai_agent.ai_brain import AIBrain


def test_supervisor_orchestrator_creation():
    """Test SupervisorOrchestrator creation and initialization"""
    supervisor = SupervisorOrchestrator("supervisor_001")
    
    # Test basic properties
    print(f"Expected agent_id: supervisor_001, Actual: {supervisor.agent_id}")
    print(f"Expected pattern: SUPERVISOR, Actual: {supervisor.orchestration_pattern}")
    print(f"Expected empty assignments: {{}}, Actual: {supervisor.agent_assignments}")
    print(f"Expected empty workloads: {{}}, Actual: {supervisor.agent_workloads}")
    
    assert supervisor.agent_id == "supervisor_001"
    assert supervisor.orchestration_pattern == OrchestrationPattern.SUPERVISOR
    assert supervisor.agent_assignments == {}
    assert supervisor.agent_workloads == {}
    
    print("[SUCCESS] SupervisorOrchestrator created successfully")


def test_supervisor_orchestrator_pattern_enum():
    """Test that supervisor uses correct orchestration pattern"""
    supervisor = SupervisorOrchestrator("supervisor_002")
    
    print(f"Expected pattern: {OrchestrationPattern.SUPERVISOR}, Actual: {supervisor.orchestration_pattern}")
    
    assert supervisor.orchestration_pattern == OrchestrationPattern.SUPERVISOR
    assert supervisor.orchestration_pattern.value == "supervisor"
    
    print("[SUCCESS] Orchestration pattern is correct")


def test_supervisor_orchestrator_inheritance():
    """Test that SupervisorOrchestrator properly inherits from base class"""
    supervisor = SupervisorOrchestrator("supervisor_003")
    
    print(f"Expected inheritance from AgentOrchestrationBase: {isinstance(supervisor, AgentOrchestrationBase)}")
    print(f"Has child_agents attribute: {hasattr(supervisor, 'child_agents')}")
    print(f"Has brain attribute: {hasattr(supervisor, 'brain')}")
    
    assert isinstance(supervisor, AgentOrchestrationBase)
    assert hasattr(supervisor, 'child_agents')
    assert hasattr(supervisor, 'brain')
    assert hasattr(supervisor, 'execute_task')
    
    print("[SUCCESS] Inheritance structure is correct")


def test_supervisor_orchestrator_agent_assignments_structure():
    """Test agent assignments and workloads data structures"""
    supervisor = SupervisorOrchestrator("supervisor_004")
    
    # Test initial state
    print(f"Agent assignments type: {type(supervisor.agent_assignments)}")
    print(f"Agent workloads type: {type(supervisor.agent_workloads)}")
    
    assert isinstance(supervisor.agent_assignments, dict)
    assert isinstance(supervisor.agent_workloads, dict)
    
    # Test that we can modify these structures
    supervisor.agent_assignments["test_task"] = "test_agent"
    supervisor.agent_workloads["test_agent"] = 1
    
    print(f"After modification - assignments: {supervisor.agent_assignments}")
    print(f"After modification - workloads: {supervisor.agent_workloads}")
    
    assert supervisor.agent_assignments["test_task"] == "test_agent"
    assert supervisor.agent_workloads["test_agent"] == 1
    
    print("[SUCCESS] Data structures work correctly")


def test_supervisor_orchestrator_methods_exist():
    """Test that required methods exist and are callable"""
    supervisor = SupervisorOrchestrator("supervisor_005")
    
    # Test required methods exist
    required_methods = [
        '_start_orchestration',
        '_analyze_agent_capabilities', 
        'execute_task',
        '_route_message_to_child'
    ]
    
    for method_name in required_methods:
        print(f"Checking method: {method_name}")
        assert hasattr(supervisor, method_name), f"Missing method: {method_name}"
        assert callable(getattr(supervisor, method_name)), f"Method not callable: {method_name}"
    
    print("[SUCCESS] All required methods exist and are callable")


async def test_supervisor_orchestrator_empty_capability_analysis():
    """Test capability analysis with no child agents"""
    supervisor = SupervisorOrchestrator("supervisor_006")
    
    # Test with no child agents - should not crash
    await supervisor._analyze_agent_capabilities()
    
    # Should still have empty assignments
    print(f"Agent assignments after empty analysis: {supervisor.agent_assignments}")
    
    assert supervisor.agent_assignments == {}
    
    print("[SUCCESS] Empty capability analysis handled correctly")


async def test_supervisor_orchestrator_task_execution_no_brain():
    """Test task execution fails gracefully without brain"""
    supervisor = SupervisorOrchestrator("supervisor_007")
    
    task = {"id": "task_001", "type": "test_task"}
    
    # This should fail because brain is None
    try:
        result = await supervisor.execute_task(task)
        # If we get here, check if it's an error result
        if isinstance(result, dict) and "error" in result:
            print(f"Expected error result: {result}")
            assert True  # This is expected behavior
        else:
            assert False, f"Expected error but got: {result}"
    except Exception as e:
        print(f"Expected exception: {type(e).__name__}: {e}")
        # This is also acceptable - the method should fail without a brain
        assert True
    
    print("[SUCCESS] Task execution fails appropriately without brain")
