#!/usr/bin/env python3
"""
Simple test for AI Command System core functionality
Tests without external dependencies
"""

import asyncio
import logging
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported"""
    try:
        global MissionParser, Mission, MissionPriority, MissionStatus
        global AIAgentManager, AgentType, AgentStatus
        global MajorGeneral
        
        from mission_parser import MissionParser, Mission, MissionPriority, MissionStatus
        from ai_agent_manager import AIAgentManager, AgentType, AgentStatus
        from major_general import MajorGeneral
        logger.info("✅ All imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_mission_parser():
    """Test mission parser functionality"""
    try:
        parser = MissionParser()
        
        # Test basic parsing
        mission = parser.parse_mission("Create a comprehensive system analysis")
        assert mission.objective == "Create a comprehensive system analysis"
        assert mission.priority == MissionPriority.NORMAL
        assert mission.status == MissionStatus.RECEIVED
        
        # Test priority detection
        urgent_mission = parser.parse_mission("URGENT: Fix critical security vulnerability")
        assert urgent_mission.priority == MissionPriority.HIGH
        
        logger.info("✅ Mission Parser tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ Mission Parser test failed: {e}")
        return False

def test_agent_manager():
    """Test agent manager functionality"""
    try:
        manager = AIAgentManager()
        
        # Test agent spawning
        agent_id = manager.spawn_agent(AgentType.STRATEGIC_ADVISOR)
        assert agent_id in manager.agents
        
        # Test agent retrieval
        agent = manager.get_agent(agent_id)
        assert agent.agent_type == AgentType.STRATEGIC_ADVISOR
        assert agent.status == AgentStatus.ACTIVE
        
        # Test agent statistics
        stats = manager.get_agent_stats()
        assert stats['total_agents'] >= 1
        
        logger.info("✅ Agent Manager tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ Agent Manager test failed: {e}")
        return False

async def test_major_general():
    """Test Major General functionality"""
    try:
        mg = MajorGeneral()
        
        # Test initialization
        assert mg.agent_id is not None
        assert len(mg.active_missions) == 0
        
        # Test mission reception
        mission_id = await mg.receive_mission(
            "Analyze system performance and recommend optimizations",
            {"priority": "high", "test_mode": True}
        )
        
        assert mission_id in mg.active_missions
        
        # Test mission status
        status = mg.get_mission_status(mission_id)
        assert status is not None
        assert status['id'] == mission_id
        
        # Test NetworkX integration
        execution_graph = mg.mission_execution_graphs.get(mission_id)
        assert execution_graph is not None
        assert len(execution_graph.nodes()) > 0
        
        logger.info("✅ Major General tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ Major General test failed: {e}")
        return False

async def main():
    """Run all tests"""
    logger.info("🚀 Starting AI Command System Simple Tests")
    logger.info("=" * 50)
    
    passed = 0
    failed = 0
    
    # Test imports first
    logger.info("Running Imports test...")
    try:
        result = test_imports()
        if result:
            passed += 1
        else:
            failed += 1
    except Exception as e:
        logger.error(f"❌ Imports test crashed: {e}")
        failed += 1
    
    # Test mission parser
    logger.info("Running Mission Parser test...")
    try:
        result = test_mission_parser()
        if result:
            passed += 1
        else:
            failed += 1
    except Exception as e:
        logger.error(f"❌ Mission Parser test crashed: {e}")
        failed += 1
    
    # Test agent manager
    logger.info("Running Agent Manager test...")
    try:
        result = test_agent_manager()
        if result:
            passed += 1
        else:
            failed += 1
    except Exception as e:
        logger.error(f"❌ Agent Manager test crashed: {e}")
        failed += 1
    
    # Test Major General (async)
    logger.info("Running Major General test...")
    try:
        result = await test_major_general()
        if result:
            passed += 1
        else:
            failed += 1
    except Exception as e:
        logger.error(f"❌ Major General test crashed: {e}")
        failed += 1
    
    logger.info("=" * 50)
    logger.info("📊 TEST SUMMARY")
    logger.info(f"Total Tests: {passed + failed}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {(passed/(passed+failed))*100:.1f}%")
    
    if failed == 0:
        logger.info("🎉 All tests passed! AI Command System is working!")
    else:
        logger.warning(f"⚠️  {failed} tests failed")
    
    return failed == 0

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
