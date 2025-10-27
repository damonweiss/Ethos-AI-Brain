#!/usr/bin/env python3
"""
Test Script for AI Command System
Tests Major General, Mission Parser, Agent Manager, and ZMQ Bridge
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import AI Command System components
from major_general import MajorGeneral
from mission_parser import MissionParser, MissionPriority
from ai_agent_manager import AIAgentManager, AgentType
from zmq_command_bridge import ZMQCommandBridge

class AICommandSystemTester:
    """Test suite for AI Command System"""
    
    def __init__(self):
        self.test_results = []
        self.major_general = None
        self.mission_parser = None
        self.agent_manager = None
        self.zmq_bridge = None
    
    def log_test_result(self, test_name: str, success: bool, message: str = ""):
        """Log test result"""
        status = "PASS" if success else "FAIL"
        result = {
            'test': test_name,
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        logger.info(f"TEST {status}: {test_name} - {message}")
    
    def test_mission_parser(self):
        """Test Mission Parser functionality"""
        logger.info("=== Testing Mission Parser ===")
        
        try:
            self.mission_parser = MissionParser()
            
            # Test 1: Basic mission parsing
            objective = "Analyze customer feedback and create improvement recommendations"
            mission = self.mission_parser.parse_mission(objective)
            
            assert mission.objective == objective
            assert mission.priority == MissionPriority.NORMAL
            assert mission.id.startswith("mission_")
            
            self.log_test_result("Mission Parser - Basic Parsing", True, f"Mission ID: {mission.id}")
            
            # Test 2: Priority detection
            urgent_objective = "URGENT: Fix critical system vulnerability immediately"
            urgent_mission = self.mission_parser.parse_mission(urgent_objective)
            
            assert urgent_mission.priority == MissionPriority.HIGH
            self.log_test_result("Mission Parser - Priority Detection", True, f"Priority: {urgent_mission.priority.value}")
            
            # Test 3: Mission validation
            try:
                self.mission_parser.parse_mission("")  # Should fail
                self.log_test_result("Mission Parser - Validation", False, "Empty objective should fail")
            except ValueError:
                self.log_test_result("Mission Parser - Validation", True, "Empty objective correctly rejected")
            
        except Exception as e:
            self.log_test_result("Mission Parser", False, f"Error: {e}")
    
    def test_agent_manager(self):
        """Test AI Agent Manager functionality"""
        logger.info("=== Testing AI Agent Manager ===")
        
        try:
            self.agent_manager = AIAgentManager()
            
            # Test 1: Agent spawning
            agent_id = self.agent_manager.spawn_agent(AgentType.STRATEGIC_ADVISOR)
            assert agent_id in self.agent_manager.agents
            
            agent = self.agent_manager.get_agent(agent_id)
            assert agent.agent_type == AgentType.STRATEGIC_ADVISOR
            
            self.log_test_result("Agent Manager - Agent Spawning", True, f"Agent ID: {agent_id}")
            
            # Test 2: Agent lifecycle
            agent.assign_mission("test_mission_001")
            assert agent.current_mission == "test_mission_001"
            
            agent.complete_mission(success=True)
            assert agent.current_mission is None
            assert agent.performance_metrics['tasks_completed'] == 1
            
            self.log_test_result("Agent Manager - Lifecycle", True, "Mission assignment and completion")
            
            # Test 3: Agent statistics
            stats = self.agent_manager.get_agent_stats()
            assert stats['total_agents'] >= 1
            assert 'strategic_advisor' in stats['by_type']
            
            self.log_test_result("Agent Manager - Statistics", True, f"Total agents: {stats['total_agents']}")
            
            # Test 4: Agent termination
            terminated = self.agent_manager.terminate_agent(agent_id)
            assert terminated == True
            assert agent_id not in self.agent_manager.agents
            
            self.log_test_result("Agent Manager - Termination", True, "Agent terminated successfully")
            
        except Exception as e:
            self.log_test_result("Agent Manager", False, f"Error: {e}")
    
    async def test_zmq_bridge(self):
        """Test ZMQ Command Bridge functionality"""
        logger.info("=== Testing ZMQ Command Bridge ===")
        
        try:
            self.zmq_bridge = ZMQCommandBridge()
            
            # Test 1: Network initialization
            network_status = self.zmq_bridge.get_network_status()
            assert 'zmq_available' in network_status
            assert 'active_servers' in network_status
            
            self.log_test_result("ZMQ Bridge - Initialization", True, f"ZMQ Available: {network_status['zmq_available']}")
            
            # Test 2: Biological pattern selection
            characteristics = {
                'complexity': 'high',
                'urgency': 'normal',
                'collaboration_needed': True
            }
            
            pattern = self.zmq_bridge.select_biological_pattern(characteristics)
            assert pattern in ['reflex_arc', 'spreading_activation', 'hierarchical_processing', 'parallel_processing', 'integration_centers']
            
            self.log_test_result("ZMQ Bridge - Pattern Selection", True, f"Selected pattern: {pattern}")
            
            # Test 3: Command sending (mock)
            test_command = {
                'type': 'test_command',
                'data': {'message': 'Hello from test'}
            }
            
            # Use await instead of asyncio.run to avoid nested event loop
            response = await self.zmq_bridge.send_command('major_general', test_command)
            assert 'status' in response
            
            self.log_test_result("ZMQ Bridge - Command Sending", True, f"Response: {response.get('status')}")
            
        except Exception as e:
            self.log_test_result("ZMQ Bridge", False, f"Error: {e}")
    
    async def test_major_general(self):
        """Test Major General functionality"""
        logger.info("=== Testing Major General ===")
        
        try:
            self.major_general = MajorGeneral()
            
            # Test 1: Initialization
            assert self.major_general.agent_id is not None
            assert len(self.major_general.active_missions) == 0
            
            self.log_test_result("Major General - Initialization", True, f"Agent ID: {self.major_general.agent_id}")
            
            # Test 2: Mission reception
            test_objective = "Create a comprehensive analysis of system performance and recommend optimizations"
            test_context = {
                'priority': 'high',
                'deadline': '2024-01-15',
                'test_mode': True
            }
            
            mission_id = await self.major_general.receive_mission(test_objective, test_context)
            assert mission_id in self.major_general.active_missions
            
            mission = self.major_general.active_missions[mission_id]
            assert mission.objective == test_objective
            
            self.log_test_result("Major General - Mission Reception", True, f"Mission ID: {mission_id}")
            
            # Test 3: Mission status
            mission_status = self.major_general.get_mission_status(mission_id)
            assert mission_status is not None
            assert mission_status['id'] == mission_id
            assert 'execution_details' in mission_status
            
            self.log_test_result("Major General - Mission Status", True, f"Status: {mission_status['status']}")
            
            # Test 4: NetworkX integration
            execution_graph = self.major_general.mission_execution_graphs.get(mission_id)
            assert execution_graph is not None
            assert len(execution_graph.nodes()) > 0
            
            # Check if it's a valid DAG
            import networkx as nx
            is_dag = nx.is_directed_acyclic_graph(execution_graph)
            
            self.log_test_result("Major General - NetworkX Integration", True, 
                               f"Graph nodes: {len(execution_graph.nodes())}, Is DAG: {is_dag}")
            
            # Test 5: Command metrics
            metrics = self.major_general.get_command_metrics()
            assert 'active_missions' in metrics
            assert metrics['active_missions'] >= 1
            
            self.log_test_result("Major General - Metrics", True, f"Active missions: {metrics['active_missions']}")
            
        except Exception as e:
            self.log_test_result("Major General", False, f"Error: {e}")
    
    async def test_integration(self):
        """Test integration between components"""
        logger.info("=== Testing Component Integration ===")
        
        try:
            # Test mission flow from parser to major general
            if self.major_general and self.mission_parser:
                
                # Create mission with parser
                objective = "Integrate all system components and validate end-to-end functionality"
                parsed_mission = self.mission_parser.parse_mission(objective)
                
                # Submit to major general
                mission_id = await self.major_general.receive_mission(
                    parsed_mission.objective, 
                    parsed_mission.context
                )
                
                # Verify integration
                mg_mission = self.major_general.active_missions[mission_id]
                assert mg_mission.objective == parsed_mission.objective
                
                self.log_test_result("Integration - Parser to Major General", True, 
                                   f"Mission flow successful: {mission_id}")
            
            # Test ZMQ integration with Major General
            if self.major_general and self.zmq_bridge:
                
                # Test biological pattern selection for mission
                mission_characteristics = {
                    'complexity': 'high',
                    'urgency': 'normal',
                    'collaboration_needed': True,
                    'parallel_opportunities': 2
                }
                
                pattern = self.zmq_bridge.select_biological_pattern(mission_characteristics)
                
                # Execute pattern with mission data
                pattern_data = {
                    'mission_type': 'analysis',
                    'data': {'test': 'integration_test'}
                }
                
                response = await self.zmq_bridge.execute_biological_pattern(pattern, pattern_data)
                assert 'status' in response
                
                self.log_test_result("Integration - Major General to ZMQ", True, 
                                   f"Pattern executed: {pattern}")
            
        except Exception as e:
            self.log_test_result("Integration", False, f"Error: {e}")
    
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting AI Command System Tests")
        logger.info("=" * 50)
        
        # Run individual component tests
        self.test_mission_parser()
        self.test_agent_manager()
        await self.test_zmq_bridge()
        await self.test_major_general()
        await self.test_integration()
        
        # Print summary
        logger.info("=" * 50)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 50)
        
        passed = sum(1 for result in self.test_results if result['status'] == 'PASS')
        failed = sum(1 for result in self.test_results if result['status'] == 'FAIL')
        total = len(self.test_results)
        
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if failed > 0:
            logger.info("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if result['status'] == 'FAIL':
                    logger.info(f"  - {result['test']}: {result['message']}")
        
        logger.info("\n✅ AI Command System Testing Complete!")
        
        return passed, failed, total

async def main():
    """Main test function"""
    tester = AICommandSystemTester()
    passed, failed, total = await tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    asyncio.run(main())
