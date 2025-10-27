#!/usr/bin/env python3
"""
AI Command System - Must Pass Test Runner
Runs only the critical tests that MUST pass for system functionality
NO MOCKING - NO FALLBACKS - REAL TESTS ONLY
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_must_pass_tests():
    """Run all must-pass tests for AI Command System"""
    
    # Base test directory
    test_base = Path(__file__).parent
    
    # Must-pass test directories
    must_pass_dirs = [
        test_base / "zmq_command_bridge" / "must-pass",
        test_base / "ai_agent_manager" / "must-pass", 
        test_base / "mission_parser" / "must-pass",
        test_base / "meta_reasoning_engine" / "must-pass",
        test_base / "major_general" / "must-pass",
        test_base / "integration" / "must-pass"
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    logger.info("=" * 60)
    logger.info("AI COMMAND SYSTEM - MUST PASS TESTS")
    logger.info("=" * 60)
    logger.info("Running critical tests with REAL ZMQ integration")
    logger.info("NO MOCKING - NO FALLBACKS")
    logger.info("=" * 60)
    
    for test_dir in must_pass_dirs:
        if not test_dir.exists():
            logger.warning(f"Test directory not found: {test_dir}")
            continue
            
        component_name = test_dir.parent.name
        logger.info(f"\n[TESTING] {component_name.upper()}")
        logger.info("-" * 40)
        
        # Find all test files in directory
        test_files = list(test_dir.glob("test_*.py"))
        
        if not test_files:
            logger.warning(f"No test files found in {test_dir}")
            continue
        
        for test_file in test_files:
            test_name = test_file.stem
            logger.info(f"Running: {test_name}")
            
            try:
                # Run pytest on specific file
                result = subprocess.run([
                    sys.executable, "-m", "pytest", 
                    str(test_file), 
                    "-v", 
                    "--tb=short"
                ], capture_output=True, text=True, cwd=test_base)
                
                total_tests += 1
                
                if result.returncode == 0:
                    passed_tests += 1
                    logger.info(f"  [PASS] {test_name}")
                else:
                    failed_tests += 1
                    logger.error(f"  [FAIL] {test_name}")
                    logger.error(f"  Error: {result.stderr.strip()}")
                    
            except Exception as e:
                failed_tests += 1
                total_tests += 1
                logger.error(f"  [ERROR] {test_name}: {e}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    
    if total_tests > 0:
        success_rate = (passed_tests / total_tests) * 100
        logger.info(f"Success Rate: {success_rate:.1f}%")
    
    if failed_tests > 0:
        logger.error("\nFAILED TESTS:")
        logger.error("These are CRITICAL failures that must be fixed!")
        return False
    else:
        logger.info("\nALL MUST-PASS TESTS SUCCESSFUL!")
        logger.info("AI Command System core functionality is operational")
        return True

if __name__ == "__main__":
    success = run_must_pass_tests()
    sys.exit(0 if success else 1)
