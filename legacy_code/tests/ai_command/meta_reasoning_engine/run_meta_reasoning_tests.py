#!/usr/bin/env python3
"""
MetaReasoningEngine - Test Runner
Runs all MetaReasoningEngine tests with priority levels
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_meta_reasoning_tests():
    """Run all MetaReasoningEngine tests with priority levels"""
    
    # Base test directory
    test_base = Path(__file__).parent
    
    # Test directories by priority
    test_dirs = [
        ("MUST-PASS", test_base / "must-pass"),
        ("NICE-TO-PASS", test_base / "nice-to-pass"),
        ("OPTIONAL", test_base / "optional")
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    logger.info("=" * 70)
    logger.info("META-REASONING ENGINE - COMPREHENSIVE TEST SUITE")
    logger.info("=" * 70)
    logger.info("Testing cognitive architecture and LLM backend functionality")
    logger.info("=" * 70)
    
    for priority, test_dir in test_dirs:
        if not test_dir.exists():
            logger.warning(f"Test directory not found: {test_dir}")
            continue
            
        logger.info(f"\n[TESTING] {priority}")
        logger.info("-" * 50)
        
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
                    "--tb=short",
                    "-x"  # Stop on first failure for must-pass
                ], capture_output=True, text=True, cwd=test_base)
                
                total_tests += 1
                
                if result.returncode == 0:
                    passed_tests += 1
                    logger.info(f"  [PASS] {test_name}")
                else:
                    failed_tests += 1
                    logger.error(f"  [FAIL] {test_name}")
                    
                    # Show failure details for must-pass tests
                    if priority == "MUST-PASS":
                        logger.error(f"  CRITICAL FAILURE:")
                        if result.stderr:
                            logger.error(f"  {result.stderr.strip()}")
                        if result.stdout:
                            # Show last few lines of output
                            output_lines = result.stdout.strip().split('\n')
                            for line in output_lines[-10:]:
                                if 'FAILED' in line or 'ERROR' in line:
                                    logger.error(f"  {line}")
                    
            except Exception as e:
                failed_tests += 1
                total_tests += 1
                logger.error(f"  [ERROR] {test_name}: {e}")
        
        # Priority-specific reporting
        if priority == "MUST-PASS" and failed_tests > 0:
            logger.error(f"\n🚨 CRITICAL: {failed_tests} MUST-PASS tests failed!")
            logger.error("MetaReasoningEngine core functionality is broken!")
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("META-REASONING ENGINE TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    
    if total_tests > 0:
        success_rate = (passed_tests / total_tests) * 100
        logger.info(f"Success Rate: {success_rate:.1f}%")
    
    # Detailed analysis
    if failed_tests == 0:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("MetaReasoningEngine is fully operational")
        logger.info("✅ Cognitive architecture working")
        logger.info("✅ LLM backends functional")
        logger.info("✅ Reasoning workflows operational")
        return True
    else:
        logger.error(f"\n❌ {failed_tests} TESTS FAILED")
        
        if failed_tests <= total_tests * 0.1:  # 10% or less
            logger.warning("Minor issues detected - mostly functional")
        elif failed_tests <= total_tests * 0.3:  # 30% or less
            logger.warning("Moderate issues - core functionality may be impacted")
        else:
            logger.error("Major issues - significant functionality problems")
        
        return False

def run_must_pass_only():
    """Run only the must-pass tests for quick validation"""
    test_base = Path(__file__).parent
    must_pass_dir = test_base / "must-pass"
    
    if not must_pass_dir.exists():
        logger.error("Must-pass directory not found!")
        return False
    
    logger.info("=" * 50)
    logger.info("META-REASONING ENGINE - MUST-PASS ONLY")
    logger.info("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(must_pass_dir),
            "-v",
            "--tb=short"
        ], cwd=test_base)
        
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"Failed to run must-pass tests: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MetaReasoningEngine Test Runner")
    parser.add_argument("--must-pass-only", action="store_true", 
                       help="Run only must-pass tests")
    
    args = parser.parse_args()
    
    if args.must_pass_only:
        success = run_must_pass_only()
    else:
        success = run_meta_reasoning_tests()
    
    sys.exit(0 if success else 1)
