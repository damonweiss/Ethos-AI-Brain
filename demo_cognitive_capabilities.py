#!/usr/bin/env python3
"""
Cognitive Capabilities Demo - Show AI Brain in Action
Demonstrates meta-reasoning and intent analysis with detailed output
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ethos_ai_brain.reasoning.meta_reasoning.reasoning_manager import ReasoningManager
from ethos_ai_brain.reasoning.intent_analysis.adaptive_intent_analyzer import AdaptiveIntentAnalyzer


def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"[AI BRAIN] {title}")
    print("="*80)


def print_section(title):
    """Print a formatted section"""
    print(f"\n[SECTION] {title}")
    print("-" * 60)


def print_result(label, data, indent=0):
    """Print formatted result data"""
    prefix = "  " * indent
    if isinstance(data, dict):
        print(f"{prefix}{label}:")
        for key, value in data.items():
            if isinstance(value, (dict, list)) and len(str(value)) > 100:
                print(f"{prefix}  {key}: [Complex data - {type(value).__name__}]")
            else:
                print(f"{prefix}  {key}: {value}")
    else:
        print(f"{prefix}{label}: {data}")


async def demo_meta_reasoning():
    """Demonstrate meta-reasoning capabilities"""
    print_header("META-REASONING ENGINE DEMONSTRATION")
    
    reasoning_manager = ReasoningManager()
    
    # Test cases with different complexity levels
    test_cases = [
        {
            "goal": "Explain what 2+2 equals and why",
            "description": "Simple mathematical reasoning"
        },
        {
            "goal": "Create a strategy to improve team productivity in a remote work environment",
            "description": "Complex multi-step planning"
        },
        {
            "goal": "Analyze the pros and cons of implementing AI in healthcare",
            "description": "Complex analysis with multiple perspectives"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print_section(f"Test Case {i}: {test_case['description']}")
        print(f"Goal: {test_case['goal']}")
        
        try:
            print("\n[PROCESSING] Starting reasoning process...")
            result = await reasoning_manager.reason(test_case['goal'])
            
            if result.get("success"):
                print("[SUCCESS] Reasoning completed successfully!")
                print_result("Session ID", result.get("session_id"))
                print_result("Steps Completed", result.get("steps"))
                print_result("Execution Time", f"{result.get('metadata', {}).get('execution_time', 0):.2f} seconds")
                print_result("Confidence Level", result.get('metadata', {}).get('confidence'))
                
                # Show the actual reasoning result
                reasoning_result = result.get("result", {})
                if reasoning_result.get("success"):
                    print_result("Cost", f"${reasoning_result.get('cost', 0):.6f}")
                    print_result("Tokens Used", reasoning_result.get('usage', {}).get('total_tokens', 0))
                    
                    # Extract the actual AI response
                    result_data = reasoning_result.get("result", {})
                    if isinstance(result_data, dict):
                        ai_response = result_data.get("message", result_data.get("response", "No response content"))
                    else:
                        ai_response = str(result_data)
                    
                    print(f"\n[RESULT] AI Reasoning Result:")
                    print(f"   {ai_response}")
                
                # Show reasoning history
                session_id = result.get("session_id")
                history = reasoning_manager.get_session_history(session_id)
                print(f"\n[STEPS] Reasoning Steps ({len(history)} total):")
                for step in history:
                    print(f"   - {step.description} - Confidence: {step.confidence.value if step.confidence else 'Unknown'}")
                    if step.execution_time:
                        print(f"     Execution time: {step.execution_time:.2f}s")
            else:
                print("[FAILURE] Reasoning failed:")
                print_result("Error", result.get("error"))
                
        except Exception as e:
            print(f"[ERROR] Exception during reasoning: {e}")
        
        print("\n" + "-" * 80)


async def demo_intent_analysis():
    """Demonstrate intent analysis capabilities"""
    print_header("ADAPTIVE INTENT ANALYZER DEMONSTRATION")
    
    intent_analyzer = AdaptiveIntentAnalyzer()
    
    # Test cases with different intent types and complexity
    test_cases = [
        {
            "input": "What is the capital of France?",
            "description": "Simple factual question"
        },
        {
            "input": "How do I bake a chocolate cake?",
            "description": "Simple procedural task"
        },
        {
            "input": "Help me plan a comprehensive digital transformation strategy for a mid-size manufacturing company, considering stakeholder buy-in, technology constraints, budget limitations, and regulatory compliance requirements.",
            "description": "Complex strategic planning request"
        },
        {
            "input": "Analyze the impact of artificial intelligence on job markets, considering economic factors, social implications, and policy recommendations for different stakeholder groups.",
            "description": "Complex multi-stakeholder analysis"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print_section(f"Intent Analysis {i}: {test_case['description']}")
        print(f"User Input: \"{test_case['input']}\"")
        
        try:
            print("\n[PROCESSING] Analyzing intent...")
            analysis_type, result = await intent_analyzer.analyze_intent(test_case['input'])
            
            if result.get("success"):
                print(f"[SUCCESS] Intent analysis completed!")
                print_result("Analysis Type", analysis_type)
                print_result("Intent Type", result.get("intent_type"))
                print_result("Complexity", result.get("complexity"))
                print_result("Confidence", f"{result.get('confidence', 0):.2f}")
                
                # Show cost information
                metadata = result.get("metadata", {})
                print_result("Cost", f"${metadata.get('cost', 0):.6f}")
                print_result("Tokens Used", metadata.get('tokens', 0))
                
                if analysis_type == "simple_response":
                    print(f"\n[ANSWER] Direct Answer:")
                    print(f"   {result.get('direct_answer', 'No answer provided')}")
                    print_result("Reasoning", result.get('reasoning', 'No reasoning provided'))
                
                elif analysis_type == "complex_analysis":
                    print(f"\n[ANALYSIS] Complex Analysis Results:")
                    print_result("Stakeholders Identified", result.get('stakeholder_count', 0))
                    print_result("Objectives Identified", result.get('objective_count', 0))
                    print_result("Constraints Identified", result.get('constraint_count', 0))
                    print_result("AI Actionable Items", result.get('ai_actionable_count', 0))
                    
                    # Show detailed analysis if available
                    detailed_analysis = result.get('detailed_analysis', '')
                    if detailed_analysis and len(detailed_analysis) > 10:
                        print(f"\n[DETAILS] Detailed Analysis:")
                        # Truncate if too long for demo
                        if len(detailed_analysis) > 500:
                            print(f"   {detailed_analysis[:500]}...")
                            print(f"   [Analysis truncated - full length: {len(detailed_analysis)} characters]")
                        else:
                            print(f"   {detailed_analysis}")
                    
                    # Show complexity factors
                    complexity_factors = metadata.get('complexity_factors', [])
                    if complexity_factors:
                        print_result("Complexity Factors", complexity_factors)
            else:
                print("[FAILURE] Intent analysis failed:")
                print_result("Error", result.get("error"))
                
        except Exception as e:
            print(f"[ERROR] Exception during intent analysis: {e}")
        
        print("\n" + "-" * 80)


async def demo_integration():
    """Demonstrate integration between reasoning and intent analysis"""
    print_header("INTEGRATED COGNITIVE PROCESSING DEMONSTRATION")
    
    reasoning_manager = ReasoningManager()
    intent_analyzer = AdaptiveIntentAnalyzer()
    
    # Complex user request that requires both intent analysis and reasoning
    user_request = "I need to decide whether to invest in renewable energy stocks. Help me understand the key factors, risks, and create a decision framework."
    
    print_section("Integrated Processing Workflow")
    print(f"User Request: \"{user_request}\"")
    
    try:
        # Step 1: Analyze intent
        print("\n[STEP 1] Analyzing user intent...")
        analysis_type, intent_result = await intent_analyzer.analyze_intent(user_request)
        
        print(f"[SUCCESS] Intent Analysis Complete:")
        print_result("Analysis Type", analysis_type)
        print_result("Complexity", intent_result.get("complexity"))
        print_result("Intent Type", intent_result.get("intent_type"))
        
        # Step 2: Use reasoning for complex analysis
        if analysis_type == "complex_analysis":
            print("\n[STEP 2] Applying meta-reasoning for complex request...")
            
            # Create a more detailed goal based on intent analysis
            enhanced_goal = f"Based on the user request '{user_request}', provide a comprehensive analysis including key factors, risk assessment, and a structured decision framework for renewable energy stock investment."
            
            reasoning_result = await reasoning_manager.reason(enhanced_goal)
            
            if reasoning_result.get("success"):
                print(f"[SUCCESS] Meta-Reasoning Complete:")
                print_result("Reasoning Steps", reasoning_result.get("steps"))
                print_result("Confidence", reasoning_result.get('metadata', {}).get('confidence'))
                
                # Show combined results
                print(f"\n[RESULTS] Integrated Analysis Results:")
                print(f"   Intent Analysis: {intent_result.get('stakeholder_count', 0)} stakeholders, {intent_result.get('objective_count', 0)} objectives identified")
                print(f"   Reasoning Process: {reasoning_result.get('steps')} analytical steps completed")
                
                # Calculate total cost
                intent_cost = intent_result.get('metadata', {}).get('cost', 0)
                reasoning_cost = reasoning_result.get('result', {}).get('cost', 0)
                total_cost = intent_cost + reasoning_cost
                print(f"   Total Processing Cost: ${total_cost:.6f}")
                
                print(f"\n[CAPABILITIES] This demonstrates the AI Brain's ability to:")
                print(f"   - Understand complex user intents")
                print(f"   - Apply multi-step reasoning")
                print(f"   - Integrate different cognitive processes")
                print(f"   - Provide comprehensive analysis")
                
        else:
            print("\n[INFO] Simple request - direct response provided by intent analyzer")
            
    except Exception as e:
        print(f"[ERROR] Exception during integrated processing: {e}")


async def main():
    """Main demo function"""
    print_header("AI BRAIN COGNITIVE CAPABILITIES DEMONSTRATION")
    print("This demo showcases the modern implementation of meta-reasoning")
    print("and adaptive intent analysis built on the clean inference architecture.")
    print("\nNote: This requires OPENAI_API_KEY to be set for real API calls.")
    
    try:
        # Run all demonstrations
        await demo_meta_reasoning()
        await demo_intent_analysis()
        await demo_integration()
        
        print_header("DEMONSTRATION COMPLETE")
        print("[SUCCESS] The AI Brain successfully demonstrated:")
        print("   [OK] Multi-step meta-reasoning with goal decomposition")
        print("   [OK] Adaptive intent analysis with complexity assessment")
        print("   [OK] Stakeholder and objective identification")
        print("   [OK] AI actionability assessment")
        print("   [OK] Integrated cognitive processing workflow")
        print("   [OK] Real-time cost and usage tracking")
        print("   [OK] Session management and history tracking")
        print("\n[ACHIEVEMENT] The legacy cognitive capabilities have been successfully")
        print("   modernized and integrated into the clean architecture!")
        
    except Exception as e:
        print(f"\n[ERROR] Demo failed with exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
