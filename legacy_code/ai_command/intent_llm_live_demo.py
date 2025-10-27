#!/usr/bin/env python3
"""
Intent Knowledge Graph Live LLM Demo
Demonstrates actual LLM calls using intent graph prompts and validation
"""

import json
import os
from typing import Optional
from dotenv import load_dotenv
from intent_knowledge_graph import IntentKnowledgeGraph
from knowledge_graph_visualizer import MatplotlibKnowledgeGraphVisualizer

# Load environment variables
load_dotenv()

from openai import OpenAI

def get_llm_client() -> OpenAI:
    """Get OpenAI client with API key from environment"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    return OpenAI(api_key=api_key)

def call_llm_with_prompt(client: OpenAI, prompt: str) -> str:
    """Call LLM with the generated prompt"""
    print("Calling OpenAI GPT-4...")
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert intent analysis system. Generate only valid JSON responses that match the provided schema exactly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,  # Low temperature for consistent JSON
        max_tokens=4000
    )
    
    return response.choices[0].message.content


def demo_live_llm_integration():
    """Demonstrate live LLM integration with intent graph"""
    print("INTENT KNOWLEDGE GRAPH LIVE LLM DEMO")
    print("=" * 70)
    print("Demonstrating actual LLM calls with prompt generation and validation")
    print("=" * 70)
    
    # Test cases for different domains
    test_cases = [
        {
            'user_input': "I need a new POS system for my restaurant that handles orders, payments, and inventory",
            'domain': "restaurant_management"
        },
        {
            'user_input': "Build a customer service chatbot for our e-commerce website with AI capabilities",
            'domain': "ai_development"
        },
        {
            'user_input': "Create a project management system for our construction company with mobile access",
            'domain': "construction_management"
        }
    ]
    
    # Get LLM client
    print("Initializing OpenAI client...")
    client = get_llm_client()
    print("[PASS] OpenAI client initialized successfully")
    
    # Process each test case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*20} TEST CASE {i} {'='*20}")
        print(f"User Input: {test_case['user_input']}")
        print(f"Domain: {test_case['domain']}")
        
        # Step 1: Generate LLM prompt
        print(f"\n1. GENERATING LLM PROMPT")
        print("-" * 40)
        
        prompt = IntentKnowledgeGraph.get_llm_query_prompt(
            test_case['user_input'], 
            test_case['domain']
        )
        
        print(f"Generated prompt: {len(prompt)} characters")
        print("Prompt preview (first 200 chars):")
        print(f"  {prompt[:200]}...")
        
        # Step 2: Call LLM
        print(f"\n2. CALLING LLM")
        print("-" * 40)
        
        llm_response = call_llm_with_prompt(client, prompt)
        
        if llm_response:
            print(f"LLM response received: {len(llm_response)} characters")
            
            # Try to extract JSON from response
            try:
                # LLM might return JSON wrapped in markdown code blocks
                if '```json' in llm_response:
                    json_start = llm_response.find('```json') + 7
                    json_end = llm_response.find('```', json_start)
                    json_str = llm_response[json_start:json_end].strip()
                elif '{' in llm_response:
                    # Find the JSON object
                    json_start = llm_response.find('{')
                    json_end = llm_response.rfind('}') + 1
                    json_str = llm_response[json_start:json_end]
                else:
                    json_str = llm_response
                
                llm_json = json.loads(json_str)
                print("[PASS] Valid JSON extracted from LLM response")
                
                # Output the full JSON response from LLM
                print(f"\nFULL LLM JSON RESPONSE:")
                print("=" * 60)
                print(json.dumps(llm_json, indent=2))
                print("=" * 60)
                
            except json.JSONDecodeError as e:
                print(f"[FAIL] JSON parsing failed: {str(e)}")
                print("Raw LLM response:")
                print(llm_response[:500] + "..." if len(llm_response) > 500 else llm_response)
                continue
        else:
            print("[FAIL] No response from LLM")
            continue
        
        # Step 3: Validate with Pydantic
        print(f"\n3. VALIDATING WITH PYDANTIC")
        print("-" * 40)
        
        intent = IntentKnowledgeGraph("llm_generated", "LLM Generated Intent")
        validation_result = intent.validate_and_import_json(llm_json)
        
        if validation_result['valid']:
            print("[PASS] Pydantic validation PASSED")
            
            # Show import statistics
            stats = validation_result['import_stats']
            print("Import statistics:")
            for key, value in stats.items():
                if value > 0:
                    print(f"  - {key}: {value}")
            
            # Step 4: Show 3D visualization BEFORE auto-linking
            print(f"\n4. 3D VISUALIZATION - BEFORE AUTO-LINKING")
            print("-" * 40)
            
            print(f"Before auto-linking: {len(intent.nodes())} nodes, {len(intent.edges())} edges")
            
            viz_before = MatplotlibKnowledgeGraphVisualizer(intent)
            viz_before.render_3d_graph(
                figsize=(12, 8),
                show_labels=True,
                positioning_strategy="layered"
            )
            
            # Step 5: Apply auto-linking
            print(f"\n5. APPLYING AUTO-LINKING")
            print("-" * 40)
            
            link_stats = intent.comprehensive_auto_link()
            
            print(f"After auto-linking: {len(intent.nodes())} nodes, {len(intent.edges())} edges")
            print(f"Links added: {link_stats['total_links_added']}")
            print(f"Edge types: {link_stats['edge_type_distribution']}")
            
            # Step 6: Show 3D visualization AFTER auto-linking
            print(f"\n6. 3D VISUALIZATION - AFTER AUTO-LINKING")
            print("-" * 40)
            
            viz_after = MatplotlibKnowledgeGraphVisualizer(intent)
            viz_after.render_3d_graph(
                figsize=(14, 10),
                show_labels=True,
                positioning_strategy="layered"
            )
            
            # Step 7: Show analysis results
            print(f"\n7. INTENT ANALYSIS RESULTS")
            print("-" * 40)
            
            # Gap assessment
            gaps = intent.assess_gaps()
            print(f"Gaps identified: {len(gaps)}")
            if gaps:
                critical_gaps = [g for g in gaps if g['severity'] in ['critical', 'high']]
                print(f"Critical/High gaps: {len(critical_gaps)}")
                for gap in critical_gaps[:2]:  # Show first 2
                    print(f"  - {gap['gap_type'].upper()}: {gap['description']}")
            
            # Feasibility assessment
            feasibility = intent.assess_feasibility()
            print(f"Feasibility score: {feasibility['overall_feasibility']:.2f}")
            
            # Intent completeness
            completeness = intent.get_intent_completeness_score()
            print(f"Intent completeness: {completeness:.2f}")
            
            # Show some extracted data
            print(f"\nExtracted intent data:")
            print(f"  - Domain: {intent.domain_context}")
            print(f"  - Sentiment: {intent.user_sentiment}")
            print(f"  - Clarifying questions: {len(intent.clarifying_questions)}")
            print(f"  - Implicit requirements: {len(intent.implicit_requirements)}")
            
            if intent.clarifying_questions:
                print(f"  - Clarifying questions ({len(intent.clarifying_questions)}):")
                for i, q in enumerate(intent.clarifying_questions, 1):
                    print(f"    {i}. {q['question']}")
                    if 'related_element' in q:
                        print(f"       Related to: {q['related_element']}")
            
            if intent.implicit_requirements:
                print(f"  - Implicit requirements ({len(intent.implicit_requirements)}):")
                for i, req in enumerate(intent.implicit_requirements, 1):
                    print(f"    {i}. {req['requirement']} (confidence: {req.get('confidence', 'N/A')})")
                    if 'rationale' in req:
                        print(f"       Rationale: {req['rationale']}")
            
        else:
            print("[FAIL] Pydantic validation FAILED")
            print("Validation errors:")
            for error in validation_result['errors']:
                print(f"  - {error}")
        
        # Only process first test case for detailed demo
        if i == 1:
            print(f"\n{'='*50}")
            print("Detailed demo complete for first test case.")
            print("Additional test cases would follow the same pattern.")
            print(f"{'='*50}")
            break
    
    print(f"\nLIVE LLM INTEGRATION DEMO COMPLETE!")
    print("=" * 70)
    print("Key Features Demonstrated:")
    print("- Generated comprehensive LLM prompts from intent graph schema")
    print("- Made actual LLM API calls (or used mock responses)")
    print("- Extracted and parsed JSON from LLM responses")
    print("- Validated LLM responses with Pydantic schema")
    print("- Automatically imported valid JSON into intent graphs")
    print("- Applied comprehensive auto-linking for rich relationships")
    print("- Performed complete intent analysis on LLM-generated data")
    print("- Demonstrated end-to-end LLM integration workflow")
    print("=" * 70)

def demo_refinement_workflow():
    """Demonstrate iterative refinement with LLM"""
    print(f"\nREFINEMENT WORKFLOW DEMO")
    print("=" * 50)
    
    # Simulate user feedback
    user_feedback = "Actually, I also need the POS system to integrate with our existing accounting software QuickBooks, and we need support for loyalty programs and gift cards."
    
    print(f"User feedback: {user_feedback}")
    
    # Create a simple current state
    current_json = {
        "objectives": [
            {"id": "payment_processing", "name": "Payment Processing", "priority": "critical"}
        ],
        "constraints": [
            {"id": "budget", "constraint_type": "budget", "value": 25000}
        ]
    }
    
    # Generate refinement prompt
    refinement_prompt = IntentKnowledgeGraph.get_llm_refinement_prompt(current_json, user_feedback)
    
    print(f"Generated refinement prompt: {len(refinement_prompt)} characters")
    print("Refinement prompt preview:")
    lines = refinement_prompt.split('\n')
    for line in lines[:10]:
        print(f"  {line}")
    print("  ... (truncated)")
    
    print("\nIn a real implementation, this prompt would be sent to the LLM")
    print("to generate an updated intent graph incorporating the user feedback.")

if __name__ == "__main__":
    demo_live_llm_integration()
    demo_refinement_workflow()
