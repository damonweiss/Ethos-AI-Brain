#!/usr/bin/env python3
"""
Intent Knowledge Graph LLM Schema Demo
Demonstrates LLM query generation, Pydantic validation, and JSON import capabilities
"""

import json
from intent_knowledge_graph import IntentKnowledgeGraph, IntentGraphSchema

def demo_schema_template():
    """Demonstrate the schema template generation"""
    print("INTENT GRAPH SCHEMA TEMPLATE")
    print("=" * 60)
    
    # Get the complete schema
    schema = IntentKnowledgeGraph.get_schema_template()
    
    print("Pydantic Schema Structure:")
    print(f"- Title: {schema.get('title', 'IntentGraphSchema')}")
    print(f"- Properties: {len(schema.get('properties', {}))} top-level fields")
    
    # Show the main properties
    properties = schema.get('properties', {})
    for prop_name, prop_info in properties.items():
        prop_type = prop_info.get('type', 'unknown')
        description = prop_info.get('description', 'No description')
        print(f"  - {prop_name} ({prop_type}): {description}")
    
    print(f"\nComplete schema: {len(json.dumps(schema))} characters")
    return schema

def demo_llm_query_generation():
    """Demonstrate LLM query prompt generation"""
    print("\nLLM QUERY PROMPT GENERATION")
    print("=" * 60)
    
    # Test with different user inputs
    test_cases = [
        {
            'user_input': "I need to build a mobile app for my bakery to take online orders",
            'domain': "mobile_app_development"
        },
        {
            'user_input': "Plan my wedding for 150 guests in December with a $50k budget",
            'domain': "wedding_planning"
        },
        {
            'user_input': "Create a data science pipeline to analyze customer behavior",
            'domain': "data_science"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['domain']}")
        print("-" * 40)
        
        # Generate LLM prompt
        prompt = IntentKnowledgeGraph.get_llm_query_prompt(
            test_case['user_input'], 
            test_case['domain']
        )
        
        print(f"User Input: {test_case['user_input']}")
        print(f"Domain: {test_case['domain']}")
        print(f"Generated Prompt Length: {len(prompt)} characters")
        
        # Show first few lines of the prompt
        prompt_lines = prompt.split('\n')
        print("Prompt Preview:")
        for line in prompt_lines[:10]:
            print(f"  {line}")
        print("  ... (truncated)")
        
        # Show the JSON structure section
        json_start = next((i for i, line in enumerate(prompt_lines) if '```json' in line), -1)
        if json_start >= 0:
            print(f"\nJSON Structure starts at line {json_start + 1}")
    
    return test_cases[0]  # Return first test case for further demo

def demo_sample_json_creation():
    """Create a sample JSON that matches our schema"""
    print("\nSAMPLE JSON CREATION")
    print("=" * 60)
    
    # Create a sample JSON that matches our schema
    sample_json = {
        "graph_metadata": {
            "graph_id": "bakery_app_intent_001",
            "graph_type": "intent",
            "intent_name": "Bakery Mobile App Development",
            "created_date": "2024-01-15T10:30:00Z"
        },
        "intent_data": {
            "raw_user_prompt": "I need to build a mobile app for my bakery to take online orders",
            "domain_context": "mobile_app_development",
            "user_sentiment": "excited",
            "confidence_scores": {
                "objectives": 0.9,
                "constraints": 0.7,
                "stakeholders": 0.8
            },
            "ambiguities": [
                {
                    "element": "payment_methods",
                    "description": "Which payment methods should be supported?",
                    "severity": "medium"
                }
            ],
            "clarifying_questions": [
                {
                    "question": "Do you want delivery or pickup only?",
                    "related_element": "delivery_options"
                },
                {
                    "question": "What's your target launch date?",
                    "related_element": "timeline_constraint"
                }
            ],
            "implicit_requirements": [
                {
                    "requirement": "User authentication system",
                    "confidence": 0.9,
                    "rationale": "Online ordering requires user accounts for order tracking"
                },
                {
                    "requirement": "Push notifications for order updates",
                    "confidence": 0.8,
                    "rationale": "Customers expect real-time order status updates"
                }
            ]
        },
        "objectives": [
            {
                "id": "online_ordering",
                "name": "Enable Online Ordering",
                "description": "Allow customers to browse menu and place orders through mobile app",
                "priority": "critical",
                "measurable": True,
                "measurement_method": "order completion rate",
                "target_value": 0.85,
                "unit": "percentage",
                "complexity": "medium"
            },
            {
                "id": "increase_sales",
                "name": "Increase Sales Revenue",
                "description": "Boost bakery sales through convenient mobile ordering",
                "priority": "high",
                "measurable": True,
                "measurement_method": "monthly revenue increase",
                "target_value": 20.0,
                "unit": "percentage",
                "complexity": "low"
            }
        ],
        "constraints": [
            {
                "id": "budget_constraint",
                "constraint_type": "budget",
                "constraint": "Development budget cannot exceed $15,000",
                "value": 15000.0,
                "flexibility": "somewhat_flexible",
                "budget_breakdown": {
                    "development": 8000.0,
                    "design": 3000.0,
                    "testing": 2000.0,
                    "deployment": 2000.0
                }
            },
            {
                "id": "timeline_constraint",
                "constraint_type": "timeline",
                "constraint": "App must be ready before holiday season (3 months)",
                "value": 90.0,
                "flexibility": "rigid",
                "time_pressure": "tight"
            }
        ],
        "stakeholders": [
            {
                "id": "bakery_owner",
                "name": "Bakery Owner",
                "role": "primary_decision_maker",
                "influence_level": "high",
                "support_level": "strong",
                "decision_authority": "final"
            },
            {
                "id": "customers",
                "name": "Bakery Customers",
                "role": "end_users",
                "influence_level": "medium",
                "support_level": "unknown",
                "decision_authority": "none"
            }
        ],
        "technical_requirements": [
            {
                "id": "mobile_compatibility",
                "name": "Cross-Platform Mobile App",
                "description": "App must work on both iOS and Android devices",
                "complexity": "medium",
                "importance": "critical"
            },
            {
                "id": "payment_processing",
                "name": "Secure Payment Processing",
                "description": "Integration with payment gateway for credit card processing",
                "complexity": "high",
                "importance": "critical"
            }
        ],
        "assumptions": [
            {
                "id": "customer_adoption",
                "assumption": "Customers will adopt mobile ordering over phone orders",
                "confidence": 0.8,
                "impact_if_wrong": "high",
                "validation_method": "Customer survey and pilot testing"
            },
            {
                "id": "technical_feasibility",
                "assumption": "Required features can be implemented within budget",
                "confidence": 0.7,
                "impact_if_wrong": "critical",
                "validation_method": "Technical feasibility study"
            }
        ],
        "relationships": [
            {
                "source": "budget_constraint",
                "target": "online_ordering",
                "relationship": "constrains"
            },
            {
                "source": "timeline_constraint",
                "target": "online_ordering",
                "relationship": "constrains"
            },
            {
                "source": "bakery_owner",
                "target": "increase_sales",
                "relationship": "concerned_about"
            }
        ]
    }
    
    print("Created sample JSON with:")
    print(f"- Objectives: {len(sample_json['objectives'])}")
    print(f"- Constraints: {len(sample_json['constraints'])}")
    print(f"- Stakeholders: {len(sample_json['stakeholders'])}")
    print(f"- Technical Requirements: {len(sample_json['technical_requirements'])}")
    print(f"- Assumptions: {len(sample_json['assumptions'])}")
    print(f"- Relationships: {len(sample_json['relationships'])}")
    print(f"- Total JSON size: {len(json.dumps(sample_json))} characters")
    
    return sample_json

def demo_pydantic_validation(sample_json):
    """Demonstrate Pydantic validation"""
    print("\nPYDANTIC VALIDATION DEMO")
    print("=" * 60)
    
    # Test 1: Valid JSON
    print("Test 1: Validating correct JSON structure")
    try:
        validated = IntentGraphSchema.model_validate(sample_json)
        print("[PASS] Validation PASSED - JSON structure is correct")
        print(f"  - Graph ID: {validated.graph_metadata['graph_id']}")
        print(f"  - Domain: {validated.intent_data.domain_context}")
        print(f"  - Objectives: {len(validated.objectives)}")
        print(f"  - Sentiment: {validated.intent_data.user_sentiment}")
    except Exception as e:
        print(f"[FAIL] Validation FAILED: {str(e)}")
    
    # Test 2: Invalid priority value
    print("\nTest 2: Testing invalid priority value")
    invalid_json = json.loads(json.dumps(sample_json))  # Deep copy
    invalid_json['objectives'][0]['priority'] = 'super_critical'  # Invalid value
    
    try:
        validated = IntentGraphSchema.model_validate(invalid_json)
        print("[FAIL] Validation should have failed but didn't")
    except Exception as e:
        print(f"[PASS] Validation correctly FAILED: {str(e)}")
    
    # Test 3: Missing required field
    print("\nTest 3: Testing missing required field")
    invalid_json2 = json.loads(json.dumps(sample_json))  # Deep copy
    del invalid_json2['objectives'][0]['name']  # Remove required field
    
    try:
        validated = IntentGraphSchema.model_validate(invalid_json2)
        print("[FAIL] Validation should have failed but didn't")
    except Exception as e:
        print(f"[PASS] Validation correctly FAILED: {str(e)}")
    
    # Test 4: Invalid confidence score range
    print("\nTest 4: Testing invalid confidence score range")
    invalid_json3 = json.loads(json.dumps(sample_json))  # Deep copy
    invalid_json3['assumptions'][0]['confidence'] = 1.5  # Out of range (0-1)
    
    try:
        validated = IntentGraphSchema.model_validate(invalid_json3)
        print("[FAIL] Validation should have failed but didn't")
    except Exception as e:
        print(f"[PASS] Validation correctly FAILED: {str(e)}")

def demo_json_import(sample_json):
    """Demonstrate JSON import into intent graph"""
    print("\nJSON IMPORT DEMO")
    print("=" * 60)
    
    # Create new intent graph
    intent = IntentKnowledgeGraph("imported_intent", "Imported from JSON")
    
    print("Before import:")
    print(f"- Nodes: {len(intent.nodes())}")
    print(f"- Edges: {len(intent.edges())}")
    print(f"- Domain context: {intent.domain_context}")
    
    # Import the JSON
    validation_result = intent.validate_and_import_json(sample_json)
    
    print("\nImport results:")
    print(f"- Valid: {validation_result['valid']}")
    print(f"- Errors: {len(validation_result['errors'])}")
    
    if validation_result['valid']:
        stats = validation_result['import_stats']
        print("Import statistics:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
        
        print(f"\nAfter import:")
        print(f"- Nodes: {len(intent.nodes())}")
        print(f"- Edges: {len(intent.edges())}")
        print(f"- Domain context: {intent.domain_context}")
        print(f"- User sentiment: {intent.user_sentiment}")
        print(f"- Clarifying questions: {len(intent.clarifying_questions)}")
        print(f"- Implicit requirements: {len(intent.implicit_requirements)}")
        
        # Show some imported data
        print(f"\nSample imported data:")
        if intent.clarifying_questions:
            print(f"First clarifying question: {intent.clarifying_questions[0]['question']}")
        
        if intent.implicit_requirements:
            print(f"First implicit requirement: {intent.implicit_requirements[0]['requirement']}")
    
    return intent

def demo_refinement_prompt():
    """Demonstrate refinement prompt generation"""
    print("\nREFINEMENT PROMPT DEMO")
    print("=" * 60)
    
    # Sample current JSON (simplified)
    current_json = {
        "objectives": [{"id": "main_goal", "name": "Build mobile app", "priority": "high"}],
        "constraints": [{"id": "budget", "constraint_type": "budget", "value": 15000}]
    }
    
    user_feedback = "Actually, the budget is $20,000 and I also need a web version, not just mobile"
    
    # Generate refinement prompt
    refinement_prompt = IntentKnowledgeGraph.get_llm_refinement_prompt(current_json, user_feedback)
    
    print("User feedback:", user_feedback)
    print(f"Generated refinement prompt length: {len(refinement_prompt)} characters")
    
    # Show first few lines
    prompt_lines = refinement_prompt.split('\n')
    print("Refinement prompt preview:")
    for line in prompt_lines[:15]:
        print(f"  {line}")
    print("  ... (truncated)")

def main():
    """Run all schema and LLM integration demos"""
    print("INTENT KNOWLEDGE GRAPH LLM SCHEMA INTEGRATION")
    print("=" * 70)
    print("Demonstrating Pydantic validation, LLM prompts, and JSON import")
    print("=" * 70)
    
    # Demo 1: Schema template
    schema = demo_schema_template()
    
    # Demo 2: LLM query generation
    test_case = demo_llm_query_generation()
    
    # Demo 3: Sample JSON creation
    sample_json = demo_sample_json_creation()
    
    # Demo 4: Pydantic validation
    demo_pydantic_validation(sample_json)
    
    # Demo 5: JSON import
    imported_intent = demo_json_import(sample_json)
    
    # Demo 6: Refinement prompts
    demo_refinement_prompt()
    
    print("\n" + "=" * 70)
    print("LLM SCHEMA INTEGRATION DEMO COMPLETE!")
    print("=" * 70)
    print("Key Features Demonstrated:")
    print("- Pydantic schema generation for LLM guidance")
    print("- Comprehensive LLM query prompts with examples")
    print("- Strict JSON validation with helpful error messages")
    print("- Automatic import of validated JSON into intent graphs")
    print("- Refinement prompts for iterative LLM improvement")
    print("- Complete digital twin sync with schema enforcement")
    print("=" * 70)

if __name__ == "__main__":
    main()
