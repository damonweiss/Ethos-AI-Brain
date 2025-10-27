#!/usr/bin/env python3
"""
Show Intent Knowledge Graph JSON Structure
Displays the actual JSON format for digital twin sync
"""

import json
from intent_knowledge_graph import IntentKnowledgeGraph

def show_json_structure():
    """Show the actual JSON structure of an intent graph"""
    
    print("INTENT KNOWLEDGE GRAPH JSON STRUCTURE")
    print("=" * 60)
    
    # Create simple intent graph
    intent = IntentKnowledgeGraph("sample_intent", "Sample Intent")
    
    # Add minimal data
    intent.add_objective("main_goal", 
                        name="Primary Objective",
                        priority="high",
                        measurable=True)
    
    intent.add_constraint("budget", 
                         constraint_type="budget",
                         value=5000)
    
    intent.add_stakeholder("user", 
                          name="Primary User",
                          role="decision_maker")
    
    # Add intent derivation data
    intent.set_raw_user_prompt("Sample user prompt for testing")
    intent.set_domain_context("testing")
    intent.add_confidence_score("objectives", 0.8)
    intent.add_ambiguity({
        'element': 'test_element',
        'description': 'Sample ambiguity',
        'severity': 'medium'
    })
    
    # Export to JSON
    json_data = intent.export_to_json(include_metadata=True)
    
    # Pretty print the JSON
    print("EXPORTED JSON STRUCTURE:")
    print("-" * 40)
    print(json.dumps(json_data, indent=2))
    
    print(f"\nJSON SIZE: {len(json.dumps(json_data))} characters")
    print(f"COMPRESSED SIZE: {len(json.dumps(json_data, separators=(',', ':')))} characters")

if __name__ == "__main__":
    show_json_structure()
