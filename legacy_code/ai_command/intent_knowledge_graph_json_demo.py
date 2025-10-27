#!/usr/bin/env python3
"""
Intent Knowledge Graph JSON Export/Update Demo
Demonstrates digital twin sync capabilities with intent-specific data
"""

import json
from intent_knowledge_graph import IntentKnowledgeGraph

def demo_intent_json_workflow():
    """Demonstrate complete JSON workflow with intent graph"""
    
    print("INTENT KNOWLEDGE GRAPH JSON DEMO")
    print("=" * 60)
    print("Simulating Agent Brain <-> Intent Graph digital twin sync")
    print("=" * 60)
    
    # Step 1: Create initial intent graph (simulating user input)
    print("\n1. CREATING INITIAL INTENT GRAPH")
    print("-" * 40)
    
    intent = IntentKnowledgeGraph("mobile_app_intent", "Bakery Mobile App")
    
    # Add initial objectives
    intent.add_objective("online_ordering", 
                        name="Enable Online Cake Orders",
                        description="Allow customers to order custom cakes through mobile app",
                        priority="critical",
                        measurable=True,
                        measurement_method="order completion rate",
                        target_value=0.85)
    
    intent.add_objective("payment_processing", 
                        name="Secure Payment Processing", 
                        description="Handle credit card payments securely",
                        priority="critical",
                        measurable=True)
    
    # Add constraints
    intent.add_constraint("budget_limit",
                         constraint_type="budget",
                         constraint="Total development cost must not exceed $15,000",
                         value=15000,
                         flexibility="somewhat_flexible")
    
    intent.add_constraint("timeline_deadline",
                         constraint_type="timeline", 
                         constraint="Must be completed before holiday season (3 months)",
                         time_pressure="tight",
                         flexibility="rigid")
    
    # Add stakeholders
    intent.add_stakeholder("bakery_owner",
                          name="Bakery Owner",
                          role="primary_decision_maker", 
                          influence_level="high",
                          support_level="strong")
    
    intent.add_stakeholder("business_partner",
                          name="Business Partner Sarah",
                          role="advisor_stakeholder",
                          influence_level="medium", 
                          support_level="cautious")
    
    # Simulate agent brain setting intent derivation data
    intent.set_raw_user_prompt("I need to create a mobile app for my small bakery. I want customers to be able to order custom cakes online and pick them up. My budget is around $15,000, and I need it done before the holiday season in 3 months.")
    intent.set_domain_context("mobile_app_development")
    intent.set_user_sentiment("urgent")
    
    # Add confidence scores
    intent.add_confidence_score("objectives", 0.9)
    intent.add_confidence_score("budget_constraint", 0.8)
    intent.add_confidence_score("timeline_constraint", 0.7)
    
    # Add ambiguities
    intent.add_ambiguity({
        'element': 'technical_complexity',
        'description': 'User is not tech-savvy but needs complex payment processing',
        'severity': 'high',
        'impact': 'May need simplified interface or external help'
    })
    
    # Add clarifying questions
    intent.add_clarifying_question("What payment methods do you want to support?", "payment_processing")
    intent.add_clarifying_question("Do you have existing POS system to integrate with?", "technical_integration")
    
    # Add implicit requirements
    intent.add_implicit_requirement({
        'requirement': 'Mobile-responsive design for customer convenience',
        'confidence': 0.9,
        'rationale': 'Customers will primarily use mobile devices for ordering'
    })
    
    intent.add_implicit_requirement({
        'requirement': 'Order notification system for bakery staff',
        'confidence': 0.8,
        'rationale': 'Bakery needs to know when orders are placed'
    })
    
    # Auto-link relationships using helper functions
    print("\nAuto-linking relationships...")
    link_stats = intent.auto_link_graph_relationships()
    print(f"Relationship linking stats: {link_stats}")
    
    print(f"\nInitial intent graph: {len(intent.nodes())} nodes, {len(intent.edges())} edges")
    print(f"Domain context: {intent.domain_context}")
    print(f"User sentiment: {intent.user_sentiment}")
    print(f"Ambiguities: {len(intent.ambiguities)}")
    print(f"Clarifying questions: {len(intent.clarifying_questions)}")
    print(f"Implicit requirements: {len(intent.implicit_requirements)}")
    
    # Show the relationships that were created
    print(f"\nRelationships created:")
    for source, target, attrs in intent.graph.edges(data=True):
        relationship = attrs.get('relationship', 'unknown')
        print(f"  {source} --[{relationship}]--> {target}")
    
    # Step 2: Export to JSON (simulating brain querying current state)
    print("\n2. EXPORTING INTENT STATE TO JSON")
    print("-" * 40)
    
    exported_json = intent.export_to_json(include_metadata=True)
    
    print(f"Exported JSON structure:")
    print(f"- Graph data: {len(exported_json['graph_data']['nodes'])} nodes, {len(exported_json['graph_data']['links'])} edges")
    print(f"- Graph type: {exported_json['graph_type']}")
    print(f"- Graph ID: {exported_json['graph_id']}")
    print(f"- Export timestamp: {exported_json['export_timestamp']}")
    print(f"- Total JSON size: {len(json.dumps(exported_json))} characters")
    
    # Show sample of the JSON structure
    print(f"\nSample node data:")
    for i, node in enumerate(exported_json['graph_data']['nodes'][:2]):
        print(f"  Node {i+1}: {node['id']} (type: {node.get('type', 'unknown')})")
    
    # Step 3: Simulate LLM analysis and updates
    print("\n3. SIMULATING LLM ANALYSIS AND UPDATES")
    print("-" * 40)
    
    # Simulate LLM providing additional insights and updates
    llm_updates = {
        'graph_data': {
            'nodes': [
                # New requirement identified by LLM
                {
                    'id': 'inventory_integration',
                    'type': 'technical_requirement',
                    'name': 'Inventory Integration',
                    'description': 'Integration with bakery inventory system to show available items',
                    'complexity': 'medium',
                    'importance': 'high'
                },
                # Updated constraint with more detail
                {
                    'id': 'budget_limit',
                    'type': 'constraint',
                    'constraint_type': 'budget',
                    'constraint': 'Total development cost must not exceed $15,000',
                    'value': 15000,
                    'flexibility': 'somewhat_flexible',
                    'budget_breakdown': {
                        'development': 8000,
                        'design': 2000,
                        'testing': 1500,
                        'deployment': 1000,
                        'contingency': 2500
                    }
                }
            ],
            'links': [
                # New relationship identified by LLM
                {
                    'source': 'online_ordering',
                    'target': 'inventory_integration',
                    'relationship': 'requires'
                }
            ]
        }
    }
    
    print("LLM identified:")
    print("- New technical requirement: inventory_integration")
    print("- Updated budget constraint with detailed breakdown")
    print("- New dependency: online_ordering requires inventory_integration")
    
    # Step 4: Apply LLM updates
    print("\n4. APPLYING LLM UPDATES TO INTENT GRAPH")
    print("-" * 40)
    
    update_stats = intent.update_from_json(llm_updates, merge_mode="update")
    
    print(f"Update statistics:")
    print(f"- Nodes added: {update_stats['nodes_added']}")
    print(f"- Nodes updated: {update_stats['nodes_updated']}")
    print(f"- Edges added: {update_stats['edges_added']}")
    print(f"- Edges updated: {update_stats['edges_updated']}")
    
    print(f"\nUpdated intent graph: {len(intent.nodes())} nodes, {len(intent.edges())} edges")
    
    # Verify the budget breakdown was added
    budget_node = intent.get_node_attributes('budget_limit')
    if 'budget_breakdown' in budget_node:
        print(f"Budget breakdown successfully added: {budget_node['budget_breakdown']}")
    
    # Step 5: Run analysis on updated graph
    print("\n5. ANALYZING UPDATED INTENT GRAPH")
    print("-" * 40)
    
    # Gap assessment
    gaps = intent.assess_gaps()
    print(f"Gap assessment: {len(gaps)} gaps identified")
    if gaps:
        critical_gaps = [g for g in gaps if g['severity'] in ['critical', 'high']]
        print(f"Critical gaps: {len(critical_gaps)}")
        for gap in critical_gaps[:2]:
            print(f"  - {gap['gap_type'].upper()}: {gap['description']}")
    
    # Feasibility assessment
    feasibility = intent.assess_feasibility()
    print(f"Feasibility score: {feasibility['overall_feasibility']:.2f}")
    print(f"Feasibility factors: technical={feasibility['feasibility_factors']['technical']:.2f}, "
          f"resource={feasibility['feasibility_factors']['resource']:.2f}")
    
    # Intent completeness
    completeness = intent.get_intent_completeness_score()
    print(f"Intent completeness: {completeness:.2f}")
    
    # Step 6: Export final state
    print("\n6. EXPORTING FINAL INTENT STATE")
    print("-" * 40)
    
    final_json = intent.export_to_json(include_metadata=True)
    
    print(f"Final JSON structure:")
    print(f"- Graph data: {len(final_json['graph_data']['nodes'])} nodes, {len(final_json['graph_data']['links'])} edges")
    print(f"- Total JSON size: {len(json.dumps(final_json))} characters")
    
    # State summary
    summary = intent.get_state_summary()
    print(f"\nFinal state summary:")
    print(f"- Node types: {summary['node_types']}")
    print(f"- Edge types: {summary['edge_types']}")
    print(f"- Last modified: {summary['last_modified']}")
    
    # Step 7: Demonstrate partial updates (simulating iterative LLM refinement)
    print("\n7. DEMONSTRATING PARTIAL UPDATES")
    print("-" * 40)
    
    # Simulate user providing clarification
    clarification_update = {
        'graph_data': {
            'nodes': [
                # Update payment processing with specific details
                {
                    'id': 'payment_processing',
                    'type': 'primary_objective',
                    'name': 'Secure Payment Processing',
                    'description': 'Handle credit card payments securely using Stripe integration',
                    'priority': 'critical',
                    'measurable': True,
                    'measurement_method': 'payment success rate',
                    'target_value': 0.98,
                    'payment_methods': ['credit_card', 'debit_card', 'apple_pay'],
                    'integration_complexity': 'medium'
                }
            ],
            'links': []
        }
    }
    
    print("Applying user clarification about payment methods...")
    clarification_stats = intent.update_from_json(clarification_update, merge_mode="update")
    print(f"Clarification update: {clarification_stats['nodes_updated']} nodes updated")
    
    # Verify payment methods were added
    payment_node = intent.get_node_attributes('payment_processing')
    if 'payment_methods' in payment_node:
        print(f"Payment methods added: {payment_node['payment_methods']}")
    
    print("\n" + "=" * 60)
    print("INTENT JSON DEMO COMPLETE!")
    print("=" * 60)
    print("Key Demonstrations:")
    print("- Intent graph creation with comprehensive data")
    print("- JSON export with NetworkX serialization")
    print("- LLM-style updates with merge mode")
    print("- Partial updates for iterative refinement")
    print("- Analysis methods on updated graph")
    print("- Complete digital twin sync workflow")
    print("=" * 60)

if __name__ == "__main__":
    demo_intent_json_workflow()
