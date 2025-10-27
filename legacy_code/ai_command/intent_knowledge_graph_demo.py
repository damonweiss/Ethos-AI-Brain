#!/usr/bin/env python3
"""
Clean Intent Knowledge Graph Demo
Demonstrates intent-specific query capabilities with a wedding planning scenario
"""

from datetime import datetime
from intent_knowledge_graph import IntentKnowledgeGraph
from knowledge_graph_visualizer import MatplotlibKnowledgeGraphVisualizer

def create_wedding_intent():
    """Create a wedding planning intent graph"""
    
    print("CREATING WEDDING INTENT GRAPH")
    print("=" * 50)
    
    # Create intent graph
    intent = IntentKnowledgeGraph("wedding_intent_2024", "Sarah & Mike's Wedding Intent")
    
    # Add primary objectives
    intent.add_objective("memorable_celebration", 
                        name="Create Memorable Celebration",
                        description="Create a beautiful, memorable wedding celebration for family and friends",
                        priority="critical",
                        measurable=True,
                        measurement_method="guest satisfaction survey",
                        target_value=4.5,
                        unit="rating (1-5)")
    
    intent.add_objective("stress_free_planning", 
                        name="Stress-Free Planning Process",
                        description="Ensure the planning process is enjoyable and not overwhelming",
                        priority="high",
                        measurable=True,
                        measurement_method="stress level tracking",
                        target_value=3.0,
                        unit="stress level (1-5)")
    
    intent.add_objective("budget_adherence", 
                        name="Stay Within Budget",
                        description="Complete wedding planning without exceeding the allocated budget",
                        priority="critical",
                        measurable=True,
                        measurement_method="expense tracking",
                        target_value=50000,
                        unit="dollars")
    
    # Add constraints
    intent.add_constraint("budget_limit", 
                         constraint_type="budget",
                         constraint="Total wedding cost must not exceed $50,000",
                         value=50000,
                         flexibility="rigid")
    
    intent.add_constraint("guest_count", 
                         constraint_type="capacity",
                         constraint="Wedding must accommodate 150 guests",
                         value=150,
                         flexibility="somewhat_flexible")
    
    intent.add_constraint("wedding_date", 
                         constraint_type="timeline",
                         constraint="Wedding must be held in December 2024",
                         value="2024-12-15",
                         flexibility="rigid",
                         time_pressure="moderate")
    
    intent.add_constraint("dietary_requirements", 
                         constraint_type="accessibility",
                         constraint="Must provide vegetarian and gluten-free options",
                         flexibility="rigid")
    
    # Add stakeholders
    intent.add_stakeholder("sarah", 
                          name="Sarah (Bride)",
                          role="primary_decision_maker",
                          influence_level="high",
                          support_level="strong")
    
    intent.add_stakeholder("mike", 
                          name="Mike (Groom)",
                          role="primary_decision_maker",
                          influence_level="high",
                          support_level="strong")
    
    intent.add_stakeholder("parents", 
                          name="Both Sets of Parents",
                          role="advisors_contributors",
                          influence_level="medium",
                          support_level="strong")
    
    intent.add_stakeholder("wedding_party", 
                          name="Wedding Party",
                          role="participants",
                          influence_level="low",
                          support_level="strong")
    
    # Add stakeholder needs
    intent.add_node("sarah_vision", 
                   type="stakeholder_need",
                   need="Elegant outdoor ceremony with rustic reception",
                   importance="high",
                   rationale="Always dreamed of outdoor wedding")
    
    intent.add_node("mike_preference", 
                   type="stakeholder_need",
                   need="Great food and music for dancing",
                   importance="high",
                   rationale="Wants guests to have fun and remember the celebration")
    
    intent.add_node("parents_concern", 
                   type="stakeholder_need",
                   need="Weather backup plan and elderly guest accessibility",
                   importance="medium",
                   rationale="Practical concerns for guest comfort and safety")
    
    # Link stakeholders to their needs
    intent.link_stakeholder_need("sarah", "sarah_vision")
    intent.link_stakeholder_need("mike", "mike_preference")
    intent.link_stakeholder_need("parents", "parents_concern")
    
    # Add technical requirements
    intent.add_node("venue_requirements", 
                   type="technical_requirement",
                   description="Venue must support outdoor ceremony with indoor backup",
                   complexity="medium",
                   importance="critical")
    
    intent.add_node("catering_requirements", 
                   type="technical_requirement",
                   description="Catering for 150 guests with dietary restrictions",
                   complexity="medium",
                   importance="high")
    
    # Add context and background
    intent.add_node("previous_weddings", 
                   type="previous_experience",
                   experience="Attended several outdoor weddings, one was rained out",
                   outcome="mixed",
                   lessons_learned=["Always have indoor backup", "Weather contingency is crucial"],
                   relevance="high")
    
    intent.add_node("planning_timeline", 
                   type="timeline_constraint",
                   constraint="6 months to plan everything",
                   time_pressure="moderate",
                   flexibility="somewhat_flexible")
    
    # Add assumptions
    intent.add_node("weather_assumption", 
                   type="assumption",
                   assumption="December weather will be mild enough for outdoor ceremony",
                   confidence=0.7,
                   impact_if_wrong="high",
                   validation_method="Check historical weather data")
    
    intent.add_node("vendor_availability", 
                   type="assumption",
                   assumption="Quality vendors will be available for December wedding",
                   impact_if_wrong="medium",
                   validation_method="Contact vendors early")
    
    print(f"Created intent graph with {len(intent.nodes())} nodes and {len(intent.edges())} edges")
    
    # Show 3D visualization BEFORE auto-linking
    print("\n3D VISUALIZATION - BEFORE AUTO-LINKING:")
    print("=" * 50)
    viz_before = MatplotlibKnowledgeGraphVisualizer(intent)
    viz_before.render_3d_graph(
        figsize=(12, 8),
        show_labels=True,
        positioning_strategy="layered"
    )
    
    # Apply comprehensive auto-linking
    print("\nApplying comprehensive auto-linking...")
    link_stats = intent.comprehensive_auto_link()
    
    print(f"Linking results:")
    print(f"- Initial edges: {link_stats['initial_edges']}")
    print(f"- Final edges: {link_stats['final_edges']}")
    print(f"- Total links added: {link_stats['total_links_added']}")
    print(f"- Edge types: {link_stats['edge_type_distribution']}")
    
    # Simulate agent brain populating intent derivation data
    intent.set_raw_user_prompt("We want to plan Sarah & Mike's wedding for December 2024. Budget is $50k max, expecting 150 guests. Sarah wants outdoor ceremony, Mike wants great food and music. Parents are concerned about weather backup and elderly guest access.")
    intent.set_domain_context("wedding_planning")
    intent.set_user_sentiment("excited")
    
    # Add confidence scores (simulating LLM analysis)
    intent.add_confidence_score("objectives", 0.9)
    intent.add_confidence_score("budget_constraint", 0.95)
    intent.add_confidence_score("timeline_constraint", 0.8)
    intent.add_confidence_score("stakeholder_needs", 0.85)
    
    # Add some ambiguities (simulating LLM detection)
    intent.add_ambiguity({
        'element': 'venue_requirements',
        'description': 'Outdoor ceremony preference conflicts with weather concerns',
        'severity': 'medium',
        'impact': 'Could affect guest comfort and ceremony success'
    })
    
    # Add clarifying questions (simulating LLM generation)
    intent.add_clarifying_question("What is the backup plan if weather doesn't permit outdoor ceremony?", "venue_requirements")
    intent.add_clarifying_question("What specific accessibility needs do elderly guests have?", "guest_accessibility")
    
    # Add implicit requirements (simulating LLM inference)
    intent.add_implicit_requirement({
        'requirement': 'Professional photography and videography services',
        'confidence': 0.8,
        'rationale': 'Weddings typically require professional documentation for memories'
    })
    intent.add_implicit_requirement({
        'requirement': 'Wedding coordination/planning services',
        'confidence': 0.7,
        'rationale': 'Complex event with multiple vendors requires coordination'
    })
    
    # Show 3D visualization AFTER auto-linking
    print("\n3D VISUALIZATION - AFTER AUTO-LINKING:")
    print("=" * 50)
    viz_after = MatplotlibKnowledgeGraphVisualizer(intent)
    viz_after.render_3d_graph(
        figsize=(14, 10),
        show_labels=True,
        positioning_strategy="layered",
        show_edge_labels=True,  # Enable edge labels on hover
        interactive=True        # Enable interactive features
    )
    
    return intent

def demo_intent_queries(intent):
    """Demonstrate intent query capabilities"""
    
    print("\nINTENT QUERY DEMONSTRATIONS")
    print("=" * 50)
    
    # User Requirements Analysis
    print("\n1. USER REQUIREMENTS ANALYSIS:")
    requirements = intent.analyze_user_requirements()
    print(f"   Primary objectives: {len(requirements['primary_objectives'])}")
    for obj in requirements['primary_objectives']:
        print(f"   - {obj['description']} (priority: {obj['priority']})")
    
    print(f"   Success criteria: {len(requirements['success_criteria'])}")
    print(f"   User preferences categories: {list(requirements['user_preferences'].keys())}")
    
    # Constraint Conflicts
    print("\n2. CONSTRAINT CONFLICT ANALYSIS:")
    conflicts = intent.identify_constraint_conflicts()
    print(f"   Found {len(conflicts)} constraint conflicts:")
    for conflict in conflicts:
        print(f"   - {conflict['type']} ({conflict['severity']} severity)")
        print(f"     {conflict['description']}")
    
    # Feasibility Assessment
    print("\n3. FEASIBILITY ASSESSMENT:")
    feasibility = intent.assess_feasibility()
    print(f"   Overall feasibility: {feasibility['overall_feasibility']:.2f}")
    print(f"   Confidence score: {feasibility['confidence_score']:.2f}")
    print("   Feasibility factors:")
    for factor, score in feasibility['feasibility_factors'].items():
        print(f"   - {factor}: {score:.2f}")
    
    if feasibility['recommended_adjustments']:
        print("   Recommended adjustments:")
        for rec in feasibility['recommended_adjustments']:
            print(f"   - {rec}")
    
    # Success Metrics
    print("\n4. SUCCESS METRICS GENERATION:")
    metrics = intent.generate_success_metrics()
    print(f"   Generated {len(metrics)} success metrics:")
    for metric in metrics[:3]:  # Show top 3
        print(f"   - {metric['name']}: {metric['description']}")
        print(f"     Target: {metric['target_value']} {metric['unit']}")
        print(f"     Measurement: {metric['measurement_method']}")
    
    # Decision Criteria
    print("\n5. DECISION CRITERIA EXTRACTION:")
    criteria = intent.extract_decision_criteria()
    print(f"   Risk tolerance: {criteria['risk_tolerance']}")
    print(f"   Non-negotiables: {len(criteria['non_negotiables'])}")
    print(f"   Nice-to-haves: {len(criteria['nice_to_haves'])}")
    print(f"   Decision factors: {list(criteria['decision_factors'].keys())}")
    
    # Stakeholder Mapping
    print("\n6. STAKEHOLDER NEEDS MAPPING:")
    stakeholders = intent.map_stakeholder_needs()
    print(f"   Mapped {len(stakeholders)} stakeholders:")
    for stakeholder_id, stakeholder_list in stakeholders.items():
        for stakeholder in stakeholder_list:
            print(f"   - {stakeholder['name']} ({stakeholder['role']})")
            print(f"     Influence: {stakeholder['influence_level']}, Support: {stakeholder['support_level']}")
            print(f"     Needs: {len(stakeholder['needs'])}")
    
    # Context Capture
    print("\n7. USER CONTEXT CAPTURE:")
    context = intent.capture_user_context()
    print(f"   User expertise level: {context['user_expertise_level']}")
    print(f"   Assumptions: {len(context['assumptions'])}")
    print(f"   External factors: {len(context['external_factors'])}")
    print(f"   Previous experiences: {len(context['previous_experience'])}")
    
    if context['assumptions']:
        print("   Key assumptions:")
        for assumption in context['assumptions'][:2]:  # Show top 2
            print(f"   - {assumption['assumption']} (confidence: {assumption['confidence']})")
    
    # Gap Assessment
    print("\n8. GAP ASSESSMENT:")
    gaps = intent.assess_gaps()
    print(f"   Identified {len(gaps)} gaps:")
    
    gap_types = {}
    for gap in gaps:
        gap_type = gap['gap_type']
        if gap_type not in gap_types:
            gap_types[gap_type] = 0
        gap_types[gap_type] += 1
    
    print(f"   Gap distribution: {dict(gap_types)}")
    
    # Show top 3 most critical gaps
    critical_gaps = [g for g in gaps if g['severity'] in ['critical', 'high']][:3]
    if critical_gaps:
        print("   Critical gaps requiring attention:")
        for gap in critical_gaps:
            print(f"   - {gap['gap_type'].upper()}: {gap['description']}")
            print(f"     Severity: {gap['severity']}, HILT: {gap['hilt_required']}")
            print(f"     Mitigation: {gap['mitigation']}")
            print(f"     Effort: {gap['estimated_effort']}")
            print()
    
    # Intent Derivation Summary
    print("\n9. INTENT DERIVATION SUMMARY:")
    summary = intent.get_intent_summary()
    print(f"   Raw prompt length: {summary['raw_prompt_length']} characters")
    print(f"   Domain context: {summary['domain_context']}")
    print(f"   User sentiment: {summary['user_sentiment']}")
    print(f"   Intent completeness: {summary['completeness_score']:.2f}")
    
    print(f"   Confidence distribution:")
    conf_dist = summary['confidence_distribution']
    if conf_dist['count'] > 0:
        print(f"   - Mean: {conf_dist['mean']:.2f}, Range: {conf_dist['min']:.2f}-{conf_dist['max']:.2f}")
        print(f"   - Elements scored: {conf_dist['count']}")
    
    print(f"   Ambiguities: {summary['ambiguity_count']} identified")
    print(f"   Clarifying questions: {summary['clarifying_questions_count']} ready")
    print(f"   Implicit requirements: {summary['implicit_requirements_count']} detected")
    
    if summary['implicit_requirements_count'] > 0:
        impl_coverage = summary['implicit_coverage']
        print(f"   Implicit coverage: {impl_coverage['coverage_ratio']:.1%} explicitly captured")
    
    # Show clarifying questions
    if intent.clarifying_questions:
        print("\n   Clarifying questions for user:")
        for i, q in enumerate(intent.clarifying_questions, 1):
            print(f"   {i}. {q['question']}")
    
    # Show implicit requirements
    if intent.implicit_requirements:
        print("\n   Detected implicit requirements:")
        for req in intent.implicit_requirements:
            print(f"   - {req['requirement']} (confidence: {req['confidence']:.1f})")
            print(f"     Rationale: {req['rationale']}")
    
    # Graph Traversal Queries
    print("\n10. GRAPH TRAVERSAL QUERIES:")
    print("=" * 40)
    print("Demonstrating the power of the fully connected graph...")
    
    # Query 1: What constraints affect the memorable celebration objective?
    print("\nQuery 1: What constraints affect the 'memorable celebration' objective?")
    memorable_constraints = []
    for source, target, attrs in intent.graph.edges(data=True):
        if (target == 'memorable_celebration' and 
            attrs.get('relationship') == 'constrains'):
            constraint_attrs = intent.get_node_attributes(source)
            memorable_constraints.append({
                'constraint_id': source,
                'constraint_type': constraint_attrs.get('constraint_type', 'unknown'),
                'description': constraint_attrs.get('constraint', 'No description')
            })
    
    print(f"Found {len(memorable_constraints)} constraints:")
    for constraint in memorable_constraints:
        print(f"  - {constraint['constraint_id']} ({constraint['constraint_type']})")
        print(f"    {constraint['description']}")
    
    # Query 2: Which stakeholder needs influence budget planning?
    print("\nQuery 2: Which stakeholder needs influence budget-related objectives?")
    budget_influences = []
    for source, target, attrs in intent.graph.edges(data=True):
        if (attrs.get('relationship') == 'influences' and
            'budget' in target.lower()):
            need_attrs = intent.get_node_attributes(source)
            if need_attrs.get('type') == 'stakeholder_need':
                budget_influences.append({
                    'need_id': source,
                    'need_description': need_attrs.get('need', 'No description'),
                    'target_objective': target
                })
    
    if budget_influences:
        print(f"Found {len(budget_influences)} stakeholder needs affecting budget:")
        for influence in budget_influences:
            print(f"  - {influence['need_description']}")
            print(f"    -> Influences: {influence['target_objective']}")
    else:
        print("  No direct stakeholder need influences on budget objectives found.")
    
    # Query 3: What assumptions impact our timeline constraints?
    print("\nQuery 3: What assumptions impact our timeline constraints?")
    timeline_assumptions = []
    for source, target, attrs in intent.graph.edges(data=True):
        if (attrs.get('relationship') in ['affects', 'impacts'] and
            'timeline' in target.lower()):
            assumption_attrs = intent.get_node_attributes(source)
            if assumption_attrs.get('type') == 'assumption':
                timeline_assumptions.append({
                    'assumption_id': source,
                    'assumption_text': assumption_attrs.get('assumption', 'No description'),
                    'confidence': assumption_attrs.get('confidence', 0.5),
                    'impact_if_wrong': assumption_attrs.get('impact_if_wrong', 'unknown')
                })
    
    print(f"Found {len(timeline_assumptions)} assumptions affecting timeline:")
    for assumption in timeline_assumptions:
        print(f"  - {assumption['assumption_text']}")
        print(f"    Confidence: {assumption['confidence']:.1f}, Impact if wrong: {assumption['impact_if_wrong']}")
    
    # Query 4: Show the dependency chain for stress-free planning
    print("\nQuery 4: What elements are connected to 'stress-free planning'?")
    stress_free_connections = []
    
    # Find all edges connected to stress_free_planning
    for source, target, attrs in intent.graph.edges(data=True):
        relationship = attrs.get('relationship', 'unknown')
        
        if source == 'stress_free_planning':
            target_attrs = intent.get_node_attributes(target)
            stress_free_connections.append({
                'direction': 'outgoing',
                'relationship': relationship,
                'connected_node': target,
                'node_type': target_attrs.get('type', 'unknown'),
                'description': target_attrs.get('name', target_attrs.get('constraint', target))
            })
        elif target == 'stress_free_planning':
            source_attrs = intent.get_node_attributes(source)
            stress_free_connections.append({
                'direction': 'incoming',
                'relationship': relationship,
                'connected_node': source,
                'node_type': source_attrs.get('type', 'unknown'),
                'description': source_attrs.get('name', source_attrs.get('constraint', source))
            })
    
    print(f"Found {len(stress_free_connections)} connections to stress-free planning:")
    for conn in stress_free_connections:
        direction_symbol = "->" if conn['direction'] == 'outgoing' else "<-"
        print(f"  {direction_symbol} {conn['relationship']}: {conn['connected_node']} ({conn['node_type']})")
        print(f"    {conn['description']}")
    
    # Query 5: Find the most connected nodes (highest degree centrality)
    print("\nQuery 5: Most connected nodes in the intent graph:")
    node_degrees = []
    for node_id in intent.nodes():
        degree = intent.degree(node_id)
        node_attrs = intent.get_node_attributes(node_id)
        node_degrees.append({
            'node_id': node_id,
            'degree': degree,
            'type': node_attrs.get('type', 'unknown'),
            'name': node_attrs.get('name', node_attrs.get('constraint', node_id))
        })
    
    # Sort by degree (most connected first)
    node_degrees.sort(key=lambda x: x['degree'], reverse=True)
    
    print("Top 5 most connected nodes:")
    for i, node in enumerate(node_degrees[:5], 1):
        print(f"  {i}. {node['node_id']} ({node['type']}) - {node['degree']} connections")
        print(f"     {node['name']}")
    
    # Query 6: Path analysis - find paths between stakeholders and objectives
    print("\nQuery 6: Paths from stakeholders to primary objectives:")
    stakeholders = [n for n in intent.nodes() 
                   if intent.get_node_attributes(n).get('type') == 'stakeholder']
    objectives = [n for n in intent.nodes() 
                 if intent.get_node_attributes(n).get('type') == 'primary_objective']
    
    import networkx as nx
    paths_found = 0
    for stakeholder in stakeholders[:2]:  # Limit to first 2 stakeholders for brevity
        for objective in objectives[:2]:  # Limit to first 2 objectives
            try:
                # Find shortest path
                path = nx.shortest_path(intent.graph, stakeholder, objective)
                if len(path) > 1:  # Only show if there's actually a path
                    paths_found += 1
                    print(f"  Path from {stakeholder} to {objective}:")
                    
                    # Show the path with relationship labels
                    for i in range(len(path) - 1):
                        source, target = path[i], path[i + 1]
                        edge_attrs = intent.graph.edges[source, target]
                        relationship = edge_attrs.get('relationship', 'connected_to')
                        print(f"    {source} --[{relationship}]--> {target}")
                    print()
            except nx.NetworkXNoPath:
                continue
    
    if paths_found == 0:
        print("  No paths found between stakeholders and objectives (this shouldn't happen with auto-linking!)")
    
    # Query 7: Check specific connections - catering_requirements and planning_timeline
    print("\nQuery 7: Are catering_requirements and planning_timeline connected?")
    catering_connections = []
    timeline_connections = []
    
    # Check catering_requirements connections
    for source, target, attrs in intent.graph.edges(data=True):
        relationship = attrs.get('relationship', 'unknown')
        if source == 'catering_requirements':
            catering_connections.append(f"catering_requirements --[{relationship}]--> {target}")
        elif target == 'catering_requirements':
            catering_connections.append(f"{source} --[{relationship}]--> catering_requirements")
    
    # Check planning_timeline connections
    for source, target, attrs in intent.graph.edges(data=True):
        relationship = attrs.get('relationship', 'unknown')
        if source == 'planning_timeline':
            timeline_connections.append(f"planning_timeline --[{relationship}]--> {target}")
        elif target == 'planning_timeline':
            timeline_connections.append(f"{source} --[{relationship}]--> planning_timeline")
    
    print(f"Catering requirements connections ({len(catering_connections)}):")
    for conn in catering_connections[:5]:  # Show first 5
        print(f"  {conn}")
    
    print(f"Planning timeline connections ({len(timeline_connections)}):")
    for conn in timeline_connections[:5]:  # Show first 5
        print(f"  {conn}")
    
    # Check if they're directly connected
    direct_connection = False
    for source, target, attrs in intent.graph.edges(data=True):
        if ((source == 'catering_requirements' and target == 'planning_timeline') or
            (source == 'planning_timeline' and target == 'catering_requirements')):
            relationship = attrs.get('relationship', 'unknown')
            print(f"DIRECT CONNECTION FOUND: {source} --[{relationship}]--> {target}")
            direct_connection = True
    
    if not direct_connection:
        print("No direct connection between catering_requirements and planning_timeline found.")
        print("This suggests our auto-linking needs improvement for timeline->technical requirement connections.")
    
    print(f"\nGraph traversal queries complete! Analyzed {len(intent.nodes())} nodes and {len(intent.edges())} edges.")
    print("The comprehensive auto-linking enables rich graph analysis and relationship discovery.")

def main():
    """Main demo function"""
    
    print("INTENT KNOWLEDGE GRAPH DEMONSTRATION")
    print("=" * 60)
    print("Demonstrating intent-specific query capabilities")
    print("with a wedding planning scenario")
    print("=" * 60)
    
    # Create the wedding intent graph
    intent = create_wedding_intent()
    
    # Demonstrate query capabilities
    demo_intent_queries(intent)
    
    print("\nINTENT DEMO COMPLETE!")
    print("=" * 60)
    print("Key Takeaways:")
    print("- Intent graphs capture user goals, constraints, and context")
    print("- Feasibility assessment helps validate requirements early")
    print("- Constraint conflict detection prevents planning issues")
    print("- Stakeholder mapping ensures all needs are considered")
    print("- Success metrics provide measurable outcomes")
    print("- Gap assessment identifies missing elements (HILT, resources, knowledge)")
    print("- Enum-based gap classification provides structured analysis")
    print("- Intent derivation data structures store LLM analysis results")
    print("- Mathematical analysis methods provide confidence and completeness scores")
    print("- Clear separation: LLM processing in agent brain, data storage in graph")
    print("=" * 60)

if __name__ == "__main__":
    main()
