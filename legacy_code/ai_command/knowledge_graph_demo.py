#!/usr/bin/env python3
"""
Knowledge Graph Demo - Demonstration of cross-graph functionality and visualizations
Separated from main knowledge_graph.py for better organization
"""

import asyncio
from knowledge_graph import KnowledgeGraph, GraphType
from knowledge_graph_visualizer import ASCIIKnowledgeGraphVisualizer, MatplotlibKnowledgeGraphVisualizer, BrainKnowledgeGraphVisualizer
from knowledge_graph_style import StylePresets, ColorScheme
from knowledge_graph_brain import BrainKnowledgeGraph

def demo_wedding_planning_agents():
    """Demo ZeroAgent wedding planning with complete agent hierarchy"""
    
    print("💒 WEDDING PLANNING ZEROAGENT DEMO")
    print("=" * 60)
    
    # 1. MISSION GRAPH - Overall Wedding Planning Mission
    mission_graph = KnowledgeGraph(GraphType.MISSION, "wedding_mission_2024")
    
    mission_graph.add_node("wedding_objective", 
                          type="mission", 
                          content="Plan perfect wedding for Sarah & Mike - June 15, 2024",
                          status="active",
                          priority="critical",
                          budget=50000,
                          guest_count=150)
    
    mission_graph.add_node("coordination_hub", 
                          type="coordination", 
                          role="orchestrator",
                          active_agents=5,
                          status="coordinating")
    
    mission_graph.add_node("timeline_requirements", 
                          type="requirement", 
                          details=["6 months planning", "June 15 deadline", "weather contingency"],
                          priority="critical")
    
    mission_graph.add_node("budget_requirements", 
                          type="requirement", 
                          details=["$50k total", "$15k venue", "$8k catering", "$5k photography"],
                          priority="high")
    
    mission_graph.add_node("mission_status", 
                          type="status", 
                          progress=0.4,
                          phase="planning",
                          health="on_track",
                          days_remaining=180)
    
    # Mission internal connections
    mission_graph.add_edge("wedding_objective", "coordination_hub", relationship="coordinates")
    mission_graph.add_edge("wedding_objective", "timeline_requirements", relationship="requires")
    mission_graph.add_edge("wedding_objective", "budget_requirements", relationship="requires")
    mission_graph.add_edge("coordination_hub", "mission_status", relationship="monitors")
    mission_graph.add_edge("timeline_requirements", "budget_requirements", relationship="constrains")
    
    # 2. ZEROAGENT TACTICAL GRAPH - Main AI Coordinator
    zero_agent_graph = KnowledgeGraph(GraphType.TACTICAL, "zero_agent_coordinator")
    
    zero_agent_graph.add_node("current_task", 
                             type="task", 
                             task="coordinate_wedding_planning",
                             status="in_progress",
                             progress=0.4,
                             priority="critical")
    
    zero_agent_graph.add_node("agent_status", 
                             type="status", 
                             health="optimal",
                             load=0.6,
                             capabilities=["coordination", "planning", "resource_allocation"])
    
    zero_agent_graph.add_node("decision_engine", 
                             type="coordination", 
                             decisions_made=47,
                             pending_decisions=12,
                             confidence=0.85)
    
    zero_agent_graph.add_node("progress_tracker", 
                             type="status", 
                             overall_progress=0.4,
                             milestones_completed=8,
                             milestones_total=20)
    
    # ZeroAgent internal connections
    zero_agent_graph.add_edge("agent_status", "current_task", relationship="executing")
    zero_agent_graph.add_edge("decision_engine", "current_task", relationship="guides")
    zero_agent_graph.add_edge("current_task", "progress_tracker", relationship="updates")
    
    # 3. VENUE SPECIALIST SUB-AGENT
    venue_agent_graph = KnowledgeGraph(GraphType.TACTICAL, "venue_specialist_emma")
    
    venue_agent_graph.add_node("venue_search", 
                              type="task", 
                              task="find_perfect_venue",
                              status="completed",
                              venues_evaluated=25,
                              shortlist=3)
    
    venue_agent_graph.add_node("agent_status", 
                              type="status", 
                              health="excellent",
                              load=0.3,
                              specialization="venue_coordination")
    
    venue_agent_graph.add_node("venue_recommendations", 
                              type="output", 
                              top_venues=["Rosewood Manor", "Garden Pavilion", "Historic Estate"],
                              confidence=0.92,
                              booking_status="pending_decision")
    
    venue_agent_graph.add_node("site_visits", 
                              type="task", 
                              task="coordinate_site_visits",
                              status="scheduled",
                              visits_planned=3)
    
    # Venue agent connections
    venue_agent_graph.add_edge("agent_status", "venue_search", relationship="completed")
    venue_agent_graph.add_edge("venue_search", "venue_recommendations", relationship="produces")
    venue_agent_graph.add_edge("venue_search", "site_visits", relationship="triggers")
    
    # 4. CATERING SPECIALIST SUB-AGENT
    catering_agent_graph = KnowledgeGraph(GraphType.TACTICAL, "catering_specialist_chef_antonio")
    
    catering_agent_graph.add_node("menu_planning", 
                                 type="task", 
                                 task="design_wedding_menu",
                                 status="in_progress",
                                 dietary_restrictions=["vegetarian", "gluten_free", "nut_allergy"],
                                 progress=0.7)
    
    catering_agent_graph.add_node("agent_status", 
                                 type="status", 
                                 health="good",
                                 load=0.8,
                                 specialization="culinary_coordination")
    
    catering_agent_graph.add_node("vendor_coordination", 
                                 type="task", 
                                 task="coordinate_catering_vendors",
                                 status="active",
                                 vendors_contacted=8,
                                 quotes_received=5)
    
    catering_agent_graph.add_node("menu_proposals", 
                                 type="output", 
                                 proposals=["Mediterranean Feast", "Farm-to-Table", "Classic Elegance"],
                                 tasting_scheduled=True)
    
    # Catering agent connections
    catering_agent_graph.add_edge("agent_status", "menu_planning", relationship="executing")
    catering_agent_graph.add_edge("agent_status", "vendor_coordination", relationship="managing")
    catering_agent_graph.add_edge("menu_planning", "menu_proposals", relationship="produces")
    catering_agent_graph.add_edge("vendor_coordination", "menu_proposals", relationship="supports")
    
    # 5. PHOTOGRAPHY SPECIALIST SUB-AGENT
    photo_agent_graph = KnowledgeGraph(GraphType.TACTICAL, "photography_specialist_alex")
    
    photo_agent_graph.add_node("photographer_search", 
                              type="task", 
                              task="find_wedding_photographer",
                              status="completed",
                              photographers_reviewed=15,
                              interviews_conducted=5)
    
    photo_agent_graph.add_node("agent_status", 
                              type="status", 
                              health="excellent",
                              load=0.4,
                              specialization="visual_coordination")
    
    photo_agent_graph.add_node("photo_timeline", 
                              type="task", 
                              task="create_photo_schedule",
                              status="in_progress",
                              key_moments=["ceremony", "reception", "portraits", "candids"])
    
    photo_agent_graph.add_node("photographer_selection", 
                              type="output", 
                              selected_photographer="Luna Photography Studio",
                              contract_status="signed",
                              engagement_shoot_scheduled=True)
    
    # Photography agent connections
    photo_agent_graph.add_edge("agent_status", "photographer_search", relationship="completed")
    photo_agent_graph.add_edge("photographer_search", "photographer_selection", relationship="produces")
    photo_agent_graph.add_edge("photographer_selection", "photo_timeline", relationship="enables")
    
    # 6. LOCAL RAG - Guest List & Personal Info
    local_rag_graph = KnowledgeGraph(GraphType.LOCAL_RAG, "guest_list_local_rag")
    
    local_rag_graph.add_node("guest_database", 
                            type="resource", 
                            total_guests=150,
                            confirmed_rsvp=89,
                            pending_rsvp=61,
                            dietary_restrictions=23)
    
    local_rag_graph.add_node("family_contacts", 
                            type="resource", 
                            bride_family=45,
                            groom_family=38,
                            mutual_friends=67,
                            plus_ones=34)
    
    local_rag_graph.add_node("invitation_tracker", 
                            type="status", 
                            invitations_sent=150,
                            responses_received=89,
                            follow_ups_needed=25)
    
    local_rag_graph.add_node("seating_algorithm", 
                            type="coordination", 
                            tables_planned=15,
                            seating_conflicts_resolved=8,
                            optimization_score=0.87)
    
    # Local RAG connections
    local_rag_graph.add_edge("guest_database", "family_contacts", relationship="organizes")
    local_rag_graph.add_edge("guest_database", "invitation_tracker", relationship="tracks")
    local_rag_graph.add_edge("family_contacts", "seating_algorithm", relationship="informs")
    
    # 7. CLOUD RAG - Venue & Vendor Database
    cloud_rag_graph = KnowledgeGraph(GraphType.CLOUD_RAG, "venue_vendor_cloud_rag")
    
    cloud_rag_graph.add_node("venue_database", 
                            type="resource", 
                            venues_indexed=500,
                            location_radius="50_miles",
                            price_range="$5k-$25k",
                            availability_checked=True)
    
    cloud_rag_graph.add_node("vendor_network", 
                            type="resource", 
                            photographers=150,
                            caterers=89,
                            florists=67,
                            musicians=45,
                            ratings_available=True)
    
    cloud_rag_graph.add_node("market_intelligence", 
                            type="coordination", 
                            seasonal_pricing=True,
                            availability_trends=True,
                            quality_scores=True)
    
    cloud_rag_graph.add_node("recommendation_engine", 
                            type="coordination", 
                            ml_model_version="v2.3",
                            confidence_threshold=0.8,
                            personalization_active=True)
    
    # Cloud RAG connections
    cloud_rag_graph.add_edge("venue_database", "market_intelligence", relationship="informs")
    cloud_rag_graph.add_edge("vendor_network", "market_intelligence", relationship="informs")
    cloud_rag_graph.add_edge("market_intelligence", "recommendation_engine", relationship="powers")
    
    # Create second Local RAG - Budget & Preferences
    budget_rag_graph = KnowledgeGraph(GraphType.LOCAL_RAG, "budget_preferences_local_rag")
    budget_rag_graph.add_node("budget_tracker", type="data", status="active", 
                             budget_total=50000, spent=12000, remaining=38000)
    budget_rag_graph.add_node("preference_engine", type="algorithm", 
                             learning_active=True, confidence_threshold=0.85)
    budget_rag_graph.add_node("vendor_preferences", type="data", 
                             preferred_style="modern", dietary_restrictions=["vegetarian", "gluten_free"])
    budget_rag_graph.add_node("timeline_constraints", type="constraint",
                             wedding_date="2024-06-15", booking_deadline="2024-03-01")
    
    # Budget RAG connections
    budget_rag_graph.add_edge("budget_tracker", "preference_engine", relationship="informs")
    budget_rag_graph.add_edge("vendor_preferences", "preference_engine", relationship="configures")
    budget_rag_graph.add_edge("timeline_constraints", "preference_engine", relationship="constrains")
    
    print("📊 Created complete wedding planning agent network:")
    print(f"   - Mission Graph: {len(mission_graph.nodes())} nodes (Wedding Objective)")
    print(f"   - Venue Specialist: {len(venue_agent_graph.nodes())} nodes (Emma)")
    print(f"   - Catering Specialist: {len(catering_agent_graph.nodes())} nodes (Chef Antonio)")
    print(f"   - Photography Specialist: {len(photo_agent_graph.nodes())} nodes (Alex)")
    print(f"   - Guest List RAG: {len(local_rag_graph.nodes())} nodes (Guest List)")
    print(f"   - Budget/Preferences RAG: {len(budget_rag_graph.nodes())} nodes (Budget & Preferences)")
    print(f"   - Cloud RAG: {len(cloud_rag_graph.nodes())} nodes (Venue/Vendor DB)")
    
    # Create cross-agent connections
    print("\n🔗 Creating cross-agent coordination connections...")
    
    # Mission -> Direct coordination with specialists (no ZeroAgent)
    mission_graph.add_external_reference("coordination_hub", "venue_specialist_emma", "agent_status",
                                        relationship="coordinates", agent_type="venue_specialist")
    mission_graph.add_external_reference("coordination_hub", "catering_specialist_chef_antonio", "agent_status",
                                        relationship="coordinates", agent_type="catering_specialist")
    mission_graph.add_external_reference("coordination_hub", "photography_specialist_alex", "agent_status",
                                        relationship="coordinates", agent_type="photography_specialist")
    
    # Mission delegates tasks directly to specialists
    mission_graph.add_external_reference("wedding_objective", "venue_specialist_emma", "venue_search",
                                        relationship="delegates_to", task_type="venue_coordination")
    mission_graph.add_external_reference("wedding_objective", "catering_specialist_chef_antonio", "menu_planning",
                                        relationship="delegates_to", task_type="catering_coordination")
    mission_graph.add_external_reference("wedding_objective", "photography_specialist_alex", "photographer_search",
                                        relationship="delegates_to", task_type="photography_coordination")
    
    # Sub-agents -> RAG systems
    venue_agent_graph.add_external_reference("venue_search", "venue_vendor_cloud_rag", "venue_database",
                                            relationship="queries", query_type="venue_search")
    catering_agent_graph.add_external_reference("vendor_coordination", "venue_vendor_cloud_rag", "vendor_network",
                                               relationship="queries", query_type="catering_vendors")
    photo_agent_graph.add_external_reference("photographer_search", "venue_vendor_cloud_rag", "vendor_network",
                                            relationship="queries", query_type="photographer_search")
    
    # RAG -> Guest coordination
    local_rag_graph.add_external_reference("seating_algorithm", "venue_specialist_emma", "venue_recommendations",
                                          relationship="depends_on", dependency_type="venue_capacity")
    
    # Budget RAG -> All specialists for budget constraints
    budget_rag_graph.add_external_reference("budget_tracker", "venue_specialist_emma", "venue_search",
                                           relationship="constrains", constraint_type="budget_limit")
    budget_rag_graph.add_external_reference("budget_tracker", "catering_specialist_chef_antonio", "menu_planning",
                                           relationship="constrains", constraint_type="budget_limit")
    budget_rag_graph.add_external_reference("preference_engine", "photography_specialist_alex", "photographer_search",
                                           relationship="guides", preference_type="style_matching")
    
    print("✅ Cross-agent coordination network established")
    
    return (mission_graph, venue_agent_graph, catering_agent_graph, 
            photo_agent_graph, local_rag_graph, budget_rag_graph, cloud_rag_graph)

def demo_knowledge_graph():
    """Demo the knowledge graph base class with cross-graph functionality"""
    
    print("🧠 KNOWLEDGE GRAPH CROSS-GRAPH DEMO")
    print("=" * 60)
    
    # Create AgentZero's mission knowledge graph (5 nodes)
    agent_zero_graph = KnowledgeGraph(GraphType.MISSION, "agent_zero_mission")
    
    agent_zero_graph.add_node("mission_objective", 
                             type="mission", 
                             content="Build secure trading API",
                             status="active",
                             priority="high")
    
    agent_zero_graph.add_node("coordination_hub", 
                             type="coordination", 
                             role="orchestrator",
                             active_agents=2)
    
    agent_zero_graph.add_node("security_requirements", 
                             type="requirement", 
                             details=["OAuth2", "TLS", "audit_logs"],
                             priority="critical")
    
    agent_zero_graph.add_node("performance_requirements", 
                             type="requirement", 
                             details=["low_latency", "high_throughput"],
                             priority="high")
    
    agent_zero_graph.add_node("mission_status", 
                             type="status", 
                             progress=0.6,
                             phase="execution",
                             health="healthy")
    
    # Add internal connections within AgentZero graph
    agent_zero_graph.add_edge("mission_objective", "coordination_hub", 
                             relationship="coordinates", strength=0.9)
    agent_zero_graph.add_edge("mission_objective", "security_requirements", 
                             relationship="requires", strength=0.8)
    agent_zero_graph.add_edge("mission_objective", "performance_requirements", 
                             relationship="requires", strength=0.7)
    agent_zero_graph.add_edge("coordination_hub", "mission_status", 
                             relationship="monitors", strength=0.9)
    agent_zero_graph.add_edge("security_requirements", "performance_requirements", 
                             relationship="constrains", strength=0.5)
    
    # Create DataAnalyst sub-agent's tactical graph (4 nodes)
    data_analyst_graph = KnowledgeGraph(GraphType.TACTICAL, "data_analyst_sarah")
    
    data_analyst_graph.add_node("current_task", 
                                type="task", 
                                task="market_data_analysis",
                                status="in_progress",
                                progress=0.7,
                                estimated_completion="2024-01-15T14:30:00")
    
    data_analyst_graph.add_node("agent_status", 
                                type="status", 
                                health="healthy",
                                load=0.7,
                                capabilities=["data_analysis", "market_research"])
    
    data_analyst_graph.add_node("data_sources", 
                                type="resource", 
                                sources=["bloomberg_api", "yahoo_finance", "internal_db"],
                                status="connected")
    
    data_analyst_graph.add_node("analysis_results", 
                                type="output", 
                                insights=["trend_bullish", "volatility_high"],
                                confidence=0.85,
                                timestamp="2024-01-15T12:30:00")
    
    # Add internal connections within DataAnalyst graph
    data_analyst_graph.add_edge("agent_status", "current_task", 
                               relationship="executing", strength=0.8)
    data_analyst_graph.add_edge("data_sources", "current_task", 
                               relationship="feeds", strength=0.9)
    data_analyst_graph.add_edge("current_task", "analysis_results", 
                               relationship="produces", strength=0.7)
    
    # Create SecurityExpert sub-agent's tactical graph (5 nodes)
    security_expert_graph = KnowledgeGraph(GraphType.TACTICAL, "security_expert_marcus")
    
    security_expert_graph.add_node("security_review", 
                                   type="task", 
                                   task="api_security_assessment",
                                   status="completed",
                                   findings=["oauth2_required", "rate_limiting_needed"])
    
    security_expert_graph.add_node("agent_status", 
                                   type="status", 
                                   health="healthy",
                                   load=0.2,
                                   capabilities=["security_analysis", "compliance"])
    
    security_expert_graph.add_node("vulnerability_scan", 
                                   type="task", 
                                   scan_type="penetration_test",
                                   status="scheduled",
                                   priority="high")
    
    security_expert_graph.add_node("compliance_check", 
                                   type="task", 
                                   standards=["SOX", "PCI-DSS", "GDPR"],
                                   status="in_progress",
                                   completion=0.4)
    
    security_expert_graph.add_node("security_recommendations", 
                                   type="output", 
                                   recommendations=["implement_2fa", "api_rate_limiting", "audit_logging"],
                                   priority="critical",
                                   approved=True)
    
    # Add internal connections within SecurityExpert graph
    security_expert_graph.add_edge("agent_status", "security_review", 
                                  relationship="completed", strength=1.0)
    security_expert_graph.add_edge("security_review", "vulnerability_scan", 
                                  relationship="triggers", strength=0.8)
    security_expert_graph.add_edge("security_review", "compliance_check", 
                                  relationship="informs", strength=0.7)
    security_expert_graph.add_edge("security_review", "security_recommendations", 
                                  relationship="produces", strength=0.9)
    security_expert_graph.add_edge("compliance_check", "security_recommendations", 
                                  relationship="influences", strength=0.6)
    
    print("📊 Created 3 knowledge graphs:")
    print(f"   - AgentZero Mission: {len(agent_zero_graph.nodes())} nodes")
    print(f"   - DataAnalyst Tactical: {len(data_analyst_graph.nodes())} nodes")
    print(f"   - SecurityExpert Tactical: {len(security_expert_graph.nodes())} nodes")
    
    # Create cross-graph connections (AgentZero hive-mind into sub-agents)
    print("\n🔗 Creating cross-graph connections...")
    
    # AgentZero references sub-agent status
    agent_zero_graph.add_external_reference(
        "coordination_hub", 
        "data_analyst_sarah", 
        "agent_status",
        relationship="monitors",
        monitoring_type="real_time"
    )
    
    agent_zero_graph.add_external_reference(
        "coordination_hub", 
        "security_expert_marcus", 
        "agent_status",
        relationship="monitors",
        monitoring_type="real_time"
    )
    
    # AgentZero references sub-agent tasks
    agent_zero_graph.add_external_reference(
        "mission_objective", 
        "data_analyst_sarah", 
        "current_task",
        relationship="depends_on",
        dependency_type="critical"
    )
    
    agent_zero_graph.add_external_reference(
        "mission_objective", 
        "security_expert_marcus", 
        "security_review",
        relationship="depends_on",
        dependency_type="blocking"
    )
    
    print("✅ Cross-graph connections established")
    
    # Show graph visualizations using new visualizer classes
    print("\n📊 GRAPH VISUALIZATIONS:")
    print("\n1. AgentZero Mission Graph:")
    ascii_viz = ASCIIKnowledgeGraphVisualizer(agent_zero_graph, StylePresets.professional())
    ascii_viz.print_graph(show_attributes=True)
    
    print("\n2. DataAnalyst Tactical Graph:")
    ascii_viz2 = ASCIIKnowledgeGraphVisualizer(data_analyst_graph, StylePresets.minimal())
    ascii_viz2.print_graph(show_attributes=True, show_external_refs=False)
    
    # Demo hive-mind queries (AgentZero can see into all sub-agent graphs)
    print("\n🧠 HIVE-MIND QUERIES:")
    
    # Query all agent status across the network
    print("\n1. Query all agent status:")
    status_results = agent_zero_graph.hive_mind_query(type="status")
    for graph_id, nodes in status_results.items():
        for node in nodes:
            print(f"   {graph_id}: {node['node_id']} - Health: {node['attributes'].get('health', 'unknown')}, Load: {node['attributes'].get('load', 'unknown')}")
    
    # Query all active tasks
    print("\n2. Query all active tasks:")
    task_results = agent_zero_graph.hive_mind_query(type="task")
    for graph_id, nodes in task_results.items():
        for node in nodes:
            print(f"   {graph_id}: {node['attributes'].get('task', 'unknown')} - Status: {node['attributes'].get('status', 'unknown')}")
    
    # Get complete network view
    print("\n🌐 CONNECTED KNOWLEDGE NETWORK:")
    network = agent_zero_graph.get_connected_knowledge_network()
    print(f"   Root graph: {network['root_graph']['graph_id']} ({network['root_graph']['node_count']} nodes)")
    print(f"   Connected graphs: {len(network['connected_graphs'])}")
    for graph_id, info in network['connected_graphs'].items():
        print(f"     - {graph_id}: {info['graph_type']} ({info['node_count']} nodes)")
    print(f"   Cross-graph connections: {len(network['cross_graph_connections'])}")
    
    # Demo real-time status updates (no message passing needed!)
    print("\n⚡ REAL-TIME STATUS UPDATE DEMO:")
    print("   DataAnalyst completes task...")
    data_analyst_graph.set_node_attribute("current_task", "status", "completed")
    data_analyst_graph.set_node_attribute("current_task", "progress", 1.0)
    data_analyst_graph.set_node_attribute("agent_status", "load", 0.1)
    
    # AgentZero can immediately see the update without message passing
    resolved_data = agent_zero_graph.resolve_external_reference(
        "mission_objective", 
        "data_analyst_sarah", 
        "current_task"
    )
    
    if resolved_data:
        print(f"   ✅ AgentZero sees update: Task {resolved_data['attributes']['task']} is now {resolved_data['attributes']['status']}")
        print(f"   📊 Progress: {resolved_data['attributes']['progress']*100}%")
    
    # Demo JSON serialization with cross-graph references
    print(f"\n📄 JSON serialization: {len(agent_zero_graph.to_json())} characters")
    
    return agent_zero_graph, data_analyst_graph, security_expert_graph

def demo_visualizations():
    """Demo all visualization capabilities"""
    
    print("\n📊 VISUALIZATION DEMO")
    print("=" * 40)
    
    # Get the graphs from the main demo
    agent_zero_graph, data_analyst_graph, security_expert_graph = demo_knowledge_graph()
    
    # Demo 3D visualizations
    print("\n🎲 3D VISUALIZATIONS:")
    print("   Generating 3D and isometric plots...")
    
    print("   4. AgentZero Mission Graph (3D - Layered Strategy)")
    matplotlib_viz_3d = MatplotlibKnowledgeGraphVisualizer(agent_zero_graph, StylePresets.professional())
    matplotlib_viz_3d.render_3d_graph(figsize=(12, 9), show_labels=True, 
                                      show_attributes=True, positioning_strategy="layered")
    
    print("   5. DataAnalyst Tactical Graph (3D - Layered Strategy)")
    matplotlib_viz_3d2 = MatplotlibKnowledgeGraphVisualizer(data_analyst_graph, StylePresets.dark_mode())
    matplotlib_viz_3d2.render_3d_graph(figsize=(10, 8), show_labels=True, 
                                       show_attributes=True, positioning_strategy="layered")
    
    print("   6. SecurityExpert Tactical Graph (3D - Layered Strategy)")
    matplotlib_viz_3d3 = MatplotlibKnowledgeGraphVisualizer(security_expert_graph, StylePresets.high_contrast())
    matplotlib_viz_3d3.render_3d_graph(figsize=(10, 8), show_labels=True, 
                                       show_attributes=True, positioning_strategy="layered")
    
    print("   7. BRAIN KNOWLEDGE NETWORK (Managed Z-Layers + Physics)")
    # Create brain network manager
    brain_network = BrainKnowledgeGraph("agent_zero_brain", z_separation=2.0)
    
    # Add all graphs to the brain network
    brain_network.add_graph(agent_zero_graph)  # Will be at Z=10.0
    brain_network.add_graph(data_analyst_graph)  # Will be at Z=6.0
    brain_network.add_graph(security_expert_graph)  # Will be at Z=4.0
    
    # Apply physics to all graphs
    brain_network.apply_physics_to_all(iterations=50)
    
    # Visualize the managed brain network
    brain_viz = BrainKnowledgeGraphVisualizer(brain_network, StylePresets.professional())
    brain_viz.render_unified_3d_network(figsize=(20, 16))
    
    print("\n🎯 BENEFITS:")
    print("   ✅ No ZMQ message passing needed for status updates")
    print("   ✅ Real-time visibility into all sub-agent states")
    print("   ✅ Unified queries across distributed knowledge")
    print("   ✅ Automatic hive-mind coordination")
    print("   ✅ Logical separation with physical connectivity")
    print("   ✅ Beautiful matplotlib visualizations for debugging and analysis")
    print("   ✅ 3D and isometric visualizations for spatial understanding")

def demo_3d_positioning():
    """Demo different 3D positioning strategies"""
    
    print("\n🎲 3D POSITIONING STRATEGIES DEMO")
    print("=" * 50)
    
    # Create a test graph
    test_graph = KnowledgeGraph(GraphType.MISSION, "positioning_test")
    
    # Add nodes of different types
    test_graph.add_node("mission_1", type="mission", priority="high")
    test_graph.add_node("coord_1", type="coordination", role="hub")
    test_graph.add_node("task_1", type="task", status="active")
    test_graph.add_node("task_2", type="task", status="pending")
    test_graph.add_node("status_1", type="status", health="good")
    test_graph.add_node("output_1", type="output", ready=True)
    
    # Add some connections
    test_graph.add_edge("mission_1", "coord_1")
    test_graph.add_edge("coord_1", "task_1")
    test_graph.add_edge("coord_1", "task_2")
    test_graph.add_edge("task_1", "status_1")
    test_graph.add_edge("task_1", "output_1")
    
    strategies = ["layered", "hierarchical", "circular", "random"]
    
    for strategy in strategies:
        print(f"\n   Testing {strategy} positioning strategy:")
        test_graph.auto_assign_3d_positions(strategy)
        
        # Show positions
        for node_id in test_graph.nodes():
            pos = test_graph.get_node_3d_position(node_id)
            if pos:
                print(f"     {node_id}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        
        # Create visualization placeholder
        print(f"     Generating {strategy} visualization...")
        print(f"     (Isometric plot would be generated here with strategy: {strategy})")

def render_comprehensive_layout(brain_network, all_graphs):
    """Render comprehensive layout with subplots + composite view"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
    except ImportError:
        print("❌ matplotlib not available. Install with: pip install matplotlib")
        return
    
    # Get brain colors and network info
    brain_colors = brain_network.get_all_graph_colors()
    network_info = brain_network.get_unified_network_info()
    graph_z_indices = {info['graph_id']: info['z_index'] for info in network_info['graphs']}
    all_positions = brain_network.get_all_nodes_3d()
    
    # Create figure with custom layout using GridSpec (screen-friendly size)
    fig = plt.figure(figsize=(16, 10))  # More reasonable screen size
    import matplotlib.gridspec as gridspec
    
    # Create GridSpec: 4 rows, 4 columns with padding (left 2 cols for subplots, right 2 cols for composite)
    gs = gridspec.GridSpec(4, 4, figure=fig, 
                          hspace=0.4,    # Vertical spacing between subplots
                          wspace=0.3,    # Horizontal spacing between subplots
                          top=0.95,      # Top margin
                          bottom=0.05,   # Bottom margin
                          left=0.05,     # Left margin
                          right=0.95)    # Right margin
    
    # Left half: 4 high x 2 wide subplot grid for individual graphs
    subplot_specs = [
        gs[0, 0], gs[0, 1],  # Row 1: Left, Right
        gs[1, 0], gs[1, 1],  # Row 2: Left, Right  
        gs[2, 0], gs[2, 1],  # Row 3: Left, Right
        gs[3, 0], gs[3, 1]   # Row 4: Left, Right
    ]
    
    # Render individual graphs in subplots (2D plan view) - all 7 graphs
    for i, graph in enumerate(all_graphs):
        if i < len(subplot_specs):
            ax = fig.add_subplot(subplot_specs[i])  # 2D subplot
            render_individual_graph_2d(ax, graph, brain_colors, graph_z_indices, i+1)
    
    # Right half: Large composite view (spans full right side)
    ax_composite = fig.add_subplot(gs[:, 2:], projection='3d')
    render_composite_with_base_plane(ax_composite, brain_network, all_positions, brain_colors)
    
    plt.tight_layout()
    plt.show()

def render_individual_graph_2d(ax, graph, brain_colors, graph_z_indices, layer_num):
    """Render individual graph in 2D plan view subplot"""
    import networkx as nx
    import numpy as np
    
    # Get graph color and Z-index (no fallbacks - fail if missing)
    graph_color = brain_colors[graph.graph_id]
    z_index = graph_z_indices[graph.graph_id]
    
    # Create 2D layout
    pos_2d = nx.spring_layout(graph, k=2, iterations=50)
    
    # Extract node positions (2D only)
    node_positions = {}
    for node_id in graph.nodes():
        x, y = pos_2d[node_id]
        node_positions[node_id] = (x, y)  # Keep in -1 to 1 range for better layout
    
    # Plot edges first (so they appear behind nodes)
    for edge in graph.edges():
        source, target = edge
        x_vals = [node_positions[source][0], node_positions[target][0]]
        y_vals = [node_positions[source][1], node_positions[target][1]]
        ax.plot(x_vals, y_vals, color='gray', alpha=0.6, linewidth=1.5)
    
    # Plot nodes using standard style
    xs, ys = zip(*node_positions.values())
    style_params = StylePresets.professional().get_standard_node_style()
    ax.scatter(xs, ys, c=graph_color, **style_params)
    
    # Add labels
    for node_id, (x, y) in node_positions.items():
        # Truncate long node names for readability
        label = node_id if len(node_id) <= 12 else node_id[:10] + "..."
        ax.text(x, y, label, fontsize=8, ha='center', va='center', 
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    
    # Configure subplot
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    # Shorter title to fit better
    title = f"L{layer_num}: {graph.graph_id.replace('_', ' ').title()}\n(Z={z_index:.1f})"
    ax.set_title(title, fontsize=8, pad=5)
    
    # Remove axes for clean look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def render_composite_with_base_plane(ax, brain_network, all_positions, brain_colors):
    """Render composite view with semi-transparent base plane"""
    import numpy as np
    
    # Add semi-transparent base plane at Z=0
    xx, yy = np.meshgrid(np.linspace(0, 100, 10), np.linspace(0, 100, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='lightblue', linewidth=0)
    
    # Get network info for rendering
    network_info = brain_network.get_unified_network_info()
    
    # Plot all nodes with brain colors
    for i, graph_info in enumerate(network_info['graphs']):
        graph_id = graph_info['graph_id']
        graph = brain_network.get_graph(graph_id)
        base_color = brain_colors[graph_id]  # No fallback - fail if missing
        
        # Collect node positions for this graph
        graph_positions = []
        graph_labels = []
        
        for node_id in graph.nodes():
            unified_node_id = f"{graph_id}:{node_id}"
            if unified_node_id in all_positions:
                pos = all_positions[unified_node_id]
                graph_positions.append(pos)
                graph_labels.append(node_id)
        
        if graph_positions:
            xs, ys, zs = zip(*graph_positions)
            # Use standard style for consistency
            style_params = StylePresets.professional().get_standard_node_style()
            ax.scatter(xs, ys, zs, c=base_color, label=f"{graph_id}", **style_params)
            
            # Add node labels
            for (x, y, z), label in zip(graph_positions, graph_labels):
                ax.text(x, y, z, label, fontsize=7, ha='center', va='center')
    
    # Plot edges within graphs (render edges first so they appear behind nodes)
    print(f"🔍 Rendering edges for {len(network_info['graphs'])} graphs...")
    for graph_info in network_info['graphs']:
        graph_id = graph_info['graph_id']
        graph = brain_network.get_graph(graph_id)
        edge_count = 0
        
        for edge in graph.edges():
            source_unified = f"{graph_id}:{edge[0]}"
            target_unified = f"{graph_id}:{edge[1]}"
            
            if source_unified in all_positions and target_unified in all_positions:
                source_pos = all_positions[source_unified]
                target_pos = all_positions[target_unified]
                
                ax.plot([source_pos[0], target_pos[0]], 
                       [source_pos[1], target_pos[1]], 
                       [source_pos[2], target_pos[2]], 
                       color='gray', alpha=0.7, linewidth=1.5)
                edge_count += 1
        
        print(f"   {graph_id}: {edge_count} edges rendered")
    
    # Plot cross-graph edges (external references) in red
    print("🔗 Rendering cross-graph connections...")
    cross_edge_count = 0
    for graph_info in network_info['graphs']:
        graph_id = graph_info['graph_id']
        graph = brain_network.get_graph(graph_id)
        
        # Check for external references
        if hasattr(graph, 'external_references') and graph.external_references:
            for local_node, refs in graph.external_references.items():
                for ref_key, ref_data in refs.items():
                    source_unified = f"{graph_id}:{local_node}"
                    target_unified = f"{ref_data['target_graph']}:{ref_data['target_node']}"
                    
                    if source_unified in all_positions and target_unified in all_positions:
                        source_pos = all_positions[source_unified]
                        target_pos = all_positions[target_unified]
                        
                        ax.plot([source_pos[0], target_pos[0]], 
                               [source_pos[1], target_pos[1]], 
                               [source_pos[2], target_pos[2]], 
                               color='red', alpha=0.8, linewidth=2, linestyle='--')
                        cross_edge_count += 1
                        print(f"     {source_unified} -> {target_unified}")
    
    print(f"   Cross-graph connections: {cross_edge_count} edges rendered")
    
    # Configure composite view
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)
    ax.set_title("Unified Wedding Planning Brain\n(Spinnable 3D with Base Plane)", 
                fontsize=10, fontweight='bold', pad=10)
    
    # Add legend with better spacing and smaller markers
    legend = ax.legend(loc='upper left', fontsize=8, markerscale=0.6, 
                      framealpha=0.9, fancybox=True, shadow=True,
                      columnspacing=1.0, handletextpad=0.5)
    
    # Remove axes for clean look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

def demonstrate_brain_queries(brain_network):
    """Demonstrate brain's built-in query capabilities"""
    
    print("=" * 60)
    
    # Use brain's built-in query methods
    analysis = brain_network.run_comprehensive_analysis()
    
    # Query 1: Critical Path Analysis
    print("\n🎯 QUERY 1: Critical Path Analysis")
    critical_nodes = analysis['critical_path_nodes']
    print(f"   Found {len(critical_nodes)} critical path nodes:")
    for node in critical_nodes:
        print(f"   • {node}")
    
    # Query 2: Bottleneck Detection
    print("\n🔗 QUERY 2: Bottleneck Detection (High-Degree Nodes)")
    bottlenecks = analysis['bottlenecks']
    print(f"   Found {len(bottlenecks)} potential bottlenecks:")
    for node, connections in bottlenecks:
        print(f"   • {node} ({connections} connections)")
    
    # Query 3: Cross-Graph Dependencies
    print("\n🌐 QUERY 3: Cross-Graph Dependency Analysis")
    dependencies = analysis['cross_graph_dependencies']
    print("   Cross-graph dependencies:")
    for graph_id, deps in dependencies.items():
        print(f"   • {graph_id} depends on:")
        for dep in deps:
            print(f"     - {dep}")
    
    # Query 4: Resource Utilization
    print("\n📊 QUERY 4: Resource Utilization Analysis")
    resource_stats = analysis['resource_utilization']
    print("   Resource utilization by graph:")
    for graph_id, stats in resource_stats.items():
        print(f"   • {graph_id}:")
        print(f"     - Nodes: {stats['total_nodes']}, Edges: {stats['total_edges']}")
        print(f"     - External refs: {stats['external_refs']}")
        print(f"     - Tasks: {stats['active_tasks']} active, {stats['completed_tasks']} completed")
    
    # Query 5: Coordination Hubs
    print("\n🎛️ QUERY 5: Coordination Hub Analysis")
    coordination_hubs = analysis['coordination_hubs']
    print(f"   Found {len(coordination_hubs)} coordination hubs:")
    for hub in coordination_hubs:
        print(f"   • {hub['node']} ({hub['connections']} connections, {hub['status']})")
    
    # Query 6: Z-Layer Distribution
    print("\n📏 QUERY 6: Z-Layer Distribution Analysis")
    z_distribution = analysis['z_layer_distribution']
    print("   Node distribution by Z-level:")
    for z_level in sorted(z_distribution.keys()):
        nodes = z_distribution[z_level]
        graph_types = set(node.split(':')[0] for node in nodes)
        print(f"   • Z={z_level}: {len(nodes)} nodes from {len(graph_types)} graphs")
        print(f"     - Graphs: {', '.join(sorted(graph_types))}")
    
    print("=" * 60)

def demo_wedding_visualizations():
    """Demo wedding planning with unified brain network and consistent colors"""
    
    print("\n💒 WEDDING PLANNING VISUALIZATION DEMO")
    print("=" * 50)
    
    # Get all the wedding planning graphs
    graphs = demo_wedding_planning_agents()
    mission_graph, venue_agent_graph, catering_agent_graph, photo_agent_graph, local_rag_graph, budget_rag_graph, cloud_rag_graph = graphs
    
    # Show ASCII visualizations for key graphs
    print("\n📊 KEY GRAPH VISUALIZATIONS:")
    print("\n1. Wedding Mission Graph:")
    ascii_viz1 = ASCIIKnowledgeGraphVisualizer(mission_graph, StylePresets.professional())
    ascii_viz1.print_graph(show_attributes=True)
    
    print("\n2. Venue Specialist Emma:")
    ascii_viz2 = ASCIIKnowledgeGraphVisualizer(venue_agent_graph, StylePresets.minimal())
    ascii_viz2.print_graph(show_attributes=True, show_external_refs=False)
    
    # Create brain network manager with all graphs
    print("\n🧠 CREATING UNIFIED WEDDING PLANNING BRAIN...")
    brain_network = BrainKnowledgeGraph("wedding_planning_brain", z_separation=1.5)
    
    # Calculate dynamic Z-indices within 0-100 bounds
    all_graphs = [mission_graph, venue_agent_graph, catering_agent_graph, 
                  photo_agent_graph, local_rag_graph, budget_rag_graph, cloud_rag_graph]
    total_graphs = len(all_graphs)
    z_spacing = 100.0 / (total_graphs - 1) if total_graphs > 1 else 0.0
    
    # Add all graphs with dynamic Z-indices (Mission at Z=0, others distributed to Z=100)
    for i, graph in enumerate(all_graphs):
        z_index = i * z_spacing  # Mission=0, others evenly spaced to 100
        brain_network.add_graph(graph, z_index=z_index)
        print(f"   Added {graph.graph_id} at Z={z_index:.1f}")
    
    # Apply coordinated physics to all graphs with much more aggressive separation
    brain_network.set_physics_params(
        k_spring=0.3,      # Very weak attraction for maximum spread
        k_repulsion=8.0,   # Very strong repulsion to push nodes apart
        damping=0.95,      # Higher stability
        ideal_length=20.0  # Much larger ideal distance for full 100x100 utilization
    )
    brain_network.apply_physics_to_all(iterations=120)
    
    # Debug: Print all node coordinates
    print("\n🔍 NODE COORDINATES DEBUG:")
    all_positions = brain_network.get_all_nodes_3d()
    for unified_node_id, pos in sorted(all_positions.items()):
        graph_id, node_id = unified_node_id.split(':', 1)
        print(f"   {graph_id}:{node_id} → ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    print(f"\n📊 Z-Level Summary:")
    z_levels = {}
    for unified_node_id, pos in all_positions.items():
        graph_id = unified_node_id.split(':', 1)[0]
        z_val = pos[2]
        if z_val not in z_levels:
            z_levels[z_val] = []
        z_levels[z_val].append(graph_id)
    
    for z_val in sorted(z_levels.keys()):
        graphs_at_z = list(set(z_levels[z_val]))  # Remove duplicates
        print(f"   Z={z_val:.1f}: {', '.join(graphs_at_z)}")
    
    # Show network info
    network_info = brain_network.get_unified_network_info()
    print(f"\n📊 Wedding Planning Network Summary:")
    print(f"   Total Graphs: {network_info['total_graphs']}")
    print(f"   Total Nodes: {network_info['total_nodes']}")
    print(f"   Total Edges: {network_info['total_edges']}")
    print(f"   Cross-connections: {network_info['cross_connections']}")
    print(f"   Z-separation: {network_info['z_separation']}")
    
    # Create comprehensive matplotlib layout with subplots + composite
    print("\n🎲 COMPREHENSIVE 3D VISUALIZATION LAYOUT:")
    print("   Left half: Individual graph subplots (2x4 grid)")
    print("   Right half: Full spinnable composite with base plane")
    
    render_comprehensive_layout(brain_network, all_graphs)
    
    # Demonstrate querying capabilities using brain's built-in methods
    print("\n🔍 QUERYING THE COMPOSITE KNOWLEDGE GRAPH:")
    demonstrate_brain_queries(brain_network)
    
    print("\n🎯 WEDDING PLANNING BENEFITS:")
    print("   ✅ Mission-driven coordination with clear objectives")
    print("   ✅ Direct mission-to-specialist coordination (no middleware)")
    print("   ✅ Specialist agents handle domain expertise (venue, catering, photography)")
    print("   ✅ Local RAG manages personal data (guest list, preferences)")
    print("   ✅ Cloud RAG provides market intelligence (venues, vendors)")
    print("   ✅ Real-time coordination without message passing overhead")
    print("   ✅ Physics-based positioning prevents node overlap")
    print("   ✅ Clear Z-layer separation shows agent hierarchy")
    print("   ✅ Mission at Z=0 foundation with specialists above")

if __name__ == "__main__":
    # Run the wedding planning demo
    demo_wedding_visualizations()
    
    # Uncomment to run original demo
    # demo_knowledge_graph()
    # demo_visualizations()
    # demo_3d_positioning()
