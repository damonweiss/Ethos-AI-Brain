#!/usr/bin/env python3
"""
Simple Mission Knowledge Graph Demo
Demonstrates mission-specific query capabilities without Unicode issues
"""

from datetime import datetime
from mission_knowledge_graph import MissionKnowledgeGraph

def create_simple_mission():
    """Create a simple mission graph for testing"""
    
    print("CREATING SIMPLE MISSION GRAPH")
    print("=" * 40)
    
    # Create mission graph
    mission = MissionKnowledgeGraph("test_mission", "Simple Test Mission")
    
    # Add some tasks
    mission.add_task("task_a", name="Task A", priority="critical", duration=2.0, status="completed")
    mission.add_task("task_b", name="Task B", priority="high", duration=3.0, status="in_progress")
    mission.add_task("task_c", name="Task C", priority="medium", duration=1.0, status="pending")
    mission.add_task("task_d", name="Task D", priority="low", duration=4.0, status="pending")
    
    # Add dependencies
    mission.add_dependency("task_a", "task_b", "enables")
    mission.add_dependency("task_b", "task_c", "enables")
    mission.add_dependency("task_a", "task_d", "enables")
    
    print(f"Created mission with {len(mission.nodes())} tasks and {len(mission.edges())} dependencies")
    
    return mission

def demo_queries(mission):
    """Demonstrate mission queries"""
    
    print("\nMISSION QUERY DEMONSTRATIONS")
    print("=" * 40)
    
    # Critical Path
    print("\n1. CRITICAL PATH ANALYSIS:")
    critical_path = mission.find_critical_path()
    print(f"   Critical path: {' -> '.join(critical_path)}")
    
    # Progress
    print("\n2. MISSION PROGRESS:")
    progress = mission.calculate_mission_progress()
    print(f"   Completion: {progress['completion_percentage']:.1f}%")
    print(f"   Completed: {progress['completed_tasks']}/{progress['total_tasks']}")
    print(f"   In Progress: {progress['in_progress_tasks']}")
    print(f"   Pending: {progress['pending_tasks']}")
    
    # Blockers
    print("\n3. WORKFLOW BLOCKERS:")
    blockers = mission.detect_workflow_blockers()
    print(f"   Found {len(blockers)} blockers:")
    for blocker in blockers:
        print(f"   - {blocker['node_id']} (severity: {blocker['blocker_severity']:.1f})")
        print(f"     Blocks: {', '.join(blocker['blocked_tasks'])}")
    
    # Timeline Analysis
    print("\n4. TIMELINE ANALYSIS:")
    timeline = mission.analyze_timeline_dependencies()
    print(f"   Total tasks: {timeline['total_tasks']}")
    print(f"   Dependencies: {timeline['total_dependencies']}")
    print(f"   Dependency depth: {timeline['dependency_depth']}")
    print(f"   Estimated duration: {timeline['estimated_duration']:.1f}")
    
    # Risk Factors
    print("\n5. RISK FACTORS:")
    risks = mission.identify_risk_factors()
    print(f"   Identified {len(risks)} risks:")
    for risk in risks:
        print(f"   - {risk['type']} ({risk['severity']})")
        print(f"     {risk['description']}")

def main():
    """Main demo function"""
    
    print("MISSION KNOWLEDGE GRAPH DEMO")
    print("=" * 50)
    
    # Create mission
    mission = create_simple_mission()
    
    # Demo queries
    demo_queries(mission)
    
    print("\nDEMO COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()
