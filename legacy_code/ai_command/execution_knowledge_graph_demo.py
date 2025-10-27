#!/usr/bin/env python3
"""
Clean Execution Knowledge Graph Demo (no Unicode issues)
Demonstrates execution-specific query capabilities with a wedding planning scenario
"""

from datetime import datetime
from execution_knowledge_graph import ExecutionKnowledgeGraph

def create_wedding_execution():
    """Create a wedding planning execution graph"""
    
    print("CREATING WEDDING EXECUTION GRAPH")
    print("=" * 50)
    
    # Create execution graph
    execution = ExecutionKnowledgeGraph("wedding_execution_2024", "Sarah & Mike's Wedding")
    
    # Add mission tasks
    tasks = [
        ("budget_planning", {
            "name": "Create Wedding Budget",
            "priority": "critical",
            "duration": 3.0,
            "status": "completed",
            "type": "planning"
        }),
        ("guest_list", {
            "name": "Finalize Guest List", 
            "priority": "critical",
            "duration": 5.0,
            "status": "completed",
            "type": "planning"
        }),
        ("venue_booking", {
            "name": "Book Wedding Venue",
            "priority": "critical", 
            "duration": 7.0,
            "status": "completed",
            "type": "booking"
        }),
        ("catering_contract", {
            "name": "Sign Catering Contract",
            "priority": "critical",
            "duration": 4.0,
            "status": "in_progress",
            "type": "booking"
        }),
        ("photographer_booking", {
            "name": "Book Wedding Photographer",
            "priority": "high",
            "duration": 3.0,
            "status": "completed",
            "type": "booking"
        }),
        ("dress_shopping", {
            "name": "Wedding Dress Shopping",
            "priority": "high",
            "duration": 8.0,
            "status": "in_progress",
            "type": "attire"
        }),
        ("invitations_send", {
            "name": "Send Wedding Invitations",
            "priority": "critical",
            "duration": 2.0,
            "status": "pending",
            "type": "communication"
        }),
        ("wedding_day", {
            "name": "Wedding Day Coordination",
            "priority": "critical",
            "duration": 12.0,
            "status": "pending",
            "type": "event",
            "is_milestone": True
        })
    ]
    
    # Add all tasks
    for task_id, attributes in tasks:
        execution.add_task(task_id, **attributes)
    
    print(f"Added {len(tasks)} execution tasks")
    
    # Add dependencies
    dependencies = [
        ("budget_planning", "venue_booking", "enables"),
        ("guest_list", "venue_booking", "informs"),
        ("venue_booking", "catering_contract", "enables"),
        ("guest_list", "invitations_send", "enables"),
        ("catering_contract", "wedding_day", "required_for"),
        ("photographer_booking", "wedding_day", "required_for"),
        ("invitations_send", "wedding_day", "required_for")
    ]
    
    for from_task, to_task, dep_type in dependencies:
        execution.add_dependency(from_task, to_task, dep_type)
    
    print(f"Added {len(dependencies)} dependencies")
    print(f"Execution Graph: {len(execution.nodes())} nodes, {len(execution.edges())} edges")
    
    return execution

def demo_execution_queries(execution):
    """Demonstrate execution query capabilities"""
    
    print("\nEXECUTION QUERY DEMONSTRATIONS")
    print("=" * 50)
    
    # Critical Path Analysis
    print("\n1. CRITICAL PATH ANALYSIS:")
    critical_path = execution.find_critical_path()
    print(f"   Critical path has {len(critical_path)} tasks:")
    for i, task in enumerate(critical_path, 1):
        attrs = execution.get_node_attributes(task)
        status = attrs.get('status', 'unknown')
        priority = attrs.get('priority', 'medium')
        print(f"   {i}. {task} ({status}, {priority} priority)")
    
    # Workflow Blockers
    print("\n2. WORKFLOW BLOCKER ANALYSIS:")
    blockers = execution.detect_workflow_blockers()
    print(f"   Found {len(blockers)} workflow blockers:")
    for blocker in blockers[:3]:  # Show top 3 blockers
        print(f"   - {blocker['node_id']} (severity: {blocker['blocker_severity']:.1f})")
        print(f"     Status: {blocker['status']}, Blocks {blocker['blocking_count']} tasks")
    
    # Execution Progress
    print("\n3. EXECUTION PROGRESS ANALYSIS:")
    progress = execution.calculate_execution_progress()
    print(f"   Overall completion: {progress['completion_percentage']:.1f}%")
    print(f"   Completed tasks: {progress['completed_tasks']}/{progress['total_tasks']}")
    print(f"   In progress: {progress['in_progress_tasks']}")
    print(f"   Pending: {progress['pending_tasks']}")
    
    # Timeline Analysis
    print("\n4. TIMELINE DEPENDENCY ANALYSIS:")
    timeline = execution.analyze_timeline_dependencies()
    print(f"   Total tasks: {timeline['total_tasks']}")
    print(f"   Total dependencies: {timeline['total_dependencies']}")
    print(f"   Dependency depth: {timeline['dependency_depth']}")
    print(f"   Estimated duration: {timeline['estimated_duration']:.1f} days")
    
    # Risk Factors
    print("\n5. RISK FACTOR ANALYSIS:")
    risks = execution.identify_risk_factors()
    print(f"   Identified {len(risks)} risk factors:")
    for risk in risks[:3]:  # Show top 3 risks
        print(f"   - {risk['type']} ({risk['severity']} severity)")
        print(f"     {risk['description']}")

def main():
    """Main demo function"""
    
    print("EXECUTION KNOWLEDGE GRAPH DEMONSTRATION")
    print("=" * 60)
    print("Demonstrating execution-specific query capabilities")
    print("with a wedding planning scenario")
    print("=" * 60)
    
    # Create the wedding execution graph
    execution = create_wedding_execution()
    
    # Demonstrate query capabilities
    demo_execution_queries(execution)
    
    print("\nEXECUTION DEMO COMPLETE!")
    print("=" * 60)
    print("Key Takeaways:")
    print("- Execution graphs provide workflow-specific analysis")
    print("- Critical path analysis identifies bottlenecks")
    print("- Risk assessment helps prevent execution failure")
    print("- Progress tracking shows real-time completion status")
    print("- Timeline analysis optimizes task scheduling")
    print("=" * 60)

if __name__ == "__main__":
    main()
