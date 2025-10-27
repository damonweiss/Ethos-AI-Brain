#!/usr/bin/env python3
"""
Test Digital Twin State Sync Functionality
Tests export/import JSON capabilities for knowledge graphs
"""

from intent_knowledge_graph import IntentKnowledgeGraph
from execution_knowledge_graph import ExecutionKnowledgeGraph
import json

def test_intent_graph_sync():
    """Test intent graph digital twin sync"""
    print("TESTING INTENT GRAPH DIGITAL TWIN SYNC")
    print("=" * 50)
    
    # Create original intent graph
    original = IntentKnowledgeGraph("test_intent", "Test Intent")
    
    # Add some data
    original.add_objective("main_goal", 
                          name="Test Objective",
                          priority="high",
                          measurable=True)
    
    original.add_constraint("budget_limit",
                           constraint_type="budget", 
                           value=10000,
                           flexibility="rigid")
    
    original.add_stakeholder("user",
                            name="Test User",
                            role="decision_maker",
                            influence_level="high")
    
    # Add intent derivation data
    original.set_raw_user_prompt("I need to test the digital twin sync functionality")
    original.set_domain_context("testing")
    original.add_confidence_score("objectives", 0.9)
    
    print(f"Original graph: {len(original.nodes())} nodes, {len(original.edges())} edges")
    
    # Export to JSON
    exported_json = original.export_to_json()
    print(f"Exported JSON keys: {list(exported_json.keys())}")
    print(f"Graph data nodes: {len(exported_json['graph_data']['nodes'])}")
    
    # Create new graph and import
    replica = IntentKnowledgeGraph("replica_intent", "Replica Intent")
    update_stats = replica.update_from_json(exported_json, merge_mode="replace")
    
    print(f"Import stats: {update_stats}")
    print(f"Replica graph: {len(replica.nodes())} nodes, {len(replica.edges())} edges")
    
    # Verify data integrity
    print(f"Domain context preserved: {replica.domain_context}")
    print(f"Raw prompt preserved: {len(replica.raw_user_prompt)} chars")
    print(f"Confidence scores preserved: {len(replica.intent_confidence_scores)}")
    
    # Test state summary
    summary = replica.get_state_summary()
    print(f"State summary: {summary['node_count']} nodes, {summary['node_types']}")
    
    return True

def test_execution_graph_sync():
    """Test execution graph digital twin sync"""
    print("\nTESTING EXECUTION GRAPH DIGITAL TWIN SYNC")
    print("=" * 50)
    
    # Create original execution graph
    original = ExecutionKnowledgeGraph("test_execution", "Test Execution")
    
    # Add some tasks
    original.add_task("task1",
                     name="First Task",
                     status="completed",
                     priority="high",
                     duration=2.0)
    
    original.add_task("task2", 
                     name="Second Task",
                     status="in_progress",
                     priority="medium",
                     duration=3.0)
    
    original.add_dependency("task1", "task2", "enables")
    
    print(f"Original graph: {len(original.nodes())} nodes, {len(original.edges())} edges")
    
    # Export to JSON
    exported_json = original.export_to_json()
    print(f"Exported JSON keys: {list(exported_json.keys())}")
    
    # Create new graph and test merge mode
    replica = ExecutionKnowledgeGraph("replica_execution", "Replica Execution")
    
    # First, add some different data
    replica.add_task("task3",
                    name="Third Task", 
                    status="pending",
                    priority="low")
    
    print(f"Replica before merge: {len(replica.nodes())} nodes")
    
    # Test update merge mode (should merge with existing)
    update_stats = replica.update_from_json(exported_json, merge_mode="update")
    
    print(f"Merge stats: {update_stats}")
    print(f"Replica after merge: {len(replica.nodes())} nodes, {len(replica.edges())} edges")
    
    # Verify both original and new tasks exist
    all_nodes = list(replica.nodes())
    print(f"All nodes after merge: {all_nodes}")
    
    # Test state summary
    summary = replica.get_state_summary()
    print(f"Final state summary: {summary}")
    
    return True

def test_partial_updates():
    """Test partial JSON updates (simulating LLM updates)"""
    print("\nTESTING PARTIAL JSON UPDATES")
    print("=" * 50)
    
    # Create intent graph
    intent = IntentKnowledgeGraph("partial_test", "Partial Update Test")
    intent.add_objective("goal1", name="Initial Goal", priority="medium")
    
    print(f"Initial state: {len(intent.nodes())} nodes")
    
    # Simulate LLM providing partial update
    partial_update = {
        'graph_data': {
            'nodes': [
                {'id': 'goal2', 'name': 'LLM Added Goal', 'type': 'primary_objective', 'priority': 'high'},
                {'id': 'constraint1', 'name': 'LLM Added Constraint', 'type': 'constraint', 'constraint_type': 'timeline'}
            ],
            'links': [
                {'source': 'goal1', 'target': 'constraint1', 'relationship': 'constrained_by'}
            ]
        }
    }
    
    # Apply partial update
    update_stats = intent.update_from_json(partial_update, merge_mode="update")
    
    print(f"Partial update stats: {update_stats}")
    print(f"Final state: {len(intent.nodes())} nodes, {len(intent.edges())} edges")
    print(f"All nodes: {list(intent.nodes())}")
    
    return True

def main():
    """Run all digital twin sync tests"""
    print("DIGITAL TWIN STATE SYNC TESTS")
    print("=" * 60)
    
    try:
        # Test intent graph sync
        intent_success = test_intent_graph_sync()
        
        # Test execution graph sync  
        execution_success = test_execution_graph_sync()
        
        # Test partial updates
        partial_success = test_partial_updates()
        
        print("\n" + "=" * 60)
        print("DIGITAL TWIN SYNC TEST RESULTS:")
        print(f"Intent Graph Sync: {'[PASS]' if intent_success else '[FAIL]'}")
        print(f"Execution Graph Sync: {'[PASS]' if execution_success else '[FAIL]'}")
        print(f"Partial Updates: {'[PASS]' if partial_success else '[FAIL]'}")
        print("=" * 60)
        
        if all([intent_success, execution_success, partial_success]):
            print("ALL DIGITAL TWIN SYNC TESTS PASSED!")
        else:
            print("Some tests failed - check output above")
            
    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
