#!/usr/bin/env python3
"""
Debug mission graph step by step
"""

from mission_knowledge_graph import MissionKnowledgeGraph

def test_step_by_step():
    print("Step 1: Create mission graph...")
    mission = MissionKnowledgeGraph("debug_mission")
    print("OK Mission created")
    
    print("\nStep 2: Add first task...")
    mission.add_task("task1", name="Task 1", status="completed")
    print("OK First task added")
    
    print("\nStep 3: Add second task...")
    mission.add_task("task2", name="Task 2", status="pending")
    print("OK Second task added")
    
    print("\nStep 4: Add dependency...")
    mission.add_dependency("task1", "task2")
    print("OK Dependency added")
    
    print("\nStep 5: Test basic queries...")
    print(f"   Nodes: {mission.nodes()}")
    print(f"   Edges: {mission.edges()}")
    
    print("\nStep 6: Test get_node_attributes...")
    attrs1 = mission.get_node_attributes("task1")
    print(f"   Task1 attrs: {attrs1}")
    
    print("\nStep 7: Test critical path (this might cause recursion)...")
    try:
        critical_path = mission.find_critical_path()
        print(f"   Critical path: {critical_path}")
    except RecursionError as e:
        print(f"   ERROR RECURSION in find_critical_path: {e}")
        return
    except Exception as e:
        print(f"   ERROR OTHER in find_critical_path: {e}")
        return
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_step_by_step()
