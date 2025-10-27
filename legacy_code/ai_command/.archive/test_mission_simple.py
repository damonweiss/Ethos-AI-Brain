#!/usr/bin/env python3
"""
Simple test to isolate the recursion issue
"""

from mission_knowledge_graph import MissionKnowledgeGraph

def test_simple_creation():
    print("Creating mission graph...")
    mission = MissionKnowledgeGraph("test_mission")
    print("Mission graph created successfully!")
    
    print("Adding a simple task...")
    mission.add_task("test_task", name="Test Task", status="pending")
    print("Task added successfully!")
    
    print("Getting task attributes...")
    attrs = mission.get_node_attributes("test_task")
    print(f"Task attributes: {attrs}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_simple_creation()
