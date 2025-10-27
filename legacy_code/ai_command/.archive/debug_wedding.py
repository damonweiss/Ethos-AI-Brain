#!/usr/bin/env python3
"""
Debug wedding graph step by step to find recursion issue
"""

from mission_knowledge_graph import MissionKnowledgeGraph

def test_wedding_tasks():
    print("Creating wedding mission graph...")
    mission = MissionKnowledgeGraph("wedding_mission_2024", "Sarah & Mike's Wedding")
    
    # Add just a few tasks first
    print("Adding budget planning task...")
    mission.add_task("budget_planning", 
                    name="Create Wedding Budget",
                    priority="critical",
                    duration=3.0,
                    status="completed",
                    type="planning",
                    estimated_cost=2000,
                    description="Establish overall wedding budget and allocations")
    
    print("Adding guest list task...")
    mission.add_task("guest_list", 
                    name="Finalize Guest List", 
                    priority="critical",
                    duration=5.0,
                    status="completed",
                    type="planning",
                    guest_count=150,
                    description="Create and finalize complete guest list")
    
    print("Adding venue booking task...")
    mission.add_task("venue_booking", 
                    name="Book Wedding Venue",
                    priority="critical", 
                    duration=7.0,
                    status="completed",
                    type="booking",
                    estimated_cost=8000,
                    deadline="2024-07-15",
                    description="Research, visit, and book wedding venue")
    
    print("Adding dependencies...")
    mission.add_dependency("budget_planning", "venue_booking", "enables")
    mission.add_dependency("guest_list", "venue_booking", "informs")
    
    print("Testing critical path with 3 tasks...")
    try:
        critical_path = mission.find_critical_path()
        print(f"Critical path: {critical_path}")
    except RecursionError as e:
        print(f"RECURSION ERROR with 3 tasks: {str(e)[:200]}...")
        return
    except Exception as e:
        print(f"OTHER ERROR with 3 tasks: {str(e)[:200]}...")
        return
    
    # Add more tasks to see when it breaks
    print("\nAdding more tasks...")
    
    tasks_to_add = [
        ("catering_contract", {
            "name": "Sign Catering Contract",
            "priority": "critical",
            "duration": 4.0,
            "status": "in_progress",
            "type": "booking", 
            "estimated_cost": 12000,
            "deadline": "2024-08-01",
            "description": "Finalize catering menu and sign contract"
        }),
        ("photographer_booking", {
            "name": "Book Wedding Photographer",
            "priority": "high",
            "duration": 3.0,
            "status": "completed",
            "type": "booking",
            "estimated_cost": 3500,
            "description": "Research and book wedding photographer"
        })
    ]
    
    for task_id, attrs in tasks_to_add:
        print(f"Adding {task_id}...")
        mission.add_task(task_id, **attrs)
    
    print("Adding more dependencies...")
    mission.add_dependency("venue_booking", "catering_contract", "enables")
    mission.add_dependency("budget_planning", "catering_contract", "constrains")
    
    print("Testing critical path with 5 tasks...")
    try:
        critical_path = mission.find_critical_path()
        print(f"Critical path: {critical_path}")
    except RecursionError as e:
        print(f"RECURSION ERROR with 5 tasks: {str(e)[:200]}...")
        return
    except Exception as e:
        print(f"OTHER ERROR with 5 tasks: {str(e)[:200]}...")
        return
    
    print("All tests passed!")

if __name__ == "__main__":
    test_wedding_tasks()
