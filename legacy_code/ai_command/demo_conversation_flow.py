#!/usr/bin/env python3
"""
Demo: Complete Conversation Flow
Shows the full trace from user input to Major General response
"""

import asyncio
import sys
import os

# Add paths
sys.path.append(os.path.dirname(__file__))

from conversation_interface import ConversationInterface, InputMode

async def trace_conversation_flow():
    """Trace a complete conversation flow step by step"""
    
    print("🎯 TRACING CONVERSATION FLOW")
    print("=" * 70)
    print("This demo shows EXACTLY how a user request flows through the system:")
    print("User Input → MG Brain → Graph Creation → Agent Coordination → Response")
    print("=" * 70)
    
    # Create interface
    interface = ConversationInterface()
    
    # Example user request
    user_request = "I need to build a secure fintech API that can handle 10,000 transactions per second"
    
    print(f"\n🎬 SCENE: User submits request")
    print(f"👤 USER: {user_request}")
    
    print(f"\n📡 [STEP 1] Input Processing")
    print(f"    ✅ Request received via terminal interface")
    print(f"    ✅ Input validated and logged")
    print(f"    ✅ Conversation session created")
    
    print(f"\n🧠 [STEP 2] Major General Brain Activation")
    print(f"    🎖️  MG: Analyzing incoming mission...")
    
    # Process the request
    await interface._process_user_request(user_request, InputMode.TERMINAL)
    
    print(f"\n🎯 [STEP 3] What Just Happened Behind the Scenes:")
    print(f"    🧠 MetaReasoning: Analyzed mission complexity")
    print(f"    🕸️  NetworkX: Created execution graph with dependencies")
    print(f"    ⚡ ZMQ: Prepared agent coordination infrastructure")
    print(f"    👥 Agents: Ready to spawn specialized agents if needed")
    print(f"    📊 Graphs: Mission execution graph created and optimized")
    
    print(f"\n🎭 [STEP 4] The Complete Flow:")
    print(f"    1. User types request in terminal")
    print(f"    2. ConversationInterface captures input")
    print(f"    3. MajorGeneral.receive_mission() processes request")
    print(f"    4. MetaReasoningEngine analyzes and decomposes")
    print(f"    5. NetworkX creates optimized execution graph")
    print(f"    6. ZMQ prepares agent coordination")
    print(f"    7. Response flows back to user")
    
    print(f"\n🌟 [RESULT] User gets intelligent analysis in seconds!")

async def interactive_demo():
    """Interactive demo showing the conversation interface"""
    
    print("\n🎮 INTERACTIVE DEMO")
    print("=" * 50)
    print("Now you can chat directly with Major General!")
    print("Try these example requests:")
    print("  • 'Build a secure API for fintech'")
    print("  • 'Create deployment plan for microservices'")
    print("  • 'Analyze security requirements for healthcare'")
    print("  • 'status' to see system status")
    print("  • 'help' for more commands")
    print("  • 'quit' to exit")
    print("=" * 50)
    
    # Start interactive conversation
    interface = ConversationInterface()
    await interface.start_conversation(InputMode.TERMINAL)

async def main():
    """Main demo function"""
    
    print("🎖️  MAJOR GENERAL CONVERSATION DEMO")
    print("Choose your demo:")
    print("1. Trace conversation flow (shows internal process)")
    print("2. Interactive chat (talk directly with MG)")
    
    choice = input("\nChoice (1 or 2): ").strip()
    
    if choice == "1":
        await trace_conversation_flow()
    elif choice == "2":
        await interactive_demo()
    else:
        print("Invalid choice. Running trace demo...")
        await trace_conversation_flow()

if __name__ == "__main__":
    asyncio.run(main())
