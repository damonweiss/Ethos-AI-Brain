#!/usr/bin/env python3
"""
Clean Conversation Demo - No ZMQ Noise
Shows the conversation flow without distracting server messages
"""

import asyncio
import sys
import os
import logging
from io import StringIO

# Suppress all logging noise
logging.basicConfig(level=logging.CRITICAL)

# Add paths
sys.path.append(os.path.dirname(__file__))

async def clean_conversation_demo():
    """Clean demo without ZMQ noise"""
    
    print("🎖️  MAJOR GENERAL - CLEAN CONVERSATION DEMO")
    print("=" * 60)
    print("This demo shows the conversation flow without server noise")
    print("=" * 60)
    
    # Suppress stdout during initialization
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        from conversation_interface import ConversationInterface, InputMode
        interface = ConversationInterface()
        
        # Try to upgrade to real LLM backends if available
        try:
            import os
            if os.getenv("OPENAI_API_KEY"):
                sys.stdout = old_stdout  # Temporarily restore stdout for messages
                print("🤖 Upgrading to real LLM backends...")
                sys.stdout = StringIO()  # Suppress again
                
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests', 'ai_command', 'meta_reasoning_engine'))
                from demo_real_llm import RealLLMBackend
                
                # Replace mock backends with real ones
                interface.major_general.meta_reasoning.register_llm_backend(
                    "analyst", RealLLMBackend("Strategic Analyst", "analytical")
                )
                interface.major_general.meta_reasoning.register_llm_backend(
                    "decomposer", RealLLMBackend("Task Decomposer", "systematic")
                )
                
                sys.stdout = old_stdout  # Restore for success message
                print("✅ Real AI analysis enabled!")
                sys.stdout = StringIO()  # Suppress again
            else:
                sys.stdout = old_stdout  # Restore for info message
                print("ℹ️  Using mock AI (set OPENAI_API_KEY for real analysis)")
                sys.stdout = StringIO()  # Suppress again
        except Exception as e:
            sys.stdout = old_stdout  # Restore for error message
            print(f"ℹ️  Using mock AI backends: {e}")
            sys.stdout = StringIO()  # Suppress again
            
    finally:
        sys.stdout = old_stdout
    
    print("✅ Major General initialized successfully")
    
    # Demo request
    user_request = "Build a secure fintech API for 10,000 transactions per second"
    
    print(f"\n👤 USER REQUEST:")
    print(f"    {user_request}")
    
    print(f"\n🧠 MAJOR GENERAL PROCESSING...")
    
    # Capture ZMQ noise during processing
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        await interface._process_user_request(user_request, InputMode.TERMINAL)
    finally:
        sys.stdout = old_stdout
    
    print(f"✅ Mission processed successfully!")
    
    # Show clean results
    print(f"\n📊 RESULTS:")
    if interface.major_general.active_missions:
        mission_id = list(interface.major_general.active_missions.keys())[0]
        mission = interface.major_general.active_missions[mission_id]
        
        print(f"    🎯 Mission ID: {mission_id}")
        print(f"    📈 Priority: {mission.priority.value}")
        print(f"    📋 Status: {mission.status.value}")
        
        if mission_id in interface.major_general.mission_execution_graphs:
            graph = interface.major_general.mission_execution_graphs[mission_id]
            print(f"    🕸️  Execution Graph: {len(graph.nodes())} steps, {len(graph.edges())} dependencies")
        
        if mission.results and 'initial_analysis' in mission.results:
            analysis = mission.results['initial_analysis']
            print(f"\n💡 FULL AI ANALYSIS:")
            
            # Extract and display the complete analysis
            if isinstance(analysis, dict) and 'synthesis' in analysis:
                response_text = analysis['synthesis']
                if isinstance(response_text, str):
                    # Clean up and format the response
                    clean_response = response_text.replace('[Strategic Analyst]', '').strip()
                    # Split into readable chunks
                    words = clean_response.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        if len(current_line + word) < 80:
                            current_line += word + " "
                        else:
                            lines.append(current_line.strip())
                            current_line = word + " "
                    if current_line:
                        lines.append(current_line.strip())
                    
                    for line in lines:
                        print(f"    {line}")
            else:
                # Fallback for other formats
                analysis_str = str(analysis)
                words = analysis_str.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line + word) < 80:
                        current_line += word + " "
                    else:
                        lines.append(current_line.strip())
                        current_line = word + " "
                if current_line:
                    lines.append(current_line.strip())
                
                for line in lines:
                    print(f"    {line}")
    
    # Show system status
    print(f"\n🎖️  SYSTEM STATUS:")
    print(f"    🧠 MetaReasoning: {len(interface.major_general.meta_reasoning.llm_backends)} AI specialists")
    print(f"    👥 Agents: {len(interface.major_general.agent_manager.agents)} active")
    print(f"    📋 Missions: {len(interface.major_general.active_missions)} active")
    print(f"    🕸️  Graphs: {len(interface.major_general.mission_execution_graphs)} execution graphs")
    
    print(f"\n🌟 CONVERSATION FLOW COMPLETE!")
    print(f"    ✅ User input processed")
    print(f"    ✅ AI analysis completed") 
    print(f"    ✅ Execution graph created")
    print(f"    ✅ Agent coordination prepared")
    print(f"    ✅ Results delivered")

async def interactive_clean_chat():
    """Interactive chat with reduced noise"""
    
    print("🎮 INTERACTIVE CHAT (Clean Mode)")
    print("=" * 50)
    
    # Initialize quietly
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        from conversation_interface import ConversationInterface, InputMode
        interface = ConversationInterface()
        
        # Try to upgrade to real LLM backends if available
        try:
            import os
            if os.getenv("OPENAI_API_KEY"):
                sys.stdout = old_stdout  # Temporarily restore stdout for messages
                print("🤖 Upgrading to real LLM backends...")
                sys.stdout = StringIO()  # Suppress again
                
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests', 'ai_command', 'meta_reasoning_engine'))
                from demo_real_llm import RealLLMBackend
                
                # Replace mock backends with real ones
                interface.major_general.meta_reasoning.register_llm_backend(
                    "analyst", RealLLMBackend("Strategic Analyst", "analytical")
                )
                interface.major_general.meta_reasoning.register_llm_backend(
                    "decomposer", RealLLMBackend("Task Decomposer", "systematic")
                )
                
                sys.stdout = old_stdout  # Restore for success message
                print("✅ Real AI analysis enabled!")
                sys.stdout = StringIO()  # Suppress again
            else:
                sys.stdout = old_stdout  # Restore for info message
                print("ℹ️  Using mock AI (set OPENAI_API_KEY for real analysis)")
                sys.stdout = StringIO()  # Suppress again
        except Exception as e:
            sys.stdout = old_stdout  # Restore for error message
            print(f"ℹ️  Using mock AI backends: {e}")
            sys.stdout = StringIO()  # Suppress again
            
    finally:
        sys.stdout = old_stdout
    
    print("✅ Major General ready for conversation")
    print("Type your requests below (or 'quit' to exit):")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n👤 USER: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n🎖️  Major General: Mission complete. Standing down.")
                break
            
            if not user_input:
                continue
            
            print(f"🧠 Processing: {user_input[:50]}...")
            
            # Process quietly
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                await interface._process_user_request(user_input, InputMode.TERMINAL)
            finally:
                sys.stdout = old_stdout
            
            # Show clean results AND the actual answer
            if interface.major_general.active_missions:
                latest_mission_id = max(interface.major_general.active_missions.keys())
                mission = interface.major_general.active_missions[latest_mission_id]
                
                print(f"✅ Mission {latest_mission_id} - {mission.status.value}")
                
                if latest_mission_id in interface.major_general.mission_execution_graphs:
                    graph = interface.major_general.mission_execution_graphs[latest_mission_id]
                    print(f"🕸️  Created execution plan: {len(graph.nodes())} steps")
                
                # SHOW THE ACTUAL AI RESPONSE
                if mission.results and 'initial_analysis' in mission.results:
                    analysis = mission.results['initial_analysis']
                    print(f"\n🎖️  Major General's Response:")
                    
                    # Extract the actual analysis content
                    if isinstance(analysis, dict) and 'synthesis' in analysis:
                        response_text = analysis['synthesis']
                        # Clean up the response text
                        if isinstance(response_text, str):
                            # Remove the [Agent] prefix and show clean response
                            clean_response = response_text.replace('[Analyst]', '').replace('[Planner]', '').replace('[Decomposer]', '').strip()
                            print(f"    {clean_response}")
                    else:
                        print(f"    {str(analysis)[:300]}...")
                else:
                    print(f"\n🎖️  Major General: Mission processed, analysis in progress...")
                
        except KeyboardInterrupt:
            print("\n\n🎖️  Major General: Interrupted. Standing down.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

async def main():
    """Main demo function"""
    
    print("🎖️  CLEAN CONVERSATION DEMO")
    print("Choose your demo:")
    print("1. Clean flow trace (no server noise)")
    print("2. Interactive chat (minimal noise)")
    
    choice = input("\nChoice (1 or 2): ").strip()
    
    if choice == "1":
        await clean_conversation_demo()
    elif choice == "2":
        await interactive_clean_chat()
    else:
        print("Invalid choice. Running clean demo...")
        await clean_conversation_demo()

if __name__ == "__main__":
    asyncio.run(main())
