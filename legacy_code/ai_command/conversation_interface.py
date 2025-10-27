#!/usr/bin/env python3
"""
Conversation Interface - Multiple input methods for Major General
Handles: Terminal, Web Chat, Voice, API
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

from major_general import MajorGeneral

# Reduce ZMQ logging noise
logging.getLogger('ai_agent_manager').setLevel(logging.WARNING)
logging.getLogger('major_general').setLevel(logging.WARNING)
logging.getLogger('meta_reasoning_engine').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class InputMode(Enum):
    TERMINAL = "terminal"
    WEB_CHAT = "web_chat"
    VOICE = "voice"
    API = "api"

class ConversationInterface:
    """Main interface for user conversations with Major General"""
    
    def __init__(self):
        # Suppress ZMQ noise during initialization
        import sys
        from io import StringIO
        
        # Temporarily capture stdout to reduce ZMQ initialization noise
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            self.major_general = MajorGeneral()
        finally:
            # Restore stdout
            sys.stdout = old_stdout
        
        self.conversation_history = []
        self.active_sessions = {}
        
    async def start_conversation(self, mode: InputMode = InputMode.TERMINAL):
        """Start conversation in specified mode"""
        
        print("=" * 70)
        print("🎖️  MAJOR GENERAL AI COMMAND SYSTEM")
        print("=" * 70)
        print("Welcome! I'm your AI Military Commander.")
        print("I can help you plan, analyze, and execute complex missions.")
        print("Type 'help' for commands, 'quit' to exit.")
        print("-" * 70)
        
        if mode == InputMode.TERMINAL:
            await self._terminal_conversation()
        elif mode == InputMode.WEB_CHAT:
            await self._web_chat_conversation()
        elif mode == InputMode.VOICE:
            await self._voice_conversation()
        else:
            await self._api_conversation()
    
    async def _terminal_conversation(self):
        """Terminal-based conversation loop"""
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 USER: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🎖️  Major General: Mission complete. Standing down.")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                if user_input.lower() == 'status':
                    await self._show_status()
                    continue
                
                if not user_input:
                    continue
                
                # Process with Major General
                await self._process_user_request(user_input, InputMode.TERMINAL)
                
            except KeyboardInterrupt:
                print("\n\n🎖️  Major General: Interrupted. Standing down.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Conversation error: {e}")
    
    async def _process_user_request(self, user_input: str, mode: InputMode):
        """Process user request through Major General's brain"""
        
        # Record conversation
        conversation_entry = {
            "timestamp": datetime.now(),
            "user_input": user_input,
            "mode": mode.value,
            "session_id": "terminal_session"
        }
        
        print(f"\n🎖️  Major General: Analyzing your request...")
        print(f"    📝 Mission: {user_input}")
        
        try:
            # STEP 1: MG receives mission
            print(f"\n🧠 [COGNITIVE ANALYSIS] Major General thinking...")
            mission_id = await self.major_general.receive_mission(user_input)
            
            print(f"    ✅ Mission {mission_id} received and analyzed")
            
            # STEP 2: Show what MG is doing
            await self._show_mg_thinking_process(mission_id)
            
            # STEP 3: Get mission status and results
            mission = self.major_general.active_missions.get(mission_id)
            if mission:
                print(f"\n📊 [MISSION STATUS]")
                print(f"    🎯 Objective: {mission.objective}")
                print(f"    📈 Priority: {mission.priority.value}")
                print(f"    📋 Status: {mission.status.value}")
                
                # Show execution graph if available
                if mission_id in self.major_general.mission_execution_graphs:
                    graph = self.major_general.mission_execution_graphs[mission_id]
                    print(f"    🕸️  Execution Graph: {len(graph.nodes())} steps, {len(graph.edges())} dependencies")
                
                # Show results if available
                if mission.results:
                    print(f"\n🎯 [INITIAL ANALYSIS]")
                    if 'initial_analysis' in mission.results:
                        analysis = mission.results['initial_analysis']
                        print(f"    💡 Analysis: {str(analysis)[:200]}...")
            
            # STEP 4: Show agent activity
            await self._show_agent_activity()
            
            # Record response
            conversation_entry["mission_id"] = mission_id
            conversation_entry["mg_response"] = "Mission analyzed and processed"
            self.conversation_history.append(conversation_entry)
            
        except Exception as e:
            print(f"\n❌ Major General encountered an error: {e}")
            conversation_entry["error"] = str(e)
            self.conversation_history.append(conversation_entry)
    
    async def _show_mg_thinking_process(self, mission_id: str):
        """Show Major General's thinking process"""
        
        print(f"\n🧠 [MG BRAIN ACTIVITY]")
        
        # Show meta-reasoning activity
        if hasattr(self.major_general, 'meta_reasoning'):
            print(f"    🤖 MetaReasoning: Analyzing mission complexity...")
            print(f"    📊 Strategic Analysis: Identifying key components...")
            print(f"    🔄 Task Decomposition: Breaking down into actionable steps...")
        
        # Show NetworkX activity
        print(f"    🕸️  NetworkX: Creating execution graph...")
        print(f"    📈 Graph Optimization: Finding optimal execution paths...")
        
        # Show ZMQ activity
        print(f"    ⚡ ZMQ Network: Preparing agent coordination...")
        
        # Simulate some processing time
        await asyncio.sleep(1)
    
    async def _show_agent_activity(self):
        """Show current agent activity"""
        
        print(f"\n👥 [AGENT STATUS]")
        
        agents = list(self.major_general.agent_manager.agents.values())
        if agents:
            print(f"    📊 Active Agents: {len(agents)}")
            for agent in agents[:3]:  # Show first 3
                print(f"    🤖 {agent.id}: {agent.agent_type.value} - {agent.status.value}")
        else:
            print(f"    📊 No active agents (MG operating independently)")
        
        # Show ZMQ network status
        if hasattr(self.major_general.agent_manager, 'zmq_engine'):
            print(f"    ⚡ ZMQ Network: Operational")
    
    async def _show_status(self):
        """Show system status"""
        
        print(f"\n📊 [SYSTEM STATUS]")
        print(f"    🎖️  Major General: Online and operational")
        print(f"    🧠 MetaReasoning: {len(self.major_general.meta_reasoning.llm_backends)} LLM backends")
        print(f"    👥 Agents: {len(self.major_general.agent_manager.agents)} active")
        print(f"    📋 Missions: {len(self.major_general.active_missions)} active")
        print(f"    🕸️  Graphs: {len(self.major_general.mission_execution_graphs)} execution graphs")
        print(f"    💬 Conversations: {len(self.conversation_history)} exchanges")
    
    def _show_help(self):
        """Show help information"""
        
        print(f"\n📖 [HELP - MAJOR GENERAL COMMANDS]")
        print(f"    🎯 Mission Examples:")
        print(f"       'Build a secure API for fintech'")
        print(f"       'Analyze market trends for Q4'") 
        print(f"       'Create deployment plan for microservices'")
        print(f"       'Design security architecture for healthcare app'")
        print(f"")
        print(f"    🔧 System Commands:")
        print(f"       'status' - Show system status")
        print(f"       'help' - Show this help")
        print(f"       'quit' - Exit conversation")
        print(f"")
        print(f"    💡 Tips:")
        print(f"       - Be specific about requirements and constraints")
        print(f"       - Mention budget, timeline, team size if relevant")
        print(f"       - Ask for analysis, planning, or execution help")

# Web Chat Interface (placeholder for future)
class WebChatInterface:
    """Web-based chat interface"""
    
    def __init__(self, conversation_interface: ConversationInterface):
        self.conversation_interface = conversation_interface
    
    async def handle_web_message(self, message: str, session_id: str):
        """Handle web chat message"""
        # This would integrate with Flask/FastAPI web framework
        response = await self.conversation_interface._process_user_request(
            message, InputMode.WEB_CHAT
        )
        return response

# Voice Interface (placeholder for future)
class VoiceInterface:
    """Voice-based interface"""
    
    def __init__(self, conversation_interface: ConversationInterface):
        self.conversation_interface = conversation_interface
    
    async def handle_voice_input(self, audio_data: bytes):
        """Handle voice input"""
        # This would integrate with speech-to-text
        # text = speech_to_text(audio_data)
        # response = await self.conversation_interface._process_user_request(text, InputMode.VOICE)
        # audio_response = text_to_speech(response)
        # return audio_response
        pass

# Demo function
async def demo_conversation():
    """Demo conversation with Major General"""
    
    interface = ConversationInterface()
    
    print("🎯 DEMO: Conversation with Major General")
    print("This demonstrates the complete flow from user input to MG response")
    
    # Simulate user inputs
    demo_inputs = [
        "Build a secure fintech API that can handle 10,000 transactions per second",
        "Create a deployment strategy for a microservices architecture",
        "Analyze the security requirements for a healthcare application"
    ]
    
    for user_input in demo_inputs:
        print(f"\n" + "="*70)
        print(f"👤 DEMO USER: {user_input}")
        await interface._process_user_request(user_input, InputMode.TERMINAL)
        await asyncio.sleep(2)  # Pause between demos

if __name__ == "__main__":
    # Run terminal conversation
    interface = ConversationInterface()
    asyncio.run(interface.start_conversation(InputMode.TERMINAL))
