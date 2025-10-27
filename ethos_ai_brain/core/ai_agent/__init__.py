"""
AI Agent System - Core AI agent management for Ethos AI Brain

This module provides the foundational AI agent capabilities including:
- Individual AI agents with knowledge graph integration
- AI brain for reasoning and knowledge management  
- AI engine for global agent orchestration
- AI agent orchestrator for complex multi-agent scenarios
"""

from .ai_agent import AIAgent
from .ai_brain import AIBrain
from .ai_engine import AIEngine, get_ai_engine, initialize_ai_engine
from .ai_agent_orchestrator import AIAgentOrchestrator, ManagedAgent, AgentType, AgentStatus

__all__ = [
    'AIAgent',
    'AIBrain', 
    'AIEngine',
    'get_ai_engine',
    'initialize_ai_engine',
    'AIAgentOrchestrator',
    'ManagedAgent',
    'AgentType',
    'AgentStatus'
]

__version__ = "0.1.0"
