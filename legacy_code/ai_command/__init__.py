"""
AI Command System for dtOS
Military-inspired AI agent orchestration with Major General command structure
"""

from .major_general import MajorGeneral
from .ai_agent_manager import AIAgentManager
from .mission_parser import MissionParser, Mission
from .zmq_command_bridge import ZMQCommandBridge

__version__ = "0.1.0"
__all__ = [
    "MajorGeneral",
    "AIAgentManager", 
    "MissionParser",
    "Mission",
    "ZMQCommandBridge"
]
