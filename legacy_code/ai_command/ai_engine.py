"""
AI Engine - Minimalist Global AI Management
Creates ZMQ engine instance and spawns AgentZero
"""

import asyncio
import logging
import sys
import os
from typing import Optional, Dict, Any

# Add the Ethos-ZeroMQ path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'PycharmProjects', 'Ethos-ZeroMQ'))

from ethos_zeromq import ZeroMQEngine
from ai_agent import AIAgent

logger = logging.getLogger(__name__)


class AIEngine:
    """
    Minimalist AI Engine
    - Creates singleton ZMQ engine
    - Spawns AgentZero
    """
    
    def __init__(self):
        # Singleton ZMQ Engine
        self.zmq_engine = ZeroMQEngine(name="GlobalAIEngine")
        
        # AgentZero instance
        self.agent_zero: Optional[AIAgent] = None
        
        logger.info("🤖 AI Engine initialized")
    
    async def initialize(self):
        """Initialize AI Engine and spawn AgentZero"""
        try:
            # Spawn AgentZero
            await self.spawn_agent_zero()
            
            logger.info("✅ AI Engine fully initialized")
            
        except Exception as e:
            logger.error(f"❌ AI Engine initialization failed: {e}")
            raise
    
    async def spawn_agent_zero(self):
        """Spawn AgentZero as AIAgent instance"""
        if self.agent_zero:
            logger.warning("AgentZero already exists")
            return
        
        try:
            # Create AgentZero as AIAgent
            self.agent_zero = AIAgent(
                agent_id="agent_zero",
                role="primary_agent",
                shared_zmq_engine=self.zmq_engine,
                base_port=7000,
                capabilities=["reasoning", "orchestration", "llm_processing", "mcp_tools"]
            )
            
            # Start AgentZero
            await self.agent_zero.start()
            
            logger.info("🤖 AgentZero spawned successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to spawn AgentZero: {e}")
            raise
    
    async def delegate_to_agent_zero(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate task to AgentZero"""
        if not self.agent_zero:
            raise RuntimeError("AgentZero not available")
        
        return await self.agent_zero.execute_task(task)
    
    async def shutdown(self):
        """Shutdown AI Engine"""
        logger.info("🛑 Shutting down AI Engine...")
        
        if self.agent_zero:
            await self.agent_zero.stop()
        
        self.zmq_engine.stop_all_servers()
        
        logger.info("✅ AI Engine shutdown complete")


# Global AI Engine instance
_global_ai_engine: Optional[AIEngine] = None


def get_ai_engine() -> AIEngine:
    """Get global AI Engine instance"""
    global _global_ai_engine
    if _global_ai_engine is None:
        _global_ai_engine = AIEngine()
    return _global_ai_engine


async def initialize_ai_engine() -> AIEngine:
    """Initialize global AI Engine"""
    engine = get_ai_engine()
    await engine.initialize()
    return engine
