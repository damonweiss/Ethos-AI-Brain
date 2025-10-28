"""
AI Engine - Minimalist Global AI Management
Creates ZMQ engine instance and spawns AgentZero
"""

import asyncio
import logging
import sys
import os
from typing import Optional, Dict, Any

# Add the Ethos-ZeroMQ path - TODO: Make this configurable or use proper package management
try:
    from ethos_zeromq import ZeroMQEngine
except ImportError:
    # Fallback to hardcoded path for development
    sys.path.append(
        os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'PycharmProjects', 'Ethos-ZeroMQ'))
    from ethos_zeromq import ZeroMQEngine
from .ai_agent import AIAgent
from ...orchestration.mcp.tool_management.mcp_tool_manager import MCPToolManager
from ...orchestration.mcp.server import create_development_server, create_production_server, create_server, ServerConfig
from pathlib import Path

logger = logging.getLogger(__name__)


class AIEngine:
    """
    AI Engine with MCP Server Management
    - Creates singleton ZMQ engine
    - Creates global MCP tool manager
    - Manages multi-headed MCP servers
    - Spawns AgentZero
    """

    def __init__(self):
        # Singleton ZMQ Engine
        self.zmq_engine = ZeroMQEngine(name="GlobalAIEngine")

        # Global MCP Tool Manager - singleton shared by all agents
        tools_dir = Path(__file__).parent.parent.parent / "orchestration" / "mcp_tools"
        self.mcp_manager = MCPToolManager.get_instance(tools_dir)
        
        # Discover tools at startup
        self.available_tools = self.mcp_manager.discover_tools()
        logger.info(f"Global MCP Manager initialized with {len(self.available_tools)} tools")

        # MCP Server instances - managed at AI Engine level
        self.mcp_servers = {}

        # AgentZero instance
        self.agent_zero: Optional[AIAgent] = None

        logger.info("AI Engine initialized")

    async def initialize(self):
        """Initialize AI Engine, start MCP servers, and spawn AgentZero"""
        try:
            # Start default development server
            tools_dir = Path(__file__).parent.parent.parent / "orchestration" / "mcp_tools"
            dev_server = create_development_server(
                port=8051,
                tools_dir=tools_dir,
                tool_registry=self.mcp_manager
            )
            await dev_server.start()
            self.mcp_servers["development"] = dev_server
            
            # Start production server
            prod_server = create_production_server(
                port=8052,
                tools_dir=tools_dir,
                tool_registry=self.mcp_manager
            )
            await prod_server.start()
            self.mcp_servers["production"] = prod_server
            
            # Spawn AgentZero
            await self.spawn_agent_zero()
            
            logger.info("AI Engine fully initialized with MCP servers")
        except Exception as e:
            logger.error(f"AI Engine initialization failed: {e}")
            raise

    async def spawn_agent_zero(self):
        """Spawn AgentZero as AIAgent instance"""
        if self.agent_zero:
            logger.warning("AgentZero already exists")
            return

        try:
            # Create AgentZero as AIAgent with access to global MCP manager
            self.agent_zero = AIAgent(
                agent_id="agent_zero",
                role="primary_agent",
                zmq_engine=self.zmq_engine,
                base_port=7000,
                mcp_manager=self.mcp_manager  # Pass global MCP manager
            )

            # Start AgentZero
            await self.agent_zero.start()

            logger.info("AgentZero spawned successfully")

        except Exception as e:
            logger.error(f"Failed to spawn AgentZero: {e}")
            raise

    async def delegate_to_agent_zero(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate task to AgentZero"""
        if not self.agent_zero:
            raise RuntimeError("AgentZero not available")

        return await self.agent_zero.execute_task(task)

    async def shutdown(self):
        """Shutdown AI Engine and MCP servers"""
        logger.info("Shutting down AI Engine...")

        # Stop MCP servers
        for name, server in self.mcp_servers.items():
            try:
                await server.stop()
                logger.info(f"Stopped MCP server: {name}")
            except Exception as e:
                logger.error(f"Error stopping server {name}: {e}")
        
        self.mcp_servers.clear()

        if self.agent_zero:
            await self.agent_zero.stop()

        self.zmq_engine.stop_all_servers()

        logger.info("AI Engine shutdown complete")
    
    def get_mcp_server_status(self) -> Dict[str, Dict]:
        """Get status of all MCP servers"""
        status = {}
        for name, server in self.mcp_servers.items():
            try:
                health = server.health_check()
                status[name] = {
                    "protocol": server.protocol_name,
                    "host": server.config.host,
                    "port": server.config.port,
                    "running": server.is_running,
                    "health": health
                }
            except Exception as e:
                status[name] = {
                    "error": str(e),
                    "running": False
                }
        return status
    
    async def start_additional_server(self, name: str, protocol: str, port: int) -> bool:
        """Start an additional MCP server"""
        if name in self.mcp_servers:
            logger.warning(f"Server {name} already exists")
            return False
        
        try:
            tools_dir = Path(__file__).parent.parent.parent / "orchestration" / "mcp_tools"
            server = create_server(
                protocol=protocol,
                host="localhost",
                port=port,
                tools_dir=tools_dir,
                tool_registry=self.mcp_manager
            )
            await server.start()
            self.mcp_servers[name] = server
            logger.info(f"Started additional MCP server: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to start server {name}: {e}")
            return False


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
