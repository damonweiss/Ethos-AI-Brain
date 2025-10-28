#!/usr/bin/env python3
"""
Quick test of AI Engine MCP server integration
"""

import asyncio
import aiohttp
from ethos_ai_brain.core.ai_agent.ai_engine import AIEngine


async def test_ai_engine_mcp_integration():
    """Test that AI Engine can start MCP servers."""
    print("🚀 Testing AI Engine MCP Integration...")
    
    # Create AI Engine
    engine = AIEngine()
    
    try:
        # Initialize (this should start MCP servers)
        await engine.initialize()
        print("✅ AI Engine initialized successfully")
        
        # Check server status
        server_status = engine.get_mcp_server_status()
        print(f"📊 Server Status: {server_status}")
        
        # Test health endpoint
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8051/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"✅ Health check passed: {health_data}")
                else:
                    print(f"❌ Health check failed: {response.status}")
        
        # Test tools endpoint
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8051/tools") as response:
                if response.status == 200:
                    tools_data = await response.json()
                    print(f"✅ Tools endpoint working: {tools_data.get('count', 0)} tools found")
                else:
                    print(f"❌ Tools endpoint failed: {response.status}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        # Cleanup
        await engine.shutdown()
        print("🧹 AI Engine shutdown complete")


if __name__ == "__main__":
    asyncio.run(test_ai_engine_mcp_integration())
