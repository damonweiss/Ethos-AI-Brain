#!/usr/bin/env python3
"""
Start MCP Server Script

Starts the MCP server on port 8051.
"""

import subprocess
import sys
import os
from pathlib import Path

def start_mcp_server():
    """Start the MCP server."""
    print("Starting MCP Server on port 8051...")
    
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Change to project directory
    os.chdir(project_root)
    
    try:
        # Start MCP server
        cmd = [sys.executable, "-m", "src.ethos_mcp.mcp.server"]
        print(f"Running: {' '.join(cmd)}")
        
        # Start the server process
        process = subprocess.Popen(cmd, cwd=project_root)
        
        print(f"✅ MCP Server started with PID: {process.pid}")
        print(f"🌐 Server running at: http://localhost:8051")
        print(f"📊 Health check: http://localhost:8051/health")
        print(f"🔧 Tools list: http://localhost:8051/tools")
        print("\n⚠️  Press Ctrl+C to stop the server")
        
        # Wait for the process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping MCP Server...")
        process.terminate()
        print("✅ MCP Server stopped")
    except Exception as e:
        print(f"❌ Error starting MCP Server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(start_mcp_server())
