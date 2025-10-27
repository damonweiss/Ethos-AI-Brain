#!/usr/bin/env python3
"""
Stop MCP Server Script

Stops any running MCP server processes.
"""

import subprocess
import sys
import time

def stop_mcp_server():
    """Stop the MCP server by killing Python processes on port 8051."""
    print("Stopping MCP Server...")
    
    try:
        # Find processes using port 8051
        print("🔍 Looking for processes on port 8051...")
        
        if sys.platform == "win32":
            # Windows
            result = subprocess.run(
                ["netstat", "-ano"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            pids_to_kill = []
            for line in result.stdout.split('\n'):
                if ':8051' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        if pid.isdigit():
                            pids_to_kill.append(pid)
            
            if pids_to_kill:
                for pid in pids_to_kill:
                    print(f"🎯 Killing process PID: {pid}")
                    subprocess.run(["taskkill", "/PID", pid, "/F"], check=False)
                
                print(f"✅ Stopped {len(pids_to_kill)} MCP server process(es)")
            else:
                print("ℹ️  No MCP server processes found on port 8051")
        
        else:
            # Unix/Linux/Mac
            result = subprocess.run(
                ["lsof", "-ti:8051"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid.strip():
                        print(f"🎯 Killing process PID: {pid}")
                        subprocess.run(["kill", "-9", pid], check=False)
                
                print(f"✅ Stopped {len(pids)} MCP server process(es)")
            else:
                print("ℹ️  No MCP server processes found on port 8051")
        
        # Also try to kill any python processes that might be MCP servers
        print("🔍 Looking for Python MCP server processes...")
        
        if sys.platform == "win32":
            # Kill any python processes running the MCP server module
            subprocess.run([
                "taskkill", "/F", "/FI", 
                "IMAGENAME eq python.exe", "/FI", 
                "WINDOWTITLE eq *mcp.server*"
            ], check=False, capture_output=True)
        
        time.sleep(1)  # Give processes time to stop
        print("✅ MCP Server stop completed")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: {e}")
        print("✅ MCP Server stop completed (with warnings)")
    except Exception as e:
        print(f"❌ Error stopping MCP Server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(stop_mcp_server())
