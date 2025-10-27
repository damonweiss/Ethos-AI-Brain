#!/usr/bin/env python3
"""
Stop All Servers Script

Stops both Flask Web App and MCP Server processes.
"""

import subprocess
import sys
import time

def stop_all_servers():
    """Stop all servers (Flask and MCP)."""
    print("🛑 Stopping All Servers...")
    print("=" * 40)
    
    success_count = 0
    total_count = 2
    
    try:
        # Stop Flask Web App first
        print("1️⃣  Stopping Flask Web App...")
        try:
            result = subprocess.run([sys.executable, "stop_flask.py"], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("   ✅ Flask Web App stopped successfully")
                success_count += 1
            else:
                print(f"   ⚠️  Flask stop completed with warnings")
                success_count += 1
        except Exception as e:
            print(f"   ❌ Error stopping Flask Web App: {e}")
        
        print()
        
        # Stop MCP Server
        print("2️⃣  Stopping MCP Server...")
        try:
            result = subprocess.run([sys.executable, "stop_mcp.py"], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("   ✅ MCP Server stopped successfully")
                success_count += 1
            else:
                print(f"   ⚠️  MCP stop completed with warnings")
                success_count += 1
        except Exception as e:
            print(f"   ❌ Error stopping MCP Server: {e}")
        
        print()
        
        # Additional cleanup - kill any remaining Python processes on our ports
        print("3️⃣  Additional cleanup...")
        
        if sys.platform == "win32":
            # Windows cleanup
            ports_to_clean = [8050, 8051]
            
            for port in ports_to_clean:
                try:
                    # Find processes on port
                    result = subprocess.run(
                        ["netstat", "-ano"], 
                        capture_output=True, text=True, check=True
                    )
                    
                    pids_killed = 0
                    for line in result.stdout.split('\n'):
                        if f':{port}' in line and 'LISTENING' in line:
                            parts = line.split()
                            if len(parts) >= 5:
                                pid = parts[-1]
                                if pid.isdigit():
                                    subprocess.run(["taskkill", "/PID", pid, "/F"], 
                                                 check=False, capture_output=True)
                                    pids_killed += 1
                    
                    if pids_killed > 0:
                        print(f"   🧹 Cleaned up {pids_killed} process(es) on port {port}")
                
                except Exception:
                    pass  # Ignore cleanup errors
        
        else:
            # Unix/Linux/Mac cleanup
            for port in [8050, 8051]:
                try:
                    result = subprocess.run(
                        ["lsof", f"-ti:{port}"], 
                        capture_output=True, text=True, check=False
                    )
                    
                    if result.stdout.strip():
                        pids = result.stdout.strip().split('\n')
                        for pid in pids:
                            if pid.strip():
                                subprocess.run(["kill", "-9", pid], check=False)
                        print(f"   🧹 Cleaned up {len(pids)} process(es) on port {port}")
                
                except Exception:
                    pass  # Ignore cleanup errors
        
        # Final status
        print("=" * 40)
        if success_count == total_count:
            print("🎉 All servers stopped successfully!")
        elif success_count > 0:
            print(f"⚠️  {success_count}/{total_count} servers stopped (some with warnings)")
        else:
            print("❌ Failed to stop servers cleanly")
        
        print("✅ Cleanup completed")
        
        # Verify ports are free
        time.sleep(2)
        print("\n🔍 Verifying ports are free...")
        
        if sys.platform == "win32":
            result = subprocess.run(
                ["netstat", "-an"], 
                capture_output=True, text=True, check=False
            )
            
            port_8050_free = ":8050" not in result.stdout
            port_8051_free = ":8051" not in result.stdout
            
            print(f"   Port 8050 (Flask): {'✅ Free' if port_8050_free else '⚠️  Still in use'}")
            print(f"   Port 8051 (MCP):   {'✅ Free' if port_8051_free else '⚠️  Still in use'}")
        
        return 0 if success_count == total_count else 1
        
    except Exception as e:
        print(f"❌ Error during server shutdown: {e}")
        return 1

if __name__ == "__main__":
    exit(stop_all_servers())
