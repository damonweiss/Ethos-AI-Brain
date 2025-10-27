#!/usr/bin/env python3
"""
Stop Flask Web App Script

Stops any running Flask web application processes.
"""

import subprocess
import sys
import time

def stop_flask_app():
    """Stop the Flask web app by killing processes on port 8050."""
    print("Stopping Flask Web App...")
    
    try:
        # Find processes using port 8050
        print("🔍 Looking for processes on port 8050...")
        
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
                if ':8050' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        if pid.isdigit():
                            pids_to_kill.append(pid)
            
            if pids_to_kill:
                for pid in pids_to_kill:
                    print(f"🎯 Killing process PID: {pid}")
                    subprocess.run(["taskkill", "/PID", pid, "/F"], check=False)
                
                print(f"✅ Stopped {len(pids_to_kill)} Flask app process(es)")
            else:
                print("ℹ️  No Flask app processes found on port 8050")
        
        else:
            # Unix/Linux/Mac
            result = subprocess.run(
                ["lsof", "-ti:8050"], 
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
                
                print(f"✅ Stopped {len(pids)} Flask app process(es)")
            else:
                print("ℹ️  No Flask app processes found on port 8050")
        
        # Also try to kill any python processes that might be Flask apps
        print("🔍 Looking for Python Flask processes...")
        
        if sys.platform == "win32":
            # Kill any python processes running Flask
            subprocess.run([
                "taskkill", "/F", "/FI", 
                "IMAGENAME eq python.exe", "/FI", 
                "WINDOWTITLE eq *flask*"
            ], check=False, capture_output=True)
        
        time.sleep(1)  # Give processes time to stop
        print("✅ Flask Web App stop completed")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: {e}")
        print("✅ Flask Web App stop completed (with warnings)")
    except Exception as e:
        print(f"❌ Error stopping Flask Web App: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(stop_flask_app())
