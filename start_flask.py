#!/usr/bin/env python3
"""
Start Flask Web App Script

Creates and starts a Flask web application with MCP integration on port 8050.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from flask import Flask, render_template_string
from ethos_mcp.flask_integration.dynamic_endpoints import mcp_blueprint

def create_app():
    """Create Flask application with MCP integration."""
    app = Flask(__name__)
    app.config['DEBUG'] = True
    
    # Register MCP blueprint
    app.register_blueprint(mcp_blueprint, url_prefix='/api/mcp')
    
    # Simple dashboard template
    dashboard_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MCP Tools Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            button { padding: 10px 20px; margin: 10px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            #results { margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 5px; white-space: pre-wrap; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .success { background: #d4edda; color: #155724; }
            .error { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <h1>🔧 MCP Tools Dashboard</h1>
        <p>Web interface for MCP server tools</p>
        
        <div>
            <button onclick="checkHealth()">Check Server Health</button>
            <button onclick="listTools()">List Available Tools</button>
            <button onclick="echoTest()">Test Echo Tool</button>
            <button onclick="getProjectInfo()">Get Project Info</button>
            <button onclick="getSystemTime()">Get System Time</button>
        </div>
        
        <div id="status"></div>
        <div id="results"></div>
        
        <script>
            const statusDiv = document.getElementById('status');
            const resultsDiv = document.getElementById('results');
            
            function showStatus(message, isError = false) {
                statusDiv.innerHTML = `<div class="status ${isError ? 'error' : 'success'}">${message}</div>`;
            }
            
            function showResults(data) {
                resultsDiv.innerHTML = JSON.stringify(data, null, 2);
            }
            
            async function makeRequest(url, options = {}) {
                try {
                    showStatus('Making request...');
                    const response = await fetch(url, options);
                    const data = await response.json();
                    
                    if (response.ok) {
                        showStatus(`✅ Success (${response.status})`);
                    } else {
                        showStatus(`⚠️ Warning (${response.status})`, true);
                    }
                    
                    showResults(data);
                } catch (error) {
                    showStatus(`❌ Error: ${error.message}`, true);
                    showResults({error: error.message});
                }
            }
            
            async function checkHealth() {
                await makeRequest('/api/mcp/health');
            }
            
            async function listTools() {
                await makeRequest('/api/mcp/tools');
            }
            
            async function echoTest() {
                await makeRequest('/api/mcp/tools/echo_message', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: "Hello from Flask Dashboard!"})
                });
            }
            
            async function getProjectInfo() {
                await makeRequest('/api/mcp/tools/get_project_info', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({})
                });
            }
            
            async function getSystemTime() {
                await makeRequest('/api/mcp/tools/get_system_time', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({})
                });
            }
            
            // Check health on page load
            window.onload = checkHealth;
        </script>
    </body>
    </html>
    """
    
    @app.route('/')
    def dashboard():
        return render_template_string(dashboard_template)
    
    return app

def start_flask_app():
    """Start the Flask web application."""
    print("Starting Flask Web App on port 8050...")
    
    try:
        app = create_app()
        
        print("✅ Flask app created successfully")
        print("🌐 Web dashboard: http://localhost:8050")
        print("🔧 API endpoints: http://localhost:8050/api/mcp/*")
        print("📊 Health check: http://localhost:8050/api/mcp/health")
        print("\n⚠️  Make sure MCP Server is running on port 8051")
        print("⚠️  Press Ctrl+C to stop the server")
        
        # Start the Flask app
        app.run(host='0.0.0.0', port=8050, debug=True, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping Flask Web App...")
        print("✅ Flask Web App stopped")
    except Exception as e:
        print(f"❌ Error starting Flask Web App: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(start_flask_app())
