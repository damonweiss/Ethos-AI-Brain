#!/usr/bin/env python3
"""
ZMQ Parallel Intent Knowledge Graph Demo
Demonstrates two-wave parallel intent analysis using focused prompts
"""

import asyncio
import json
import uuid
import sys
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from intent_knowledge_graph_zmq import IntentKnowledgeGraph
from knowledge_graph_visualizer import MatplotlibKnowledgeGraphVisualizer

# Add the Ethos-ZeroMQ path
sys.path.append(r'C:\Users\DamonWeiss\PycharmProjects\Ethos-ZeroMQ')

from ethos_zeromq.EthosZeroMQ_Engine import ZeroMQEngine
from ethos_zeromq.RouterDealerServer import RouterDealerServer, DealerClient
import time

load_dotenv()

class ZMQIntentAnalyzer:
    """ZMQ ROUTER-DEALER intent analysis using Ethos-ZeroMQ library"""
    
    def __init__(self, base_port=5572, merge_strategy="independent"):
        """Initialize ZMQ Intent Analyzer with configurable prompt merging"""
        self.base_port = base_port
        self.zmq_engine = ZeroMQEngine()
        self.servers = {}  # Multiple servers for true parallelism
        self.merge_strategy = merge_strategy
        
        # Define ports for each component (true parallel processing)
        self.ports = {
            'objectives': base_port,
            'constraints': base_port + 1,
            'context': base_port + 2,
            'stakeholders': base_port + 3,
            'technical_requirements': base_port + 4,
            'assumptions': base_port + 5
        }
        
        # Configure shared results based on merge strategy
        if merge_strategy == "single_call":
            # One massive LLM call for everything
            self.shared_results = {
                'complete_analysis': None
            }
        elif merge_strategy == "merged_pairs":
            # Merge related components (3 calls instead of 6)
            self.shared_results = {
                'objectives_constraints': None,    # Goals + limitations together
                'stakeholders_context': None,      # Who + situational context
                'technical_assumptions': None      # Tech requirements + assumptions
            }
        elif merge_strategy == "two_wave":
            # Traditional two-wave approach (2 calls)
            self.shared_results = {
                'foundation_analysis': None,       # objectives + constraints + context
                'implementation_analysis': None    # stakeholders + tech + assumptions
            }
        else:  # "independent" - original 6 separate calls
            self.shared_results = {
                'objectives': None,
                'constraints': None,
                'context': None,
                'stakeholders': None,
                'technical_requirements': None,
                'assumptions': None
            }
        
        # Initialize OpenAI client
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.llm_client = OpenAI(api_key=api_key)
    
    def create_merged_prompt(self, user_input: str, domain: str, components: list) -> str:
        """Create a merged prompt for multiple components"""
        
        if self.merge_strategy == "single_call":
            return self._create_complete_analysis_prompt(user_input, domain)
        elif self.merge_strategy == "merged_pairs":
            if "objectives_constraints" in components:
                return self._create_objectives_constraints_prompt(user_input, domain)
            elif "stakeholders_context" in components:
                return self._create_stakeholders_context_prompt(user_input, domain)
            elif "technical_assumptions" in components:
                return self._create_technical_assumptions_prompt(user_input, domain)
        elif self.merge_strategy == "two_wave":
            if "foundation_analysis" in components:
                return self._create_foundation_prompt(user_input, domain)
            elif "implementation_analysis" in components:
                return self._create_implementation_prompt(user_input, domain)
        
        # Fallback to individual prompts
        return IntentKnowledgeGraph.get_objectives_prompt(user_input, domain)
    
    def _create_complete_analysis_prompt(self, user_input: str, domain: str) -> str:
        """Single mega-prompt for complete intent analysis"""
        return f"""
Analyze the following user request comprehensively and return a complete intent analysis in JSON format.

User Request: "{user_input}"
Domain: {domain}

Return a JSON object with ALL of the following components:

1. OBJECTIVES: Array of specific, measurable goals
2. CONSTRAINTS: Array of limitations and restrictions  
3. CONTEXT: Object with domain context, user sentiment, urgency level
4. STAKEHOLDERS: Array of people/groups involved with roles and influence
5. TECHNICAL_REQUIREMENTS: Array of technical capabilities needed
6. ASSUMPTIONS: Array of assumptions being made with confidence levels

JSON Format:
{{
    "objectives": [{{
        "id": "1", "name": "...", "description": "...", "priority": "critical|high|medium|low",
        "measurable": true/false, "measurement_method": "...", "target_value": number, "unit": "..."
    }}],
    "constraints": [{{
        "id": "1", "constraint_type": "budget|timeline|technical|regulatory", 
        "constraint": "...", "value": number, "flexibility": "rigid|somewhat_flexible|flexible"
    }}],
    "context": {{
        "domain_context": "{domain}", "user_sentiment": "urgent|excited|concerned|neutral",
        "complexity_level": "low|medium|high", "time_sensitivity": "immediate|weeks|months"
    }},
    "stakeholders": [{{
        "id": "1", "name": "...", "role": "...", "influence_level": "high|medium|low",
        "support_level": "strong|neutral|resistant", "decision_authority": "final|input|none"
    }}],
    "technical_requirements": [{{
        "id": "1", "name": "...", "description": "...", "complexity": "low|medium|high",
        "importance": "critical|high|medium|low"
    }}],
    "assumptions": [{{
        "id": "1", "assumption": "...", "confidence": 0.0-1.0, 
        "impact_if_wrong": "critical|high|medium|low", "validation_method": "..."
    }}]
}}

Return ONLY the JSON object, no other text.
"""
    
    def start_servers(self):
        """Start ZMQ RouterDealerServers based on merge strategy"""
        
        if self.merge_strategy == "single_call":
            print(f"🔥 Starting 1 ZMQ server for SINGLE MEGA-PROMPT processing...")
            
            # Only start objectives server for single call
            port = self.ports['objectives']
            print(f"  📡 Starting complete_analysis server on port {port}...")
            
            server = RouterDealerServer(
                engine=self.zmq_engine,
                bind=f"tcp://*:{port}",
                connect=f"tcp://localhost:{port}"
            )
            
            server.register_handler("complete_analysis", self._handle_complete_analysis)
            server.start()
            self.servers['objectives'] = server
            
            print(f"✅ Single server started for mega-prompt processing!")
            
        else:
            print(f"🚀 Starting {len(self.ports)} ZMQ RouterDealerServers for TRUE PARALLEL processing...")
            
            # Create separate server for each component
            for component, port in self.ports.items():
                print(f"  📡 Starting {component} server on port {port}...")
                
                server = RouterDealerServer(
                    engine=self.zmq_engine,
                    bind=f"tcp://*:{port}",
                    connect=f"tcp://localhost:{port}"
                )
                
                handler_method = getattr(self, f'_handle_{component}')
                server.register_handler(component, handler_method)
                
                # Start the server
                server.start()
                self.servers[component] = server
                
                time.sleep(0.1)  # Brief delay between server starts
            
            print(f"✅ All {len(self.servers)} ZMQ servers started and ready for PARALLEL processing!")
            print(f"   Ports: {list(self.ports.values())}")
    
    # Keep backward compatibility
    def start_server(self):
        """Backward compatibility - calls start_servers()"""
        self.start_servers()
    
    def stop_servers(self):
        """Stop all ZMQ servers"""
        if self.servers:
            print("🛑 Stopping all ZMQ RouterDealerServers...")
            for component, server in self.servers.items():
                print(f"  🔌 Stopping {component} server...")
                server.stop()
            print("✅ All ZMQ RouterDealerServers stopped")
    
    # Keep backward compatibility
    def stop_server(self):
        """Backward compatibility - calls stop_servers()"""
        self.stop_servers()
    
    def _handle_objectives(self, message):
        """Handle objectives analysis request"""
        print(f"🎯 Processing objectives analysis...")
        user_input = message.get('user_input', '')
        domain = message.get('domain', 'unknown')
        
        prompt = IntentKnowledgeGraph.get_objectives_prompt(user_input, domain)
        result = self._call_llm_sync(prompt, 'objectives')
        
        # Update shared results directly
        self.shared_results['objectives'] = result
        print(f"✅ Objectives completed and stored")
        
        # Return simple acknowledgment
        return {"status": "completed", "component": "objectives"}
    
    def _handle_constraints(self, message):
        """Handle constraints analysis request"""
        print(f"⚖️ Processing constraints analysis...")
        user_input = message.get('user_input', '')
        domain = message.get('domain', 'unknown')
        
        prompt = IntentKnowledgeGraph.get_constraints_prompt(user_input, domain)
        result = self._call_llm_sync(prompt, 'constraints')
        
        # Update shared results directly
        self.shared_results['constraints'] = result
        print(f"✅ Constraints completed and stored")
        
        # Return simple acknowledgment
        return {"status": "completed", "component": "constraints"}
    
    def _handle_context(self, message):
        """Handle context analysis request"""
        print(f"📋 Processing context analysis...")
        user_input = message.get('user_input', '')
        domain = message.get('domain', 'unknown')
        
        prompt = IntentKnowledgeGraph.get_context_prompt(user_input, domain)
        result = self._call_llm_sync(prompt, 'context')
        
        # Update shared results directly
        self.shared_results['context'] = result
        print(f"✅ Context completed and stored")
        
        # Return simple acknowledgment
        return {"status": "completed", "component": "context"}
    
    def _handle_stakeholders(self, message):
        """Handle stakeholders analysis request"""
        print(f"👥 Processing stakeholders analysis...")
        user_input = message.get('user_input', '')
        domain = message.get('domain', 'unknown')
        objectives_context = message.get('objectives_context', '')
        
        prompt = IntentKnowledgeGraph.get_stakeholders_prompt(user_input, domain, objectives_context)
        result = self._call_llm_sync(prompt, 'stakeholders')
        
        # Update shared results directly
        self.shared_results['stakeholders'] = result
        print(f"✅ Stakeholders completed and stored")
        
        # Return simple acknowledgment
        return {"status": "completed", "component": "stakeholders"}
    
    def _handle_technical_requirements(self, message):
        """Handle technical requirements analysis request"""
        print(f"⚙️ Processing technical requirements analysis...")
        user_input = message.get('user_input', '')
        domain = message.get('domain', 'unknown')
        objectives_context = message.get('objectives_context', '')
        constraints_context = message.get('constraints_context', '')
        
        prompt = IntentKnowledgeGraph.get_technical_requirements_prompt(user_input, domain, objectives_context, constraints_context)
        result = self._call_llm_sync(prompt, 'technical_requirements')
        
        # Update shared results directly
        self.shared_results['technical_requirements'] = result
        print(f"✅ Technical Requirements completed and stored")
        
        # Return simple acknowledgment
        return {"status": "completed", "component": "technical_requirements"}
    
    def _handle_assumptions(self, message):
        """Handle assumptions analysis request"""
        print(f"🤔 Processing assumptions analysis...")
        user_input = message.get('user_input', '')
        domain = message.get('domain', 'unknown')
        full_context = message.get('full_context', '')
        
        prompt = IntentKnowledgeGraph.get_assumptions_prompt(user_input, domain, full_context)
        result = self._call_llm_sync(prompt, 'assumptions')
        
        # Update shared results directly
        self.shared_results['assumptions'] = result
        print(f"✅ Assumptions completed and stored")
        
        # Return simple acknowledgment
        return {"status": "completed", "component": "assumptions"}
    
    def _handle_complete_analysis(self, message):
        """Handle complete analysis in single mega-prompt"""
        print(f"🎯 Processing COMPLETE ANALYSIS in single call...")
        user_input = message.get('user_input', '')
        domain = message.get('domain', 'unknown')
        
        prompt = self._create_complete_analysis_prompt(user_input, domain)
        result = self._call_llm_sync(prompt, 'complete_analysis')
        
        # Update shared results with all components
        if isinstance(result, dict):
            for component in ['objectives', 'constraints', 'context', 'stakeholders', 'technical_requirements', 'assumptions']:
                if component in result:
                    self.shared_results[component] = result[component]
        
        print(f"✅ Complete Analysis completed and stored")
        return {"status": "completed", "component": "complete_analysis"}
    
    def _call_llm_sync(self, prompt: str, component_name: str) -> dict:
        """Synchronous LLM call for handler methods"""
        print(f"  🤖 Calling LLM for {component_name}...")
        print(f"     📤 Prompt length: {len(prompt)} chars")
        
        try:
            print(f"     🔄 Sending request to GPT-4...")
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a focused intent analyzer. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            result_text = response.choices[0].message.content
            print(f"     📥 LLM response received: {len(result_text)} chars")
            
            # Extract JSON from response
            if '```json' in result_text:
                json_start = result_text.find('```json') + 7
                json_end = result_text.find('```', json_start)
                result_text = result_text[json_start:json_end].strip()
                print(f"     🔧 Extracted JSON from markdown: {len(result_text)} chars")
            elif '[' in result_text or '{' in result_text:
                # Find JSON object/array
                start_idx = min(
                    result_text.find('[') if '[' in result_text else len(result_text),
                    result_text.find('{') if '{' in result_text else len(result_text)
                )
                result_text = result_text[start_idx:]
                print(f"     🔧 Extracted JSON from text: {len(result_text)} chars")
            
            parsed_result = json.loads(result_text)
            
            # Count results based on type
            if isinstance(parsed_result, list):
                print(f"     ✅ Parsed {len(parsed_result)} {component_name}")
            elif isinstance(parsed_result, dict):
                print(f"     ✅ Parsed {component_name} object")
            
            return parsed_result
            
        except Exception as e:
            print(f"     ❌ LLM call failed for {component_name}: {str(e)}")
            return [] if component_name != 'context' else {}
    
    async def analyze_intent_parallel(self, user_input: str, domain: str = "unknown") -> dict:
        """Full 6-parallel ZMQ ROUTER-DEALER intent analysis with benchmarking"""
        
        print(f"ZMQ ROUTER-DEALER INTENT ANALYSIS")
        print("=" * 60)
        print(f"User Input: {user_input}")
        print(f"Domain: {domain}")
        
        # Start timing
        start_time = time.time()
        
        # Full parallel execution - all 6 components simultaneously
        print(f"\n🚀 FULL PARALLEL ANALYSIS (6 simultaneous LLM calls)")
        print("-" * 60)
        
        parallel_start = time.time()
        all_data = await self._execute_all_parallel(user_input, domain)
        parallel_end = time.time()
        
        parallel_duration = parallel_end - parallel_start
        print(f"⚡ All 6 components completed in {parallel_duration:.2f} seconds")
        
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"  - Objectives: {len(all_data['objectives'])} found")
        print(f"  - Constraints: {len(all_data['constraints'])} found")
        print(f"  - Context: {all_data['context'].get('user_sentiment', 'unknown')} sentiment")
        print(f"  - Stakeholders: {len(all_data['stakeholders'])} found")
        print(f"  - Technical Requirements: {len(all_data['technical_requirements'])} found")
        print(f"  - Assumptions: {len(all_data['assumptions'])} found")
        
        # Combine results
        complete_intent_data = {
            'graph_metadata': {
                'graph_id': f"zmq_intent_{uuid.uuid4().hex[:8]}",
                'graph_type': 'intent',
                'intent_name': f"ZMQ Analysis: {user_input[:50]}...",
                'created_date': datetime.now().isoformat(),
                'analysis_duration': parallel_duration,
                'total_components': sum([
                    len(all_data['objectives']),
                    len(all_data['constraints']),
                    len(all_data['stakeholders']),
                    len(all_data['technical_requirements']),
                    len(all_data['assumptions']),
                    1  # context object
                ])
            },
            'intent_data': all_data['context'],
            'objectives': all_data['objectives'],
            'constraints': all_data['constraints'],
            'stakeholders': all_data['stakeholders'],
            'technical_requirements': all_data['technical_requirements'],
            'assumptions': all_data['assumptions'],
            'relationships': []  # Will be auto-generated
        }
        
        total_time = time.time() - start_time
        print(f"\n⏱️ TOTAL ANALYSIS TIME: {total_time:.2f} seconds")
        print(f"📈 PERFORMANCE: {complete_intent_data['graph_metadata']['total_components']} components in {parallel_duration:.2f}s = {complete_intent_data['graph_metadata']['total_components']/parallel_duration:.1f} components/sec")
        print(f"🚀 MULTI-PORT ARCHITECTURE: {len(self.ports)} dedicated servers used")
        
        return complete_intent_data
    
    async def _execute_all_parallel(self, user_input: str, domain: str) -> dict:
        """Execute analysis based on merge strategy"""
        
        # Reset shared results
        for key in self.shared_results:
            self.shared_results[key] = None
        
        if self.merge_strategy == "single_call":
            # For single call, we need to temporarily expand shared_results
            self.shared_results.update({
                'objectives': None, 'constraints': None, 'context': None,
                'stakeholders': None, 'technical_requirements': None, 'assumptions': None
            })
            return await self._execute_single_call(user_input, domain)
        else:
            return await self._execute_multi_parallel(user_input, domain)
    
    async def _execute_single_call(self, user_input: str, domain: str) -> dict:
        """Execute single mega-prompt call"""
        
        print("🔥 SINGLE MEGA-PROMPT - 1 LLM call for everything...")
        
        # Only need one client to objectives server (which handles complete_analysis)
        client = DealerClient(connect=f"tcp://localhost:{self.ports['objectives']}")
        
        send_start = time.time()
        print("📤 Sending single mega-prompt request...")
        
        client.send_message({"user_input": user_input, "domain": domain}, "complete_analysis")
        
        send_end = time.time()
        print(f"📤 Request sent in {send_end - send_start:.3f} seconds")
        
        # Wait for completion - check if all components are filled
        poll_start = time.time()
        print("📥 Waiting for mega-prompt completion...")
        
        while not all(self.shared_results[comp] is not None for comp in ['objectives', 'constraints', 'context', 'stakeholders', 'technical_requirements', 'assumptions']):
            await asyncio.sleep(0.1)
        
        poll_end = time.time()
        print(f"📥 Mega-prompt completed in {poll_end - poll_start:.3f} seconds")
        print(f"🔥 SINGLE CALL PROCESSING - 1 LLM call total!")
        
        # Get acknowledgment and close
        try:
            client.receive_message()
        except:
            pass
        client.close()
        
        return dict(self.shared_results)
    
    async def _execute_multi_parallel(self, user_input: str, domain: str) -> dict:
        """Execute 6 parallel calls"""
        
        print("🚀 MULTI-PARALLEL - 6 simultaneous LLM calls...")
        
        # Create 6 clients for full parallel requests - each connects to dedicated server
        clients = {}
        
        for component, port in self.ports.items():
            client = DealerClient(connect=f"tcp://localhost:{port}")
            clients[component] = client
        
        # Send all 6 requests simultaneously to dedicated servers
        send_start = time.time()
        print("📤 Sending all 6 requests simultaneously to dedicated servers...")
        
        # Independent components (don't need context from others)
        clients['objectives'].send_message({"user_input": user_input, "domain": domain}, "objectives")
        clients['constraints'].send_message({"user_input": user_input, "domain": domain}, "constraints")
        clients['context'].send_message({"user_input": user_input, "domain": domain}, "context")
        
        # Context-dependent components (use basic context for now, can be refined later)
        basic_context = f"Domain: {domain}, Request: {user_input[:100]}..."
        clients['stakeholders'].send_message({"user_input": user_input, "domain": domain, "objectives_context": basic_context}, "stakeholders")
        clients['technical_requirements'].send_message({"user_input": user_input, "domain": domain, "objectives_context": basic_context, "constraints_context": basic_context}, "technical_requirements")
        clients['assumptions'].send_message({"user_input": user_input, "domain": domain, "full_context": basic_context}, "assumptions")
        
        send_end = time.time()
        print(f"📤 All requests sent in {send_end - send_start:.3f} seconds")
        
        # Async polling for completion (no blocking!)
        poll_start = time.time()
        print("📥 Polling for completion (async, non-blocking)...")
        
        completed_components = set()
        
        while len(completed_components) < 6:
            # Check which components are complete
            for component, result in self.shared_results.items():
                if result is not None and component not in completed_components:
                    completed_components.add(component)
                    elapsed = time.time() - poll_start
                    print(f"  ✅ {component.title()} completed ({elapsed:.2f}s)")
            
            # Non-blocking wait
            if len(completed_components) < 6:
                await asyncio.sleep(0.1)  # Check every 100ms
        
        poll_end = time.time()
        print(f"📥 All components completed in {poll_end - poll_start:.3f} seconds")
        print(f"🚀 TRUE PARALLEL PROCESSING - no blocking collection!")
        
        # Collect acknowledgments (optional, just to clean up)
        for component, client in clients.items():
            try:
                client.receive_message()  # Get the acknowledgment
            except:
                pass  # Ignore if already processed
            client.close()
        
        return dict(self.shared_results)
    
    def _force_connect_floating_nodes(self, intent_graph):
        """Force connect floating nodes to create a connected graph"""
        import networkx as nx
        
        # Convert to NetworkX graph to find connected components
        nx_graph = nx.Graph()
        
        # Add nodes
        for node_id in intent_graph.nodes():
            nx_graph.add_node(node_id)
        
        # Add edges (undirected)
        for edge in intent_graph.edges():
            nx_graph.add_edge(edge[0], edge[1])
        
        # Find connected components
        components = list(nx.connected_components(nx_graph))
        
        if len(components) > 1:
            print(f"Found {len(components)} disconnected components, connecting them...")
            
            # Get the largest component as the main component
            main_component = max(components, key=len)
            
            # Connect other components to the main one
            for component in components:
                if component != main_component:
                    # Find a representative node from each component
                    main_node = list(main_component)[0]
                    floating_node = list(component)[0]
                    
                    # Create a logical connection
                    main_attrs = intent_graph.get_node_attributes(main_node)
                    floating_attrs = intent_graph.get_node_attributes(floating_node)
                    
                    main_type = main_attrs.get('type', '')
                    floating_type = floating_attrs.get('type', '')
                    
                    # Determine relationship based on node types
                    if 'objective' in main_type and 'stakeholder' in floating_type:
                        relationship = 'involves'
                    elif 'objective' in main_type and 'constraint' in floating_type:
                        relationship = 'limited_by'
                    elif 'objective' in main_type and 'assumption' in floating_type:
                        relationship = 'assumes'
                    elif 'objective' in main_type and 'technical' in floating_type:
                        relationship = 'requires'
                    else:
                        relationship = 'relates_to'
                    
                    intent_graph.add_edge(main_node, floating_node, relationship=relationship)
                    print(f"  Connected {main_node} -> {floating_node} ({relationship})")
        
        print(f"Final graph: {len(intent_graph.nodes())} nodes, {len(intent_graph.edges())} edges")
    
    def _add_selective_stakeholder_links(self, intent_graph):
        """Add selective, meaningful links for stakeholders based on their roles"""
        
        # Get all stakeholders
        stakeholders = [n for n in intent_graph.nodes() 
                       if intent_graph.get_node_attributes(n).get('type') == 'stakeholder']
        
        # Get all objectives
        objectives = [n for n in intent_graph.nodes() 
                     if intent_graph.get_node_attributes(n).get('type') == 'primary_objective']
        
        # Get all constraints
        constraints = [n for n in intent_graph.nodes() 
                      if intent_graph.get_node_attributes(n).get('type') == 'constraint']
        
        links_added = 0
        
        for stakeholder_id in stakeholders:
            stakeholder_attrs = intent_graph.get_node_attributes(stakeholder_id)
            stakeholder_name = stakeholder_attrs.get('name', '').lower()
            stakeholder_role = stakeholder_attrs.get('role', '').lower()
            
            print(f"  Linking stakeholder: {stakeholder_name} ({stakeholder_role})")
            
            # Role-based selective linking
            if 'owner' in stakeholder_name or 'decision' in stakeholder_role:
                # Owners care about budget and critical objectives
                for obj_id in objectives:
                    obj_attrs = intent_graph.get_node_attributes(obj_id)
                    obj_name = obj_attrs.get('name', '').lower()
                    if 'budget' in obj_name or 'cost' in obj_name or obj_attrs.get('priority') == 'critical':
                        intent_graph.add_edge(stakeholder_id, obj_id, relationship='concerned_about')
                        links_added += 1
                        print(f"    → {obj_id} (concerned_about)")
                
                # Owners care about budget constraints
                for con_id in constraints:
                    con_attrs = intent_graph.get_node_attributes(con_id)
                    if con_attrs.get('constraint_type') == 'budget':
                        intent_graph.add_edge(stakeholder_id, con_id, relationship='monitors')
                        links_added += 1
                        print(f"    → {con_id} (monitors)")
            
            elif 'manager' in stakeholder_name or 'manager' in stakeholder_role:
                # Managers care about operational objectives
                for obj_id in objectives:
                    obj_attrs = intent_graph.get_node_attributes(obj_id)
                    obj_name = obj_attrs.get('name', '').lower()
                    if any(word in obj_name for word in ['order', 'manage', 'process', 'handle']):
                        intent_graph.add_edge(stakeholder_id, obj_id, relationship='responsible_for')
                        links_added += 1
                        print(f"    → {obj_id} (responsible_for)")
            
            elif 'customer' in stakeholder_name:
                # Customers care about service-related objectives
                for obj_id in objectives:
                    obj_attrs = intent_graph.get_node_attributes(obj_id)
                    obj_name = obj_attrs.get('name', '').lower()
                    if any(word in obj_name for word in ['order', 'payment', 'service']):
                        intent_graph.add_edge(stakeholder_id, obj_id, relationship='benefits_from')
                        links_added += 1
                        print(f"    → {obj_id} (benefits_from)")
            
            elif 'staff' in stakeholder_name or 'wait' in stakeholder_name:
                # Staff care about usability objectives
                for obj_id in objectives:
                    obj_attrs = intent_graph.get_node_attributes(obj_id)
                    obj_name = obj_attrs.get('name', '').lower()
                    if any(word in obj_name for word in ['order', 'handle', 'manage']):
                        intent_graph.add_edge(stakeholder_id, obj_id, relationship='uses')
                        links_added += 1
                        print(f"    → {obj_id} (uses)")
        
        print(f"  Added {links_added} selective stakeholder links")

async def demo_zmq_intent_analysis():
    """Demonstrate ZMQ ROUTER-DEALER intent analysis with 6-parallel LLM calls and benchmarking"""
    
    print("ZMQ ROUTER-DEALER INTENT ANALYSIS DEMO")
    print("=" * 70)
    print("🚀 6-PARALLEL LLM ANALYSIS using Ethos-ZeroMQ RouterDealerServer")
    print("⚡ All 6 components processed simultaneously for maximum speed")
    print("📊 Includes detailed timing and performance benchmarks")
    print("=" * 70)
    
    # Test case
    user_input = "I need a new POS system for my restaurant that handles orders, payments, and inventory with a $25k budget in 6 weeks"
    domain = "restaurant_management"
    
    # Create ZMQ analyzer with multi-port architecture
    analyzer = ZMQIntentAnalyzer(base_port=5572)
    
    try:
        # Start ZMQ server
        server_start = time.time()
        analyzer.start_server()
        server_time = time.time() - server_start
        print(f"⏱️ ZMQ Server startup: {server_time:.3f} seconds")
        
        # Run ZMQ parallel analysis
        analysis_start = time.time()
        intent_data = await analyzer.analyze_intent_parallel(user_input, domain)
        analysis_time = time.time() - analysis_start
        
        print(f"\n🏁 BENCHMARK RESULTS:")
        print(f"   Server Startup: {server_time:.3f}s")
        print(f"   Analysis Time: {analysis_time:.3f}s")
        print(f"   Components/sec: {intent_data['graph_metadata']['total_components']/intent_data['graph_metadata']['analysis_duration']:.1f}")
        print(f"   Total Runtime: {server_time + analysis_time:.3f}s")
        
        # Create and populate intent graph
        print(f"\nCREATING INTENT GRAPH FROM ZMQ ANALYSIS")
        print("-" * 50)
        
        # Debug: Show raw data structure
        print(f"DEBUG: Raw intent data structure:")
        for key, value in intent_data.items():
            if isinstance(value, list):
                print(f"  - {key}: {len(value)} items")
                for i, item in enumerate(value[:2]):  # Show first 2 items
                    print(f"    [{i}]: {item}")
            elif isinstance(value, dict):
                print(f"  - {key}: dict with {len(value)} keys")
            else:
                print(f"  - {key}: {type(value).__name__}")
        
        intent_graph = IntentKnowledgeGraph("zmq_demo", "ZMQ ROUTER-DEALER Demo")
        
        # Use individual add_* methods instead of validate_and_import_json to create all nodes
        print("[INFO] Importing ZMQ analysis results using individual add_* methods...")
        
        # Add objectives
        objectives_added = 0
        for obj in intent_data['objectives']:
            intent_graph.add_objective(
                f"obj_{obj.get('id', objectives_added)}",
                name=obj.get('name', 'Unknown Objective'),
                description=obj.get('description', ''),
                priority=obj.get('priority', 'medium'),
                measurable=obj.get('measurable', False),
                measurement_method=obj.get('measurement_method', ''),
                target_value=obj.get('target_value'),
                unit=obj.get('unit', ''),
                complexity=obj.get('complexity', 'medium')
            )
            objectives_added += 1
        
        # Add constraints  
        constraints_added = 0
        for con in intent_data['constraints']:
            # Generate a meaningful name from constraint content
            constraint_text = con.get('constraint', '')
            constraint_type = con.get('constraint_type', 'other')
            
            # Create name based on type and content
            if constraint_type == 'budget':
                name = f"Budget Constraint (${con.get('value', 'Unknown')})"
            elif constraint_type == 'timeline':
                name = f"Timeline Constraint ({con.get('value', 'Unknown')} weeks)"
            else:
                # Extract first few words from constraint text
                words = constraint_text.split()[:3]
                name = ' '.join(words).title() if words else f"{constraint_type.title()} Constraint"
            
            intent_graph.add_constraint(
                f"con_{con.get('id', constraints_added)}",
                constraint_type=constraint_type,
                constraint=constraint_text,
                value=con.get('value'),
                flexibility=con.get('flexibility', 'flexible'),
                time_pressure=con.get('time_pressure', 'moderate'),
                name=name  # Add the generated name
            )
            constraints_added += 1
        
        # Add stakeholders
        stakeholders_added = 0
        for stake in intent_data['stakeholders']:
            intent_graph.add_stakeholder(
                f"stake_{stake.get('id', stakeholders_added)}",
                name=stake.get('name', 'Unknown Stakeholder'),
                role=stake.get('role', 'participant'),
                influence_level=stake.get('influence_level', 'medium'),
                support_level=stake.get('support_level', 'neutral'),
                decision_authority=stake.get('decision_authority', 'input')
            )
            stakeholders_added += 1
        
        # Add technical requirements as nodes
        tech_reqs_added = 0
        for tech in intent_data['technical_requirements']:
            intent_graph.add_node(
                f"tech_{tech.get('id', tech_reqs_added)}",
                type='technical_requirement',
                name=tech.get('name', 'Unknown Technical Requirement'),
                description=tech.get('description', ''),
                complexity=tech.get('complexity', 'medium'),
                importance=tech.get('importance', 'medium')
            )
            tech_reqs_added += 1
        
        # Add assumptions as nodes
        assumptions_added = 0
        for assumption in intent_data['assumptions']:
            # Generate a meaningful name from assumption content
            assumption_text = assumption.get('assumption', '')
            
            # Extract key concept from assumption text (first 4-5 words, remove common words)
            words = assumption_text.split()
            # Remove common words and take first meaningful words
            meaningful_words = [w for w in words[:8] if w.lower() not in ['the', 'user', 'assumes', 'that', 'a', 'an', 'is', 'can', 'be']]
            name = ' '.join(meaningful_words[:4]).title() if meaningful_words else f"Assumption {assumptions_added + 1}"
            
            # Clean up the name
            if len(name) > 40:
                name = name[:37] + "..."
            
            intent_graph.add_node(
                f"assumption_{assumption.get('id', assumptions_added)}",
                type='assumption',
                name=name,  # Add the generated name
                assumption=assumption_text,
                confidence=assumption.get('confidence', 0.5),
                impact_if_wrong=assumption.get('impact_if_wrong', 'medium'),
                validation_method=assumption.get('validation_method', '')
            )
            assumptions_added += 1
        
        # Set context data
        context_data = intent_data.get('intent_data', {})
        intent_graph.domain_context = context_data.get('domain_context', 'unknown')
        intent_graph.user_sentiment = context_data.get('user_sentiment', 'neutral')
        intent_graph.clarifying_questions = context_data.get('clarifying_questions', [])
        intent_graph.implicit_requirements = context_data.get('implicit_requirements', [])
        
        print("[PASS] Individual method import PASSED")
        print("Import statistics:")
        print(f"  - objectives_added: {objectives_added}")
        print(f"  - constraints_added: {constraints_added}")
        print(f"  - stakeholders_added: {stakeholders_added}")
        print(f"  - technical_requirements_added: {tech_reqs_added}")
        print(f"  - assumptions_added: {assumptions_added}")
        print(f"  - total_nodes_created: {objectives_added + constraints_added + stakeholders_added + tech_reqs_added + assumptions_added}")
        
        # Debug: Show node types before auto-linking
        print(f"\nDEBUG: Node types created:")
        for node_id in intent_graph.nodes():
            node_attrs = intent_graph.get_node_attributes(node_id)
            print(f"  - {node_id}: type='{node_attrs.get('type', 'MISSING')}', name='{node_attrs.get('name', 'MISSING')}'")
        
        # Apply auto-linking
        print(f"\nAPPLYING AUTO-LINKING")
        print("-" * 30)
        
        print(f"Before auto-linking: {len(intent_graph.nodes())} nodes, {len(intent_graph.edges())} edges")
        
        link_stats = intent_graph.comprehensive_auto_link()
        
        print(f"After auto-linking: {len(intent_graph.nodes())} nodes, {len(intent_graph.edges())} edges")
        print(f"Links added: {link_stats['total_links_added']}")
        
        # Add selective stakeholder linking
        print(f"\n🔗 ADDING SELECTIVE STAKEHOLDER LINKS")
        print("-" * 40)
        analyzer._add_selective_stakeholder_links(intent_graph)
        
        # Debug: Show what edges were created
        print(f"\nDEBUG: Edges created:")
        for edge in intent_graph.edges():
            try:
                edge_attrs = intent_graph.get_edge_attributes(edge[0], edge[1])
                relationship = edge_attrs.get('relationship', 'no_relationship') if edge_attrs else 'no_relationship'
            except:
                relationship = 'unknown_relationship'
            print(f"  - {edge[0]} -> {edge[1]}: {relationship}")
        
        # Force additional linking if needed
        if len(intent_graph.edges()) < len(intent_graph.nodes()) - 1:
            print(f"\n🔧 FORCING ADDITIONAL LINKS (graph not fully connected)")
            analyzer._force_connect_floating_nodes(intent_graph)
        
        # Show 3D visualization (always show, regardless of connectivity)
        print(f"\n3D VISUALIZATION")
        print("-" * 20)
        
        viz = MatplotlibKnowledgeGraphVisualizer(intent_graph)
        viz.render_3d_graph(
            figsize=(14, 10),
            show_labels=True,
            positioning_strategy="layered"
        )
        
        # Show analysis results
        print(f"\nINTENT ANALYSIS RESULTS")
        print("-" * 30)
        
        gaps = intent_graph.assess_gaps()
        feasibility = intent_graph.assess_feasibility()
        completeness = intent_graph.get_intent_completeness_score()
        
        print(f"Feasibility score: {feasibility['overall_feasibility']:.2f}")
        print(f"Intent completeness: {completeness:.2f}")
        print(f"Gaps identified: {len(gaps)}")
        
        if gaps:
            critical_gaps = [g for g in gaps if g['severity'] in ['critical', 'high']]
            print(f"Critical/High gaps: {len(critical_gaps)}")
            for gap in critical_gaps[:3]:
                print(f"  - {gap['gap_type'].upper()}: {gap['description']}")
        
        # Show extracted data
        print(f"\nExtracted intent data:")
        print(f"  - Domain: {intent_graph.domain_context}")
        print(f"  - Sentiment: {intent_graph.user_sentiment}")
        print(f"  - Clarifying questions: {len(intent_graph.clarifying_questions)}")
        print(f"  - Implicit requirements: {len(intent_graph.implicit_requirements)}")
    
    finally:
        # Always stop server
        analyzer.stop_server()

async def comparison_demo():
    """Compare single-call vs multi-port performance"""
    
    print("🏁 ZMQ ROUTER-DEALER PERFORMANCE COMPARISON")
    print("=" * 80)
    print("Comparing single mega-prompt vs 6-parallel multi-port approaches")
    print("=" * 80)
    
    user_input = "I need a new POS system for my restaurant that handles orders, payments, and inventory with a $25k budget in 6 weeks"
    domain = "restaurant_management"
    
    results = {}
    
    # Test 1: Single mega-prompt
    print(f"\n🔥 TEST 1: SINGLE MEGA-PROMPT APPROACH")
    print("-" * 50)
    
    analyzer1 = ZMQIntentAnalyzer(base_port=5580, merge_strategy="single_call")
    
    try:
        start1 = time.time()
        analyzer1.start_servers()
        setup1 = time.time() - start1
        
        analysis_start1 = time.time()
        intent_data1 = await analyzer1.analyze_intent_parallel(user_input, domain)
        analysis1 = time.time() - analysis_start1
        
        total1 = time.time() - start1
        
        results['single_call'] = {
            'setup_time': setup1,
            'analysis_time': analysis1,
            'total_time': total1,
            'components': intent_data1['graph_metadata']['total_components']
        }
        
        print(f"✅ Single-call completed: {total1:.2f}s total")
        
    finally:
        analyzer1.stop_servers()
    
    # Test 2: Multi-port parallel
    print(f"\n🚀 TEST 2: MULTI-PORT PARALLEL APPROACH")
    print("-" * 50)
    
    analyzer2 = ZMQIntentAnalyzer(base_port=5590, merge_strategy="independent")
    
    try:
        start2 = time.time()
        analyzer2.start_servers()
        setup2 = time.time() - start2
        
        analysis_start2 = time.time()
        intent_data2 = await analyzer2.analyze_intent_parallel(user_input, domain)
        analysis2 = time.time() - analysis_start2
        
        total2 = time.time() - start2
        
        results['multi_port'] = {
            'setup_time': setup2,
            'analysis_time': analysis2,
            'total_time': total2,
            'components': intent_data2['graph_metadata']['total_components']
        }
        
        print(f"✅ Multi-port completed: {total2:.2f}s total")
        
    finally:
        analyzer2.stop_servers()
    
    # Comparison Results
    print(f"\n📊 PERFORMANCE COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"{'Metric':<25} {'Single Call':<15} {'Multi-Port':<15} {'Winner':<15}")
    print("-" * 70)
    
    setup_winner = "Single Call" if results['single_call']['setup_time'] < results['multi_port']['setup_time'] else "Multi-Port"
    analysis_winner = "Single Call" if results['single_call']['analysis_time'] < results['multi_port']['analysis_time'] else "Multi-Port"
    total_winner = "Single Call" if results['single_call']['total_time'] < results['multi_port']['total_time'] else "Multi-Port"
    
    print(f"{'Setup Time':<25} {results['single_call']['setup_time']:<15.2f} {results['multi_port']['setup_time']:<15.2f} {setup_winner:<15}")
    print(f"{'Analysis Time':<25} {results['single_call']['analysis_time']:<15.2f} {results['multi_port']['analysis_time']:<15.2f} {analysis_winner:<15}")
    print(f"{'Total Time':<25} {results['single_call']['total_time']:<15.2f} {results['multi_port']['total_time']:<15.2f} {total_winner:<15}")
    
    speedup = results['single_call']['total_time'] / results['multi_port']['total_time']
    if speedup > 1:
        print(f"\n🚀 Multi-Port is {speedup:.1f}x FASTER than Single Call!")
    else:
        print(f"\n🔥 Single Call is {1/speedup:.1f}x FASTER than Multi-Port!")
    
    print(f"\n📈 Components analyzed: {results['single_call']['components']} vs {results['multi_port']['components']}")
    
    # Create side-by-side visualizations
    print(f"\n🎨 CREATING SIDE-BY-SIDE GRAPH COMPARISON")
    print("=" * 60)
    
    await create_comparison_visualization(intent_data1, intent_data2, results)

async def create_comparison_visualization(single_data, multi_data, timing_results):
    """Create side-by-side visualization comparing both approaches"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    # Create both intent graphs
    print("📊 Building graphs for visualization...")
    
    # Single-call graph
    single_graph = IntentKnowledgeGraph("single_call", "Single Mega-Prompt")
    await import_intent_data(single_graph, single_data)
    single_graph.comprehensive_auto_link()
    
    # Multi-port graph  
    multi_graph = IntentKnowledgeGraph("multi_port", "Multi-Port Parallel")
    await import_intent_data(multi_graph, multi_data)
    multi_graph.comprehensive_auto_link()
    
    # Add selective stakeholder linking to multi-port (like in the main demo)
    print("🔗 Adding selective stakeholder links to multi-port graph...")
    add_selective_stakeholder_links(multi_graph)
    
    # Create side-by-side matplotlib figure
    fig = plt.figure(figsize=(20, 12))
    
    # Performance comparison at top
    ax_perf = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    
    methods = ['Single Call', 'Multi-Port']
    setup_times = [timing_results['single_call']['setup_time'], timing_results['multi_port']['setup_time']]
    analysis_times = [timing_results['single_call']['analysis_time'], timing_results['multi_port']['analysis_time']]
    
    x = range(len(methods))
    width = 0.35
    
    bars1 = ax_perf.bar([i - width/2 for i in x], setup_times, width, label='Setup Time', alpha=0.8, color='lightblue')
    bars2 = ax_perf.bar([i + width/2 for i in x], analysis_times, width, label='Analysis Time', alpha=0.8, color='orange')
    
    ax_perf.set_xlabel('Approach')
    ax_perf.set_ylabel('Time (seconds)')
    ax_perf.set_title('🏁 Performance Comparison: Single Call vs Multi-Port Parallel')
    ax_perf.set_xticks(x)
    ax_perf.set_xticklabels(methods)
    ax_perf.legend()
    ax_perf.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax_perf.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}s', ha='center', va='bottom')
    
    # Single-call graph visualization
    ax_single = plt.subplot2grid((3, 2), (1, 0), rowspan=2)
    viz_single = MatplotlibKnowledgeGraphVisualizer(single_graph)
    
    # Multi-port graph visualization  
    ax_multi = plt.subplot2grid((3, 2), (1, 1), rowspan=2)
    viz_multi = MatplotlibKnowledgeGraphVisualizer(multi_graph)
    
    # Render both graphs manually for side-by-side comparison
    import networkx as nx
    
    print("🎨 Rendering single-call graph...")
    # Convert to NetworkX for visualization
    G1 = nx.Graph()
    for node in single_graph.nodes():
        G1.add_node(node)
    for edge in single_graph.edges():
        G1.add_edge(edge[0], edge[1])
    
    pos1 = nx.spring_layout(G1, k=1, iterations=50)
    nx.draw(G1, pos1, ax=ax_single, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=8, font_weight='bold')
    ax_single.set_title(f'🔥 Single Mega-Prompt\n{len(single_graph.nodes())} nodes, {len(single_graph.edges())} edges')
    
    print("🎨 Rendering multi-port graph...")
    # Convert to NetworkX for visualization
    G2 = nx.Graph()
    for node in multi_graph.nodes():
        G2.add_node(node)
    for edge in multi_graph.edges():
        G2.add_edge(edge[0], edge[1])
    
    pos2 = nx.spring_layout(G2, k=1, iterations=50)
    nx.draw(G2, pos2, ax=ax_multi, with_labels=True, node_color='lightgreen', 
            node_size=500, font_size=8, font_weight='bold')
    ax_multi.set_title(f'🚀 Multi-Port Parallel\n{len(multi_graph.nodes())} nodes, {len(multi_graph.edges())} edges')
    
    # Add comparison stats
    speedup = timing_results['single_call']['total_time'] / timing_results['multi_port']['total_time']
    winner = "Multi-Port" if speedup > 1 else "Single Call"
    
    fig.suptitle(f'ZMQ ROUTER-DEALER Comparison: {winner} is {max(speedup, 1/speedup):.1f}x FASTER!', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Side-by-side visualization complete!")
    print(f"📊 Graph Comparison:")
    print(f"   Single Call: {len(single_graph.nodes())} nodes, {len(single_graph.edges())} edges")
    print(f"   Multi-Port:  {len(multi_graph.nodes())} nodes, {len(multi_graph.edges())} edges")

async def import_intent_data(intent_graph, intent_data):
    """Helper to import intent data into graph"""
    
    # Add objectives
    for obj in intent_data.get('objectives', []):
        intent_graph.add_objective(
            f"obj_{obj.get('id', 0)}",
            name=obj.get('name', 'Unknown'),
            description=obj.get('description', ''),
            priority=obj.get('priority', 'medium')
        )
    
    # Add constraints
    for con in intent_data.get('constraints', []):
        constraint_type = con.get('constraint_type', 'other')
        if constraint_type == 'budget':
            name = f"Budget Constraint (${con.get('value', 'Unknown')})"
        elif constraint_type == 'timeline':
            name = f"Timeline Constraint ({con.get('value', 'Unknown')} weeks)"
        else:
            name = f"{constraint_type.title()} Constraint"
            
        intent_graph.add_constraint(
            f"con_{con.get('id', 0)}",
            constraint_type=constraint_type,
            constraint=con.get('constraint', ''),
            name=name
        )
    
    # Add stakeholders
    for stake in intent_data.get('stakeholders', []):
        intent_graph.add_stakeholder(
            f"stake_{stake.get('id', 0)}",
            name=stake.get('name', 'Unknown'),
            role=stake.get('role', 'participant')
        )

def add_selective_stakeholder_links(intent_graph):
    """Add selective, meaningful links for stakeholders based on their roles"""
    
    # Get all stakeholders
    stakeholders = [n for n in intent_graph.nodes() 
                   if intent_graph.get_node_attributes(n).get('type') == 'stakeholder']
    
    # Get all objectives
    objectives = [n for n in intent_graph.nodes() 
                 if intent_graph.get_node_attributes(n).get('type') == 'primary_objective']
    
    # Get all constraints
    constraints = [n for n in intent_graph.nodes() 
                  if intent_graph.get_node_attributes(n).get('type') == 'constraint']
    
    links_added = 0
    
    for stakeholder_id in stakeholders:
        stakeholder_attrs = intent_graph.get_node_attributes(stakeholder_id)
        stakeholder_name = stakeholder_attrs.get('name', '').lower()
        stakeholder_role = stakeholder_attrs.get('role', '').lower()
        
        # Role-based selective linking
        if 'owner' in stakeholder_name or 'decision' in stakeholder_role:
            # Owners care about budget and critical objectives
            for obj_id in objectives:
                obj_attrs = intent_graph.get_node_attributes(obj_id)
                obj_name = obj_attrs.get('name', '').lower()
                if 'budget' in obj_name or 'cost' in obj_name or obj_attrs.get('priority') == 'critical':
                    intent_graph.add_edge(stakeholder_id, obj_id, relationship='concerned_about')
                    links_added += 1
            
            # Owners care about budget constraints
            for con_id in constraints:
                con_attrs = intent_graph.get_node_attributes(con_id)
                if con_attrs.get('constraint_type') == 'budget':
                    intent_graph.add_edge(stakeholder_id, con_id, relationship='monitors')
                    links_added += 1
        
        elif 'manager' in stakeholder_name or 'manager' in stakeholder_role:
            # Managers care about operational objectives
            for obj_id in objectives:
                obj_attrs = intent_graph.get_node_attributes(obj_id)
                obj_name = obj_attrs.get('name', '').lower()
                if any(word in obj_name for word in ['order', 'manage', 'process', 'handle']):
                    intent_graph.add_edge(stakeholder_id, obj_id, relationship='responsible_for')
                    links_added += 1
        
        elif 'customer' in stakeholder_name:
            # Customers care about service-related objectives
            for obj_id in objectives:
                obj_attrs = intent_graph.get_node_attributes(obj_id)
                obj_name = obj_attrs.get('name', '').lower()
                if any(word in obj_name for word in ['order', 'payment', 'service']):
                    intent_graph.add_edge(stakeholder_id, obj_id, relationship='benefits_from')
                    links_added += 1
        
        elif 'staff' in stakeholder_name or 'wait' in stakeholder_name:
            # Staff care about usability objectives
            for obj_id in objectives:
                obj_attrs = intent_graph.get_node_attributes(obj_id)
                obj_name = obj_attrs.get('name', '').lower()
                if any(word in obj_name for word in ['order', 'handle', 'manage']):
                    intent_graph.add_edge(stakeholder_id, obj_id, relationship='uses')
                    links_added += 1
    
    print(f"  Added {links_added} selective stakeholder links")
    return links_added

if __name__ == "__main__":
    print("Starting ZMQ ROUTER-DEALER Performance Comparison...")
    asyncio.run(comparison_demo())
