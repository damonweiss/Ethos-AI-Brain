#!/usr/bin/env python3
"""
Intent LLM Runner - Core ZMQ ROUTER-DEALER Intent Analysis Engine
Transforms LLM prompts into completed intent knowledge graphs using parallel processing
"""

import asyncio
import json
import uuid
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from openai import OpenAI

# Add the Ethos-ZeroMQ path
import sys
sys.path.append(r'C:\Users\DamonWeiss\PycharmProjects\Ethos-ZeroMQ')

from ethos_zeromq.EthosZeroMQ_Engine import ZeroMQEngine
from ethos_zeromq.RouterDealerServer import RouterDealerServer, DealerClient
from intent_knowledge_graph_zmq import IntentKnowledgeGraph

load_dotenv()

class IntentLLMRunner:
    """
    Core ZMQ ROUTER-DEALER engine for parallel intent analysis
    Transforms user prompts into rich intent knowledge graphs
    """
    
    def __init__(self, base_port: int = 5600, merge_strategy: str = "multi_port"):
        """
        Initialize Intent LLM Runner
        
        Args:
            base_port: Starting port for ZMQ servers (uses base_port to base_port+5)
            merge_strategy: "single_call", "merged_pairs", "two_wave", or "multi_port"
        """
        self.base_port = base_port
        self.merge_strategy = merge_strategy
        self.zmq_engine = ZeroMQEngine()
        self.servers = {}
        
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
            self.shared_results = {'complete_analysis': None}
        else:
            self.shared_results = {
                'objectives': None,
                'constraints': None,
                'context': None,
                'stakeholders': None,
                'technical_requirements': None,
                'assumptions': None
            }
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.llm_client = OpenAI(api_key=api_key)
    
    async def analyze_intent(self, user_input: str, domain: str = "unknown") -> IntentKnowledgeGraph:
        """
        Main entry point: Transform user input into completed intent graph
        
        Args:
            user_input: Natural language description of intent
            domain: Domain context (e.g., "restaurant_management", "software_development")
            
        Returns:
            IntentKnowledgeGraph: Fully populated and linked intent graph
        """
        
        print(f"🚀 INTENT LLM RUNNER - {self.merge_strategy.upper()} MODE")
        print("=" * 60)
        print(f"Input: {user_input}")
        print(f"Domain: {domain}")
        
        start_time = time.time()
        
        try:
            # Start ZMQ servers
            self._start_servers()
            
            # Execute parallel analysis
            analysis_start = time.time()
            intent_data = await self._execute_analysis(user_input, domain)
            analysis_time = time.time() - analysis_start
            
            # Build intent graph
            intent_graph = self._build_intent_graph(intent_data)
            
            # Apply intelligent linking
            self._apply_intelligent_linking(intent_graph)
            
            total_time = time.time() - start_time
            
            print(f"\n✅ INTENT ANALYSIS COMPLETE")
            print(f"   Analysis Time: {analysis_time:.2f}s")
            print(f"   Total Time: {total_time:.2f}s")
            print(f"   Graph: {len(intent_graph.nodes())} nodes, {len(intent_graph.edges())} edges")
            
            return intent_graph
            
        finally:
            self._stop_servers()
    
    def _start_servers(self):
        """Start ZMQ servers based on merge strategy"""
        
        if self.merge_strategy == "single_call":
            print(f"🔥 Starting 1 ZMQ server for single mega-prompt...")
            
            port = self.ports['objectives']
            server = RouterDealerServer(
                engine=self.zmq_engine,
                bind=f"tcp://*:{port}",
                connect=f"tcp://localhost:{port}"
            )
            
            server.register_handler("complete_analysis", self._handle_complete_analysis)
            server.start()
            self.servers['objectives'] = server
            
        else:
            print(f"🚀 Starting {len(self.ports)} ZMQ servers for parallel processing...")
            
            for component, port in self.ports.items():
                server = RouterDealerServer(
                    engine=self.zmq_engine,
                    bind=f"tcp://*:{port}",
                    connect=f"tcp://localhost:{port}"
                )
                
                handler_method = getattr(self, f'_handle_{component}')
                server.register_handler(component, handler_method)
                server.start()
                self.servers[component] = server
                
                time.sleep(0.05)  # Brief delay between server starts
    
    def _stop_servers(self):
        """Stop all ZMQ servers"""
        if self.servers:
            for component, server in self.servers.items():
                server.stop()
            self.servers.clear()
    
    async def _execute_analysis(self, user_input: str, domain: str) -> Dict[str, Any]:
        """Execute LLM analysis based on merge strategy"""
        
        # Reset shared results
        for key in self.shared_results:
            self.shared_results[key] = None
        
        if self.merge_strategy == "single_call":
            return await self._execute_single_call(user_input, domain)
        else:
            return await self._execute_multi_parallel(user_input, domain)
    
    async def _execute_single_call(self, user_input: str, domain: str) -> Dict[str, Any]:
        """Execute single mega-prompt call"""
        
        # Temporarily expand shared_results for single call
        self.shared_results.update({
            'objectives': None, 'constraints': None, 'context': None,
            'stakeholders': None, 'technical_requirements': None, 'assumptions': None
        })
        
        client = DealerClient(connect=f"tcp://localhost:{self.ports['objectives']}")
        
        print("📤 Sending mega-prompt request...")
        client.send_message({"user_input": user_input, "domain": domain}, "complete_analysis")
        
        # Wait for completion
        while not all(self.shared_results[comp] is not None for comp in ['objectives', 'constraints', 'context', 'stakeholders', 'technical_requirements', 'assumptions']):
            await asyncio.sleep(0.1)
        
        try:
            client.receive_message()
        except:
            pass
        client.close()
        
        return dict(self.shared_results)
    
    async def _execute_multi_parallel(self, user_input: str, domain: str) -> Dict[str, Any]:
        """Execute 6 parallel calls"""
        
        # Create clients for each component
        clients = {}
        for component, port in self.ports.items():
            clients[component] = DealerClient(connect=f"tcp://localhost:{port}")
        
        # Send all requests simultaneously
        print("📤 Sending 6 parallel requests...")
        clients['objectives'].send_message({"user_input": user_input, "domain": domain}, "objectives")
        clients['constraints'].send_message({"user_input": user_input, "domain": domain}, "constraints")
        clients['context'].send_message({"user_input": user_input, "domain": domain}, "context")
        
        basic_context = f"Domain: {domain}, Request: {user_input[:100]}..."
        clients['stakeholders'].send_message({"user_input": user_input, "domain": domain, "objectives_context": basic_context}, "stakeholders")
        clients['technical_requirements'].send_message({"user_input": user_input, "domain": domain, "objectives_context": basic_context, "constraints_context": basic_context}, "technical_requirements")
        clients['assumptions'].send_message({"user_input": user_input, "domain": domain, "full_context": basic_context}, "assumptions")
        
        # Wait for all completions
        completed_components = set()
        while len(completed_components) < 6:
            for component, result in self.shared_results.items():
                if result is not None and component not in completed_components:
                    completed_components.add(component)
                    print(f"  ✅ {component.title()} completed")
            
            if len(completed_components) < 6:
                await asyncio.sleep(0.1)
        
        # Clean up clients
        for component, client in clients.items():
            try:
                client.receive_message()
            except:
                pass
            client.close()
        
        return dict(self.shared_results)
    
    def _build_intent_graph(self, intent_data: Dict[str, Any]) -> IntentKnowledgeGraph:
        """Build intent graph from LLM analysis results"""
        
        print("🏗️ Building intent knowledge graph...")
        
        intent_graph = IntentKnowledgeGraph("llm_runner", "LLM Runner Generated Graph")
        
        # Add objectives
        for obj in intent_data.get('objectives', []):
            intent_graph.add_objective(
                f"obj_{obj.get('id', 0)}",
                name=obj.get('name', 'Unknown Objective'),
                description=obj.get('description', ''),
                priority=obj.get('priority', 'medium'),
                measurable=obj.get('measurable', False),
                measurement_method=obj.get('measurement_method', ''),
                target_value=obj.get('target_value'),
                unit=obj.get('unit', ''),
                complexity=obj.get('complexity', 'medium')
            )
        
        # Add constraints
        for con in intent_data.get('constraints', []):
            constraint_type = con.get('constraint_type', 'other')
            if constraint_type == 'budget':
                name = f"Budget Constraint (${con.get('value', 'Unknown')})"
            elif constraint_type == 'timeline':
                name = f"Timeline Constraint ({con.get('value', 'Unknown')} weeks)"
            else:
                words = con.get('constraint', '').split()[:3]
                name = ' '.join(words).title() if words else f"{constraint_type.title()} Constraint"
            
            intent_graph.add_constraint(
                f"con_{con.get('id', 0)}",
                constraint_type=constraint_type,
                constraint=con.get('constraint', ''),
                value=con.get('value'),
                flexibility=con.get('flexibility', 'flexible'),
                time_pressure=con.get('time_pressure', 'moderate'),
                name=name
            )
        
        # Add stakeholders
        for stake in intent_data.get('stakeholders', []):
            intent_graph.add_stakeholder(
                f"stake_{stake.get('id', 0)}",
                name=stake.get('name', 'Unknown Stakeholder'),
                role=stake.get('role', 'participant'),
                influence_level=stake.get('influence_level', 'medium'),
                support_level=stake.get('support_level', 'neutral'),
                decision_authority=stake.get('decision_authority', 'input')
            )
        
        # Add technical requirements
        for tech in intent_data.get('technical_requirements', []):
            intent_graph.add_node(
                f"tech_{tech.get('id', 0)}",
                type='technical_requirement',
                name=tech.get('name', 'Unknown Technical Requirement'),
                description=tech.get('description', ''),
                complexity=tech.get('complexity', 'medium'),
                importance=tech.get('importance', 'medium')
            )
        
        # Add assumptions
        for assumption in intent_data.get('assumptions', []):
            assumption_text = assumption.get('assumption', '')
            words = assumption_text.split()
            meaningful_words = [w for w in words[:8] if w.lower() not in ['the', 'user', 'assumes', 'that', 'a', 'an', 'is', 'can', 'be']]
            name = ' '.join(meaningful_words[:4]).title() if meaningful_words else f"Assumption {assumption.get('id', 0)}"
            
            if len(name) > 40:
                name = name[:37] + "..."
            
            intent_graph.add_node(
                f"assumption_{assumption.get('id', 0)}",
                type='assumption',
                name=name,
                assumption=assumption_text,
                confidence=assumption.get('confidence', 0.5),
                impact_if_wrong=assumption.get('impact_if_wrong', 'medium'),
                validation_method=assumption.get('validation_method', '')
            )
        
        # Set context data
        context_data = intent_data.get('context', {})
        intent_graph.domain_context = context_data.get('domain_context', 'unknown')
        intent_graph.user_sentiment = context_data.get('user_sentiment', 'neutral')
        intent_graph.clarifying_questions = context_data.get('clarifying_questions', [])
        intent_graph.implicit_requirements = context_data.get('implicit_requirements', [])
        
        print(f"   Created {len(intent_graph.nodes())} nodes")
        return intent_graph
    
    def _apply_intelligent_linking(self, intent_graph: IntentKnowledgeGraph):
        """Apply intelligent auto-linking and selective stakeholder linking"""
        
        print("🔗 Applying intelligent linking...")
        
        # Apply comprehensive auto-linking
        link_stats = intent_graph.comprehensive_auto_link()
        print(f"   Auto-linked: {link_stats.get('total_links_added', 0)} edges")
        
        # Apply selective stakeholder linking
        stakeholder_links = self._add_selective_stakeholder_links(intent_graph)
        print(f"   Stakeholder links: {stakeholder_links} edges")
        
        print(f"   Final graph: {len(intent_graph.nodes())} nodes, {len(intent_graph.edges())} edges")
    
    def _add_selective_stakeholder_links(self, intent_graph: IntentKnowledgeGraph) -> int:
        """Add selective stakeholder links based on roles"""
        
        stakeholders = [n for n in intent_graph.nodes() 
                       if intent_graph.get_node_attributes(n).get('type') == 'stakeholder']
        objectives = [n for n in intent_graph.nodes() 
                     if intent_graph.get_node_attributes(n).get('type') == 'primary_objective']
        constraints = [n for n in intent_graph.nodes() 
                      if intent_graph.get_node_attributes(n).get('type') == 'constraint']
        
        links_added = 0
        
        for stakeholder_id in stakeholders:
            stakeholder_attrs = intent_graph.get_node_attributes(stakeholder_id)
            stakeholder_name = stakeholder_attrs.get('name', '').lower()
            stakeholder_role = stakeholder_attrs.get('role', '').lower()
            
            # Role-based selective linking
            if 'owner' in stakeholder_name or 'decision' in stakeholder_role:
                for obj_id in objectives:
                    obj_attrs = intent_graph.get_node_attributes(obj_id)
                    obj_name = obj_attrs.get('name', '').lower()
                    if 'budget' in obj_name or 'cost' in obj_name or obj_attrs.get('priority') == 'critical':
                        intent_graph.add_edge(stakeholder_id, obj_id, relationship='concerned_about')
                        links_added += 1
                
                for con_id in constraints:
                    con_attrs = intent_graph.get_node_attributes(con_id)
                    if con_attrs.get('constraint_type') == 'budget':
                        intent_graph.add_edge(stakeholder_id, con_id, relationship='monitors')
                        links_added += 1
            
            elif 'manager' in stakeholder_name or 'manager' in stakeholder_role:
                for obj_id in objectives:
                    obj_attrs = intent_graph.get_node_attributes(obj_id)
                    obj_name = obj_attrs.get('name', '').lower()
                    if any(word in obj_name for word in ['order', 'manage', 'process', 'handle']):
                        intent_graph.add_edge(stakeholder_id, obj_id, relationship='responsible_for')
                        links_added += 1
            
            elif 'customer' in stakeholder_name:
                for obj_id in objectives:
                    obj_attrs = intent_graph.get_node_attributes(obj_id)
                    obj_name = obj_attrs.get('name', '').lower()
                    if any(word in obj_name for word in ['order', 'payment', 'service']):
                        intent_graph.add_edge(stakeholder_id, obj_id, relationship='benefits_from')
                        links_added += 1
            
            elif 'staff' in stakeholder_name or 'wait' in stakeholder_name:
                for obj_id in objectives:
                    obj_attrs = intent_graph.get_node_attributes(obj_id)
                    obj_name = obj_attrs.get('name', '').lower()
                    if any(word in obj_name for word in ['order', 'handle', 'manage']):
                        intent_graph.add_edge(stakeholder_id, obj_id, relationship='uses')
                        links_added += 1
        
        return links_added
    
    # ========================================
    # LLM Handler Methods
    # ========================================
    
    def _handle_objectives(self, message):
        """Handle objectives analysis request"""
        user_input = message.get('user_input', '')
        domain = message.get('domain', 'unknown')
        
        prompt = IntentKnowledgeGraph.get_objectives_prompt(user_input, domain)
        result = self._call_llm_sync(prompt, 'objectives')
        
        self.shared_results['objectives'] = result
        return {"status": "completed", "component": "objectives"}
    
    def _handle_constraints(self, message):
        """Handle constraints analysis request"""
        user_input = message.get('user_input', '')
        domain = message.get('domain', 'unknown')
        
        prompt = IntentKnowledgeGraph.get_constraints_prompt(user_input, domain)
        result = self._call_llm_sync(prompt, 'constraints')
        
        self.shared_results['constraints'] = result
        return {"status": "completed", "component": "constraints"}
    
    def _handle_context(self, message):
        """Handle context analysis request"""
        user_input = message.get('user_input', '')
        domain = message.get('domain', 'unknown')
        
        prompt = IntentKnowledgeGraph.get_context_prompt(user_input, domain)
        result = self._call_llm_sync(prompt, 'context')
        
        self.shared_results['context'] = result
        return {"status": "completed", "component": "context"}
    
    def _handle_stakeholders(self, message):
        """Handle stakeholders analysis request"""
        user_input = message.get('user_input', '')
        domain = message.get('domain', 'unknown')
        objectives_context = message.get('objectives_context', '')
        
        prompt = IntentKnowledgeGraph.get_stakeholders_prompt(user_input, domain, objectives_context)
        result = self._call_llm_sync(prompt, 'stakeholders')
        
        self.shared_results['stakeholders'] = result
        return {"status": "completed", "component": "stakeholders"}
    
    def _handle_technical_requirements(self, message):
        """Handle technical requirements analysis request"""
        user_input = message.get('user_input', '')
        domain = message.get('domain', 'unknown')
        objectives_context = message.get('objectives_context', '')
        constraints_context = message.get('constraints_context', '')
        
        prompt = IntentKnowledgeGraph.get_technical_requirements_prompt(user_input, domain, objectives_context, constraints_context)
        result = self._call_llm_sync(prompt, 'technical_requirements')
        
        self.shared_results['technical_requirements'] = result
        return {"status": "completed", "component": "technical_requirements"}
    
    def _handle_assumptions(self, message):
        """Handle assumptions analysis request"""
        user_input = message.get('user_input', '')
        domain = message.get('domain', 'unknown')
        full_context = message.get('full_context', '')
        
        prompt = IntentKnowledgeGraph.get_assumptions_prompt(user_input, domain, full_context)
        result = self._call_llm_sync(prompt, 'assumptions')
        
        self.shared_results['assumptions'] = result
        return {"status": "completed", "component": "assumptions"}
    
    def _handle_complete_analysis(self, message):
        """Handle complete analysis in single mega-prompt"""
        user_input = message.get('user_input', '')
        domain = message.get('domain', 'unknown')
        
        prompt = self._create_complete_analysis_prompt(user_input, domain)
        result = self._call_llm_sync(prompt, 'complete_analysis')
        
        # Update shared results with all components
        if isinstance(result, dict):
            for component in ['objectives', 'constraints', 'context', 'stakeholders', 'technical_requirements', 'assumptions']:
                if component in result:
                    self.shared_results[component] = result[component]
        
        return {"status": "completed", "component": "complete_analysis"}
    
    def _create_complete_analysis_prompt(self, user_input: str, domain: str) -> str:
        """Create single mega-prompt for complete intent analysis"""
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
    
    def _call_llm_sync(self, prompt: str, component_name: str) -> dict:
        """Synchronous LLM call for handler methods"""
        
        try:
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
            
            # Extract JSON from response
            if '```json' in result_text:
                json_start = result_text.find('```json') + 7
                json_end = result_text.find('```', json_start)
                result_text = result_text[json_start:json_end].strip()
            elif '[' in result_text or '{' in result_text:
                start_idx = min(
                    result_text.find('[') if '[' in result_text else len(result_text),
                    result_text.find('{') if '{' in result_text else len(result_text)
                )
                result_text = result_text[start_idx:]
            
            parsed_result = json.loads(result_text)
            return parsed_result
            
        except Exception as e:
            print(f"❌ LLM call failed for {component_name}: {str(e)}")
            return [] if component_name != 'context' else {}


# ========================================
# Simple Usage Example
# ========================================

async def example_usage():
    """Example of how to use IntentLLMRunner"""
    
    runner = IntentLLMRunner(base_port=5700, merge_strategy="multi_port")
    
    user_input = "I need a new POS system for my restaurant that handles orders, payments, and inventory with a $25k budget in 6 weeks"
    domain = "restaurant_management"
    
    # Transform user input into completed intent graph
    intent_graph = await runner.analyze_intent(user_input, domain)
    
    # Use the completed graph
    print(f"\n📊 COMPLETED INTENT GRAPH:")
    print(f"   Nodes: {len(intent_graph.nodes())}")
    print(f"   Edges: {len(intent_graph.edges())}")
    print(f"   Domain: {intent_graph.domain_context}")
    print(f"   Sentiment: {intent_graph.user_sentiment}")
    
    return intent_graph

if __name__ == "__main__":
    print("Intent LLM Runner - Core Engine")
    asyncio.run(example_usage())
