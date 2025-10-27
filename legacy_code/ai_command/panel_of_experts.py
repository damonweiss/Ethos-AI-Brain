#!/usr/bin/env python3
"""
Panel of Experts - Permanent ZMQ Scatter-Gather System
Strategic Advisor, Mission Planner, Quality Assurance, Technical Architect, Documentation Specialist
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import zmq.asyncio

from command_structures import (
    IntakeAssessment, ExpertConsultation, ExpertResponse, 
    ExecutionStrategy, ZMQPattern
)
from meta_reasoning_engine import MetaReasoningEngine

logger = logging.getLogger(__name__)

class ExpertType:
    STRATEGIC_ADVISOR = "strategic_advisor"
    MISSION_PLANNER = "mission_planner" 
    QUALITY_ASSURANCE = "quality_assurance"
    TECHNICAL_ARCHITECT = "technical_architect"
    DOCUMENTATION_SPECIALIST = "documentation_specialist"

class PanelOfExperts:
    """Permanent Panel of Experts with ZMQ Scatter-Gather coordination"""
    
    def __init__(self, zmq_context=None):
        self.zmq_context = zmq_context or zmq.asyncio.Context()
        self.experts = {}
        self.consultation_results = {}
        
        # ZMQ Scatter-Gather setup
        self.scatter_socket = None  # PUSH to experts
        self.gather_socket = None   # PULL from experts
        self.expert_sockets = {}    # Individual expert sockets
        
        # Panel composition
        self.expert_types = [
            ExpertType.STRATEGIC_ADVISOR,
            ExpertType.MISSION_PLANNER,
            ExpertType.QUALITY_ASSURANCE, 
            ExpertType.TECHNICAL_ARCHITECT,
            ExpertType.DOCUMENTATION_SPECIALIST
        ]
        
        self.is_initialized = False
    
    async def initialize_panel(self):
        """Initialize the permanent Panel of Experts"""
        if self.is_initialized:
            return
        
        logger.info("🎖️  Initializing Panel of Experts with ZMQ Scatter-Gather")
        
        # Setup scatter socket (Major General -> Experts)
        self.scatter_socket = self.zmq_context.socket(zmq.PUSH)
        self.scatter_socket.bind("tcp://*:5500")  # Panel command port
        
        # Setup gather socket (Experts -> Major General)
        self.gather_socket = self.zmq_context.socket(zmq.PULL)
        self.gather_socket.bind("tcp://*:5501")  # Panel response port
        
        # Initialize individual experts
        for i, expert_type in enumerate(self.expert_types):
            expert = Expert(expert_type, port_base=5510 + i, zmq_context=self.zmq_context)
            await expert.initialize()
            self.experts[expert_type] = expert
        
        self.is_initialized = True
        logger.info(f"✅ Panel of Experts initialized with {len(self.experts)} experts")
    
    async def consult_panel(self, consultation: ExpertConsultation) -> Dict[str, ExpertResponse]:
        """Scatter-Gather consultation with Panel of Experts"""
        if not self.is_initialized:
            await self.initialize_panel()
        
        logger.info(f"🎯 Consulting Panel of Experts: {consultation.consultation_id}")
        
        # Scatter: Send consultation to all required experts
        scatter_tasks = []
        for expert_type in consultation.required_experts:
            if expert_type in self.experts:
                task = self._scatter_to_expert(expert_type, consultation)
                scatter_tasks.append(task)
        
        # Execute scatter phase
        await asyncio.gather(*scatter_tasks)
        
        # Gather: Collect responses from experts
        responses = await self._gather_expert_responses(
            consultation.consultation_id,
            len(consultation.required_experts),
            consultation.timeout_seconds
        )
        
        # Store results for future reference
        self.consultation_results[consultation.consultation_id] = responses
        
        logger.info(f"✅ Panel consultation complete: {len(responses)} expert responses")
        return responses
    
    async def _scatter_to_expert(self, expert_type: str, consultation: ExpertConsultation):
        """Send consultation to specific expert"""
        expert = self.experts[expert_type]
        
        consultation_message = {
            "consultation_id": consultation.consultation_id,
            "mission_objective": consultation.mission_objective,
            "specific_question": consultation.specific_question,
            "intake_assessment": consultation.intake_assessment.__dict__,
            "expert_type": expert_type
        }
        
        await expert.receive_consultation(consultation_message)
    
    async def _gather_expert_responses(self, consultation_id: str, expected_count: int, timeout: float) -> Dict[str, ExpertResponse]:
        """Gather responses from experts"""
        responses = {}
        start_time = asyncio.get_event_loop().time()
        
        while len(responses) < expected_count:
            # Check timeout
            if asyncio.get_event_loop().time() - start_time > timeout:
                logger.warning(f"Panel consultation timeout: {consultation_id}")
                break
            
            try:
                # Poll for responses with short timeout
                response_data = await asyncio.wait_for(
                    self.gather_socket.recv_json(),
                    timeout=1.0
                )
                
                if response_data.get("consultation_id") == consultation_id:
                    expert_type = response_data.get("expert_type")
                    responses[expert_type] = ExpertResponse(
                        expert_type=expert_type,
                        consultation_id=consultation_id,
                        analysis=response_data.get("analysis", ""),
                        confidence=response_data.get("confidence", 0.5),
                        recommendations=response_data.get("recommendations", []),
                        follow_up_needed=response_data.get("follow_up_needed", False)
                    )
                
            except asyncio.TimeoutError:
                continue  # Keep polling
        
        return responses
    
    async def shutdown(self):
        """Shutdown Panel of Experts"""
        logger.info("🎖️  Shutting down Panel of Experts")
        
        # Shutdown individual experts
        for expert in self.experts.values():
            await expert.shutdown()
        
        # Close ZMQ sockets
        if self.scatter_socket:
            self.scatter_socket.close()
        if self.gather_socket:
            self.gather_socket.close()
        
        self.is_initialized = False

class Expert:
    """Individual expert in the Panel of Experts"""
    
    def __init__(self, expert_type: str, port_base: int, zmq_context=None):
        self.expert_type = expert_type
        self.port_base = port_base
        self.zmq_context = zmq_context or zmq.asyncio.Context()
        
        # Each expert has MetaReasoning capability
        self.meta_reasoning = MetaReasoningEngine()
        
        # ZMQ sockets
        self.command_socket = None  # Receive consultations
        self.response_socket = None  # Send responses
        
        self.is_running = False
    
    async def initialize(self):
        """Initialize expert with ZMQ connectivity"""
        logger.info(f"🎯 Initializing {self.expert_type} expert")
        
        # Setup command socket (receive consultations)
        self.command_socket = self.zmq_context.socket(zmq.PULL)
        self.command_socket.connect("tcp://localhost:5500")  # Connect to scatter
        
        # Setup response socket (send responses)
        self.response_socket = self.zmq_context.socket(zmq.PUSH)
        self.response_socket.connect("tcp://localhost:5501")  # Connect to gather
        
        self.is_running = True
        
        # Start expert processing loop
        asyncio.create_task(self._expert_processing_loop())
    
    async def receive_consultation(self, consultation_data: Dict[str, Any]):
        """Receive consultation request"""
        # This would be called by the scatter mechanism
        # For now, we'll handle it directly in the processing loop
        pass
    
    async def _expert_processing_loop(self):
        """Main processing loop for expert"""
        while self.is_running:
            try:
                # Wait for consultation requests
                consultation_data = await asyncio.wait_for(
                    self.command_socket.recv_json(),
                    timeout=1.0
                )
                
                # Process consultation
                response = await self._process_consultation(consultation_data)
                
                # Send response
                await self.response_socket.send_json(response)
                
            except asyncio.TimeoutError:
                continue  # Keep polling
            except Exception as e:
                logger.error(f"Expert {self.expert_type} processing error: {e}")
    
    async def _process_consultation(self, consultation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process consultation using expert's specialized knowledge"""
        
        mission_objective = consultation_data.get("mission_objective", "")
        specific_question = consultation_data.get("specific_question", "")
        
        # Create expert-specific prompt
        expert_prompt = self._create_expert_prompt(mission_objective, specific_question)
        
        # Use MetaReasoning for expert analysis
        analysis_result = await self.meta_reasoning.llm_backends["analyst"].complete(
            expert_prompt, {"expert_type": self.expert_type}
        )
        
        # Extract recommendations (simplified)
        recommendations = self._extract_recommendations(analysis_result)
        
        return {
            "consultation_id": consultation_data.get("consultation_id"),
            "expert_type": self.expert_type,
            "analysis": analysis_result,
            "confidence": 0.85,  # Could be calculated
            "recommendations": recommendations,
            "follow_up_needed": len(recommendations) > 3
        }
    
    def _create_expert_prompt(self, mission_objective: str, specific_question: str) -> str:
        """Create expert-specific analysis prompt"""
        
        expert_prompts = {
            ExpertType.STRATEGIC_ADVISOR: f"""
As a Strategic Advisor, analyze this mission from a high-level strategic perspective:
Mission: {mission_objective}
Question: {specific_question}

Provide strategic analysis covering:
1. Strategic implications and risks
2. Alignment with organizational objectives
3. Resource allocation recommendations
4. Success metrics and KPIs
""",
            
            ExpertType.MISSION_PLANNER: f"""
As a Mission Planner, create tactical execution plans for:
Mission: {mission_objective}
Question: {specific_question}

Provide planning analysis covering:
1. Phase-by-phase execution plan
2. Timeline and milestone recommendations
3. Resource requirements and dependencies
4. Risk mitigation strategies
""",
            
            ExpertType.TECHNICAL_ARCHITECT: f"""
As a Technical Architect, analyze the technical aspects of:
Mission: {mission_objective}
Question: {specific_question}

Provide technical analysis covering:
1. Architecture recommendations
2. Technology stack suggestions
3. Scalability and performance considerations
4. Integration and security requirements
""",
            
            ExpertType.QUALITY_ASSURANCE: f"""
As a Quality Assurance specialist, evaluate quality aspects of:
Mission: {mission_objective}
Question: {specific_question}

Provide QA analysis covering:
1. Quality standards and requirements
2. Testing strategies and approaches
3. Risk assessment and mitigation
4. Compliance and validation requirements
""",
            
            ExpertType.DOCUMENTATION_SPECIALIST: f"""
As a Documentation Specialist, analyze documentation needs for:
Mission: {mission_objective}
Question: {specific_question}

Provide documentation analysis covering:
1. Documentation requirements and standards
2. Knowledge management strategies
3. Training and onboarding materials
4. Communication and reporting frameworks
"""
        }
        
        return expert_prompts.get(self.expert_type, f"Analyze: {mission_objective}")
    
    def _extract_recommendations(self, analysis_result: str) -> List[str]:
        """Extract actionable recommendations from analysis"""
        # Simplified extraction - could be more sophisticated
        recommendations = []
        
        # Look for numbered recommendations
        lines = analysis_result.split('\n')
        for line in lines:
            if any(marker in line for marker in ['1.', '2.', '3.', '•', '-']):
                clean_rec = line.strip().lstrip('1234567890.-• ')
                if len(clean_rec) > 10:  # Filter out short fragments
                    recommendations.append(clean_rec)
        
        return recommendations[:5]  # Limit to top 5
    
    async def shutdown(self):
        """Shutdown expert"""
        self.is_running = False
        if self.command_socket:
            self.command_socket.close()
        if self.response_socket:
            self.response_socket.close()

# Demo function
async def demo_panel_of_experts():
    """Demo the Panel of Experts system"""
    
    panel = PanelOfExperts()
    
    try:
        await panel.initialize_panel()
        
        # Create test consultation
        consultation = ExpertConsultation(
            consultation_id=f"consult_{uuid.uuid4().hex[:8]}",
            mission_objective="Build a secure fintech API for 10,000 transactions per second",
            specific_question="What are the key considerations for this system?",
            required_experts=[
                ExpertType.STRATEGIC_ADVISOR,
                ExpertType.TECHNICAL_ARCHITECT,
                ExpertType.QUALITY_ASSURANCE
            ],
            intake_assessment=None  # Would be real assessment
        )
        
        print("🎖️  PANEL OF EXPERTS DEMO")
        print("=" * 50)
        print(f"Consultation: {consultation.mission_objective}")
        
        # Consult panel
        responses = await panel.consult_panel(consultation)
        
        print(f"\n📊 EXPERT RESPONSES ({len(responses)} experts):")
        for expert_type, response in responses.items():
            print(f"\n🎯 {expert_type.upper()}:")
            print(f"   Analysis: {response.analysis[:100]}...")
            print(f"   Confidence: {response.confidence:.1%}")
            print(f"   Recommendations: {len(response.recommendations)}")
        
    finally:
        await panel.shutdown()

if __name__ == "__main__":
    asyncio.run(demo_panel_of_experts())
