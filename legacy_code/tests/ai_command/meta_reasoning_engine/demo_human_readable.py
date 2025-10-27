#!/usr/bin/env python3
"""
MetaReasoningEngine - Human-Readable Demo
A relatable demonstration of the AI reasoning system in action
"""

import sys
import os
import asyncio
import time
from datetime import datetime

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ai_command'))

from meta_reasoning_engine import (
    MetaReasoningEngine,
    ReasoningContext,
    CollaborationMode,
    ConfidenceLevel
)

def print_header(title):
    """Print a nice header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_section(title):
    """Print a section header"""
    print(f"\n🔍 {title}")
    print("-" * 50)

def print_step(step_num, description):
    """Print a step"""
    print(f"\n[STEP {step_num}] {description}")

def print_response(speaker, message):
    """Print a formatted response"""
    print(f"\n💬 {speaker}:")
    print(f"   {message}")

async def demo_startup_planning():
    """Demo: Planning a Tech Startup - A Relatable Use Case"""
    
    print_header("🚀 DEMO: AI-Powered Startup Planning Assistant")
    print("Scenario: You're an entrepreneur with a great idea but need help planning")
    print("Let's see how the MetaReasoningEngine breaks down complex business challenges!")
    
    # Initialize the reasoning engine
    print_section("Initializing AI Reasoning System")
    engine = MetaReasoningEngine()
    print("✅ MetaReasoningEngine loaded with 4 AI specialists:")
    print("   • Strategic Analyst - For market analysis")
    print("   • Project Planner - For execution planning") 
    print("   • Citizen Engagement Bot - For customer insights")
    print("   • Task Decomposer - For breaking down complex goals")
    
    # Scenario setup
    print_section("The Business Challenge")
    startup_idea = """
    I want to create a mobile app that helps busy parents find and book 
    last-minute babysitters in their neighborhood. Think 'Uber for babysitting' 
    but with verified, trusted caregivers and real-time availability.
    """
    print(f"💡 Your Startup Idea: {startup_idea.strip()}")
    
    constraints = {
        "budget": "$50,000",
        "timeline": "6 months to MVP", 
        "team_size": "Just me (solo founder)",
        "experience": "Software developer but new to business",
        "target_market": "Urban parents aged 25-40"
    }
    
    print("\n📋 Your Constraints:")
    for key, value in constraints.items():
        print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    # Create reasoning context
    context = ReasoningContext(
        goal=startup_idea.strip(),
        constraints=constraints,
        user_preferences={
            "approach": "lean startup methodology",
            "risk_tolerance": "moderate",
            "priority": "user safety and trust"
        }
    )
    
    # Step 1: Strategic Analysis
    print_step(1, "Strategic Market Analysis")
    print("🤖 Consulting our Strategic Analyst...")
    
    analysis_prompt = f"Analyze this startup idea: {startup_idea}"
    analyst_response = await engine.llm_backends["analyst"].complete(analysis_prompt, {})
    print_response("Strategic Analyst", analyst_response)
    
    # Step 2: Goal Decomposition  
    print_step(2, "Breaking Down the Complex Goal")
    print("🤖 Our Task Decomposer is analyzing your startup challenge...")
    
    decompose_prompt = f"Decompose this startup goal into actionable steps: {startup_idea}"
    decomposer_response = await engine.llm_backends["decomposer"].complete(decompose_prompt, {})
    print_response("Task Decomposer", decomposer_response)
    
    # Step 3: Execution Planning
    print_step(3, "Creating Your 6-Month Action Plan")
    print("🤖 Our Project Planner is creating your roadmap...")
    
    planning_prompt = f"Create a 6-month plan for: {startup_idea} with budget {constraints['budget']}"
    planner_response = await engine.llm_backends["planner"].complete(planning_prompt, {})
    print_response("Project Planner", planner_response)
    
    # Step 4: Customer Insights
    print_step(4, "Understanding Your Customers")
    print("🤖 Our Citizen Engagement Bot is analyzing user needs...")
    
    citizen_prompt = "What questions should we ask parents about their babysitting needs and concerns?"
    citizen_response = await engine.llm_backends["citizen_engagement"].complete(citizen_prompt, {})
    print_response("Customer Insights Bot", citizen_response)
    
    # Step 5: Full Reasoning Engine
    print_step(5, "Complete AI Reasoning Analysis")
    print("🤖 Now let's see the full MetaReasoningEngine in action...")
    print("   This combines ALL our AI specialists working together!")
    
    start_time = time.time()
    full_result = await engine.reason(startup_idea.strip(), context)
    end_time = time.time()
    
    print(f"\n⚡ Reasoning completed in {end_time - start_time:.2f} seconds")
    print_response("MetaReasoningEngine", f"Analysis Result: {full_result}")
    
    # Summary
    print_section("What Just Happened?")
    print("✨ The MetaReasoningEngine demonstrated:")
    print("   1. 🧠 Multi-specialist AI collaboration")
    print("   2. 📊 Structured goal decomposition") 
    print("   3. 📋 Strategic planning and analysis")
    print("   4. 👥 Customer-focused insights")
    print("   5. ⚡ Fast, comprehensive reasoning")
    
    print("\n🎯 Real-World Applications:")
    print("   • Business planning and strategy")
    print("   • Project management and execution")
    print("   • Market research and analysis") 
    print("   • Risk assessment and mitigation")
    print("   • Customer development and validation")
    
    print_header("🎉 Demo Complete!")
    print("This is how AI-powered reasoning can transform complex decision-making!")
    print("From a vague startup idea to actionable insights in seconds! 🚀")

async def demo_personal_project():
    """Demo: Planning a Personal Project - Home Renovation"""
    
    print_header("🏠 BONUS DEMO: AI-Powered Home Renovation Planning")
    print("Scenario: You want to renovate your kitchen but don't know where to start")
    
    engine = MetaReasoningEngine()
    
    renovation_goal = """
    I want to renovate my 1980s kitchen to be modern, functional, and increase 
    my home's value. I love cooking but the current layout is cramped and outdated.
    """
    
    print(f"🏠 Your Project: {renovation_goal.strip()}")
    
    constraints = {
        "budget": "$25,000",
        "timeline": "3 months",
        "diy_skills": "Beginner (can paint, that's about it)",
        "family_situation": "Two kids, need kitchen functional during reno",
        "home_style": "1980s ranch house"
    }
    
    print("\n📋 Your Situation:")
    for key, value in constraints.items():
        print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    # Quick analysis
    print_section("AI Analysis in Action")
    
    analysis_prompt = f"Analyze this home renovation project: {renovation_goal}"
    analyst_response = await engine.llm_backends["analyst"].complete(analysis_prompt, {})
    print_response("Home Renovation Analyst", analyst_response)
    
    planning_prompt = f"Create a 3-month renovation plan for: {renovation_goal} with budget $25,000"
    planner_response = await engine.llm_backends["planner"].complete(planning_prompt, {})
    print_response("Renovation Planner", planner_response)
    
    print_section("Key Takeaway")
    print("🎯 The same AI reasoning system works for ANY complex planning:")
    print("   • Business ventures")
    print("   • Personal projects") 
    print("   • Career decisions")
    print("   • Investment planning")
    print("   • Life changes")
    
    print("\n💡 The power is in the COGNITIVE ARCHITECTURE:")
    print("   • Goal decomposition")
    print("   • Multi-perspective analysis")
    print("   • Constraint-aware planning")
    print("   • Risk assessment")
    print("   • Actionable recommendations")

async def main():
    """Run the human-readable demos"""
    print_header("🎭 MetaReasoningEngine: AI Reasoning Made Human")
    print("Welcome to a demonstration of advanced AI cognitive architecture!")
    print("We'll show you how artificial intelligence can break down complex")
    print("real-world problems just like a team of human experts would.")
    
    try:
        # Main startup demo
        await demo_startup_planning()
        
        # Bonus personal project demo
        await demo_personal_project()
        
        print_header("🌟 The Future of AI-Assisted Decision Making")
        print("What you just witnessed is the foundation of:")
        print("   🤖 AI agents that think like humans")
        print("   🧠 Cognitive architectures for complex reasoning")
        print("   🚀 The next generation of AI-powered tools")
        print("   💼 Business intelligence that actually understands context")
        print("   🎯 Personal AI assistants that help you plan and execute")
        
        print("\n✨ This is just the beginning...")
        print("Imagine this reasoning power integrated into:")
        print("   • Your IDE for project planning")
        print("   • Your business tools for strategy")
        print("   • Your personal apps for life decisions")
        print("   • Your team workflows for collaboration")
        
        print_header("🎉 Thank You for Exploring AI Reasoning!")
        
    except Exception as e:
        print(f"\n❌ Demo Error: {e}")
        print("Don't worry - this just shows the system is real, not scripted!")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
