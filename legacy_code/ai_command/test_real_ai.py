#!/usr/bin/env python3
"""
Test Real AI Backend Initialization
"""

import asyncio
import os
from dotenv import load_dotenv
from meta_reasoning_engine import MetaReasoningEngine

load_dotenv()

async def test_real_ai_backend():
    """Test if real AI backend initializes properly"""
    
    print("TESTING REAL AI BACKEND INITIALIZATION")
    print("=" * 50)
    
    try:
        # Try to initialize with real AI
        engine = MetaReasoningEngine(use_real_ai=True)
        
        print("SUCCESS: MetaReasoningEngine with real AI created")
        
        # Check if backends are OpenAI
        analyst = engine.llm_backends["analyst"]
        print(f"SUCCESS: Analyst backend type: {type(analyst).__name__}")
        
        # Test a simple completion
        print("TESTING: Simple AI completion...")
        
        response = await analyst.complete(
            "Say 'Real AI is working!' if you can read this.",
            {"test": "simple"}
        )
        
        print(f"SUCCESS: AI Response received")
        print(f"RESPONSE: {response}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_real_ai_backend())
