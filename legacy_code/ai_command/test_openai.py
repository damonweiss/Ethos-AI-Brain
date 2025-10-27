#!/usr/bin/env python3
"""
Test OpenAI API Key and Connection
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_openai_connection():
    """Test if OpenAI API key is working"""
    
    api_key = os.getenv('OPENAI_API_KEY')
    
    print("OPENAI API KEY TEST")
    print("=" * 40)
    
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment")
        print("SOLUTION: Set it with: set OPENAI_API_KEY=your_key_here")
        return False
    
    # Show masked key
    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
    print(f"SUCCESS: API Key found: {masked_key}")
    
    # Test OpenAI import
    try:
        import openai
        print("SUCCESS: OpenAI package imported successfully")
    except ImportError:
        print("ERROR: OpenAI package not installed")
        print("SOLUTION: Install with: pip install openai")
        return False
    
    # Test API connection
    try:
        client = openai.AsyncOpenAI(api_key=api_key)
        
        print("TESTING: API connection...")
        
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use cheaper model for testing
            messages=[
                {"role": "user", "content": "Say 'Hello from OpenAI!' if you can read this."}
            ],
            max_tokens=20
        )
        
        result = response.choices[0].message.content
        print(f"SUCCESS: API Connection successful!")
        print(f"RESPONSE: {result}")
        return True
        
    except Exception as e:
        print(f"ERROR: API Connection failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_openai_connection())
