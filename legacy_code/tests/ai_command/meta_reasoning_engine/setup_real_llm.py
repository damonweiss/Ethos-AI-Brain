#!/usr/bin/env python3
"""
Setup Script for Real LLM Demo
Helps configure OpenAI API and other requirements
"""

import os
import sys
import subprocess

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_step(step, description):
    print(f"\n[STEP {step}] {description}")

def check_openai_library():
    """Check if OpenAI library is installed"""
    try:
        import openai
        print("✅ OpenAI library is installed")
        return True
    except ImportError:
        print("❌ OpenAI library not found")
        return False

def install_openai():
    """Install OpenAI library"""
    print("📦 Installing OpenAI library...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
        print("✅ OpenAI library installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install OpenAI library: {e}")
        return False

def check_api_key():
    """Check if OpenAI API key is set"""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        # Mask the key for security
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"✅ OpenAI API key found: {masked_key}")
        return True
    else:
        print("❌ OpenAI API key not found")
        return False

def setup_instructions():
    """Provide setup instructions"""
    print_header("🔧 Real LLM Demo Setup")
    print("This script will help you set up real LLM integration")
    
    print_step(1, "Check OpenAI Library")
    has_openai = check_openai_library()
    
    if not has_openai:
        print("\n🔧 Installing OpenAI library...")
        install_choice = input("Install OpenAI library now? (y/n): ").lower()
        if install_choice == 'y':
            if install_openai():
                has_openai = True
    
    print_step(2, "Check API Key")
    has_api_key = check_api_key()
    
    if not has_api_key:
        print("\n🔑 OpenAI API Key Setup:")
        print("1. Go to: https://platform.openai.com/api-keys")
        print("2. Create a new API key")
        print("3. Set environment variable:")
        print("   Windows: set OPENAI_API_KEY=your_key_here")
        print("   Linux/Mac: export OPENAI_API_KEY=your_key_here")
        print("4. Or add to your .env file")
        
        manual_key = input("\nEnter your API key now (or press Enter to skip): ").strip()
        if manual_key:
            os.environ["OPENAI_API_KEY"] = manual_key
            print("✅ API key set for this session")
            has_api_key = True
    
    print_step(3, "Setup Status")
    if has_openai and has_api_key:
        print("🎉 Setup complete! You can now run real LLM demos:")
        print("   python demo_real_llm.py")
        print("\n💡 Try these demos:")
        print("   • Startup planning with real AI")
        print("   • Home renovation planning")
        print("   • Real-time AI streaming responses")
    else:
        print("⚠️  Setup incomplete:")
        if not has_openai:
            print("   - Install OpenAI library: pip install openai")
        if not has_api_key:
            print("   - Set OPENAI_API_KEY environment variable")
        print("\n📖 Once setup is complete, run: python demo_real_llm.py")
    
    print_step(4, "Cost Information")
    print("💰 OpenAI API Usage Costs:")
    print("   • GPT-3.5-turbo: ~$0.001 per 1K tokens")
    print("   • GPT-4: ~$0.03 per 1K tokens")
    print("   • Demo typically uses 2K-5K tokens (~$0.01-0.15)")
    print("   • Start with GPT-3.5-turbo for cost-effective testing")
    
    print_step(5, "Alternative Options")
    print("🔄 If you don't want to use OpenAI:")
    print("   • Run demo_human_readable.py for mock LLM demo")
    print("   • Modify RealLLMBackend to use other APIs (Anthropic, etc.)")
    print("   • Use local LLMs with Ollama or similar")

def test_connection():
    """Test OpenAI API connection"""
    print_header("🧪 Testing OpenAI Connection")
    
    try:
        import openai
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            print("❌ No API key found")
            return False
        
        client = openai.OpenAI(api_key=api_key)
        
        print("🔍 Testing API connection...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Hello from MetaReasoningEngine!'"}],
            max_tokens=20
        )
        
        result = response.choices[0].message.content
        print(f"✅ Connection successful!")
        print(f"🤖 AI Response: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def main():
    """Main setup function"""
    print_header("🚀 MetaReasoningEngine Real LLM Setup")
    print("Welcome! This script will help you set up real LLM integration.")
    
    choice = input("\nWhat would you like to do?\n1. Setup check\n2. Test connection\n3. Both\nChoice (1-3): ").strip()
    
    if choice in ['1', '3']:
        setup_instructions()
    
    if choice in ['2', '3']:
        print("\n" + "-" * 60)
        test_connection()
    
    print_header("🎯 Next Steps")
    print("Once setup is complete:")
    print("1. Run: python demo_real_llm.py")
    print("2. Watch real AI reasoning in action!")
    print("3. Experience the future of cognitive architecture!")

if __name__ == "__main__":
    main()
