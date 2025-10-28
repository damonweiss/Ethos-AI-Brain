#!/usr/bin/env python3
"""
Demo Web Scraper Tool
Show actual web scraper execution and output.
"""

import sys
import importlib.util
from pathlib import Path

# Load the web scraper tool
project_root = Path(__file__).resolve().parent
web_scraper_path = project_root / "ethos_ai_brain" / "orchestration" / "mcp" / "tools" / "utilities" / "web" / "web_scraper" / "tool.py"

spec = importlib.util.spec_from_file_location("web_scraper_tool", web_scraper_path)
web_scraper_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web_scraper_module)

print("🌐 Web Scraper Tool Demo")
print("=" * 40)

# Test 1: Simple URL
print("\n📄 Test 1: Simple HTML page")
result1 = web_scraper_module.scrape_webpage({"url": "https://httpbin.org/html"})
print(f"Status: {result1.get('status_code', 'N/A')}")
print(f"Content length: {len(result1.get('content', ''))}")
if 'error' in result1:
    print(f"Error: {result1['error']}")
else:
    print(f"Content preview: {result1.get('content', '')[:200]}...")

# Test 2: With CSS selector
print("\n🎯 Test 2: With CSS selector")
result2 = web_scraper_module.scrape_webpage({
    "url": "https://httpbin.org/html",
    "selector": "h1"
})
print(f"Status: {result2.get('status_code', 'N/A')}")
if 'error' in result2:
    print(f"Error: {result2['error']}")
else:
    print(f"Selected content: {result2.get('content', 'No content')}")

# Test 3: Invalid URL
print("\n❌ Test 3: Invalid URL handling")
result3 = web_scraper_module.scrape_webpage({"url": "https://invalid-url-that-does-not-exist.com"})
print(f"Status: {result3.get('status_code', 'N/A')}")
if 'error' in result3:
    print(f"Error handled gracefully: {result3['error'][:100]}...")

print("\n✅ Web scraper demo complete!")
