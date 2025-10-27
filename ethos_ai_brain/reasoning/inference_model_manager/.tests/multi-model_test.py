#!/usr/bin/env python3
"""
Quick multi-engine smoke test (LLM, Vision, Embeddings).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import json
from ethos_mcp.mcp_managers.inference_engines.llm_engine import LLMEngine
from ethos_mcp.mcp_managers.inference_engines.vision_engine import VisionEngine
from ethos_mcp.mcp_managers.inference_engines.embeddings_engine import EmbeddingsEngine

def run_tests():
    tests = [
        {
            "name": "LLM",
            "engine": LLMEngine(model="gpt-4o-mini"),
            "inputs": {"input_data": "Give me a JSON object with one key 'hello'."},
        },
        {
            "name": "Vision",
            "engine": VisionEngine(model="gpt-4o-mini"),
            "inputs": {
                "input_data": "What animal is in this picture?",
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
            },
        },
        {
            "name": "Embeddings",
            "engine": EmbeddingsEngine(model="text-embedding-ada-002"),
            "inputs": {"input_data": "The quick brown fox jumps over the lazy dog."},
        },
    ]

    for test in tests:
        print(f"\n=== {test['name']}Engine ===")
        result = test["engine"].run(**test["inputs"])
        print(json.dumps(result, indent=2)[:500])  # print first 500 chars

if __name__ == "__main__":
    run_tests()
