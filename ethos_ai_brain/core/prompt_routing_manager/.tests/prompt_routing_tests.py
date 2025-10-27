#!/usr/bin/env python3
"""
Test Suite for PromptRoutingManager

Covers:
- LLM prompt (chat/completion)
- Vision prompt (text + image)
- Embeddings prompt (vector output)

Keeps it compact: params in a dict, loop through.
"""

import json
from ethos_mcp.mcp_managers.prompt_routing_manager.prompt_routing_manager import PromptRoutingManager


def run_tests():
    router = PromptRoutingManager()

    # Register 3 test prompts (normally you'd load from template files)
    router.prompt_manager.register_prompt(
        name="text_summary",
        template="Summarize this text in one sentence: {{ text }}",
        output_schema={"summary": "string"},
    )

    router.prompt_manager.register_prompt(
        name="image_classification",
        template="Classify the object in this picture: {{ description }}",
        output_schema={"label": "string"},
    )

    router.prompt_manager.register_prompt(
        name="make_embedding",
        template="Generate embedding for: {{ phrase }}",
        output_schema={"embedding": "string"},  # just enough to infer embeddings engine
    )

    # Define tests
    tests = [
        {
            "name": "LLM Test",
            "prompt": "text_summary",
            "vars": {"text": "Digital twins connect data and simulation."},
            "prefs": {"model": "gpt-4o-mini"},
        },
        {
            "name": "Vision Test",
            "prompt": "image_classification",
            "vars": {
                "description": "a sample cat image",
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
            },
            "prefs": {"model": "gpt-4o-mini"},
        },
        {
            "name": "Embeddings Test",
            "prompt": "make_embedding",
            "vars": {"phrase": "Digital twin ecosystem"},
            "prefs": {"model": "text-embedding-ada-002"},
        },
    ]

    for t in tests:
        print(f"\n=== {t['name']} ===")
        result = router.execute_prompt(
            prompt_name=t["prompt"],
            variables=t["vars"],
            model_preferences=t["prefs"],
        )
        print(json.dumps(result, indent=2)[:600])  # truncate output for readability


if __name__ == "__main__":
    run_tests()
