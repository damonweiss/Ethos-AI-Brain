#!/usr/bin/env python3
"""
Demo script for InferenceModelManager.

Tests model discovery, provider filtering, and cost information.
"""

import json
import argparse
import sys
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# from inference_model_manager import InferenceModelManager
from ..inference_model_manager import InferenceModelManager


def show_model_info(manager: InferenceModelManager, model: str) -> None:
    """Pretty-print model info for a single model."""
    info = manager.get_model_info(model)
    if not info:
        print(f"  [INFO] No info available for {model}")
        return
    print(f"  {model} info:")
    for k, v in list(info.items())[:10]:  # show first 10 fields
        print(f"    {k}: {v}")


def demo_model_manager(as_json: bool = False, quiet: bool = False):
    """Run InferenceModelManager demo. Optionally export as JSON."""
    manager = InferenceModelManager()
    results: dict = {"errors": []}

    # --- Model discovery ---
    discovery = manager.discover_models()
    results["discovery"] = discovery
    if not quiet:
        print("Model Discovery:")

    if "error" in discovery.get("llm", {}):
        err = discovery["llm"]["error"]
        results["errors"].append(err)
        if not quiet:
            print(f"  [FAILURE] {err}")
        return results

    if not quiet:
        print(f"  Providers: {discovery['llm'].get('providers', [])}")

    # --- Provider info ---
    providers = manager.get_available_providers()
    results["providers"] = providers
    if not quiet:
        print(f"Available providers: {providers}")

    if providers:
        sample_provider = providers[0]
        models = manager.get_models_by_provider(sample_provider)
        results["sample_provider"] = {sample_provider: models[:3]}
        if not quiet:
            print(f"{sample_provider.upper()}: {len(models)} models (sample: {models[:3]})")

    # --- Cost info ---
    test_models = ["gpt-4o", "claude-3-5-sonnet-20241022"]
    results["cost_info"] = {}
    if not quiet:
        print("Cost Information:")
    for m in test_models:
        info = manager.get_model_cost_info(m)
        results["cost_info"][m] = info
        if not quiet:
            print(f"  {m}: {info}")

    # --- Model info ---
    if not quiet:
        show_model_info(manager, "gpt-4o")

    # --- Filtering ---
    cheap_models = manager.get_cheapest_models(limit=3)
    results["cheapest"] = cheap_models
    if not quiet:
        print("Cheapest models:", cheap_models)

    vision_models = manager.filter_models(supports_vision=True)
    results["vision_models"] = vision_models[:3]
    if not quiet:
        print("Vision models:", vision_models[:3])

    if as_json:
        with open("demo_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        if not quiet:
            print("[INFO] Results saved to demo_results.json")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Export results to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    args = parser.parse_args()

    try:
        demo_model_manager(as_json=args.json, quiet=args.quiet)
    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        sys.exit(1)
