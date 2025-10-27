#!/usr/bin/env python3
"""
InferenceModelEnricher - LLM-powered model metadata enrichment

Separates enrichment concerns from InferenceModelManager.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from pydantic import ValidationError
# from ethos_mcp.mcp_managers.inference_model_manager.model_schemas import EnrichedModelMetadata
from .model_schemas import EnrichedModelMetadata

# TODO: Dual-mode for LLM and non-LLM models
class InferenceModelEnricher:
    def __init__(self, manager, reasoning_model: str = "gpt-4o"):
        """
        Args:
            manager: InferenceModelManager instance
            reasoning_model: Which LLM to use for enrichment
        """
        self.manager = manager
        self.reasoning_model = reasoning_model

    def enrich_curated_models(self) -> Dict[str, Any]:
        """Enrich curated model metadata using the reasoning LLM."""
        curated_models = self._load_curated_models()
        if not curated_models:
            return {"error": "No curated models found"}

        enriched_models = {}
        for model_name in curated_models:
            litellm_data = self.manager.get_model_cost_info(model_name)
            if not litellm_data or "error" in litellm_data:
                continue

            enriched = self._enrich_model(model_name, litellm_data)
            if enriched and "error" not in enriched:
                enriched_models[model_name] = enriched

        if enriched_models:
            self._save_enriched_metadata(enriched_models)

        return {
            "enriched_models": enriched_models,
            "total_enriched": len(enriched_models),
            "reasoning_model_used": self.reasoning_model,
            "enriched_at": datetime.now().isoformat()
        }

    def _enrich_model(self, model_name: str, litellm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich a single model with metadata via LiteLLM."""
        try:
            import litellm

            prompt = self._create_prompt(model_name, litellm_data)
            response = litellm.completion(
                model=self.reasoning_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert AI model analyst. "
                                   "Analyze the provided model data and return enriched metadata in JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            enriched_data = json.loads(response.choices[0].message.content)
            cleaned = self._clean_llm_response(enriched_data)

            # Validate against schema
            validated = EnrichedModelMetadata(**cleaned)
            return validated.model_dump()

        except (ValidationError, Exception) as e:
            return {"error": f"Failed to enrich {model_name}: {e}"}

    def _create_prompt(self, model_name: str, litellm_data: Dict[str, Any]) -> str:
        """Create the enrichment prompt text."""
        input_cost = litellm_data.get("input_cost_per_token", 0) * 1000
        output_cost = litellm_data.get("output_cost_per_token", 0) * 1000
        max_input_tokens = litellm_data.get("max_input_tokens", "unknown")
        max_output_tokens = litellm_data.get("max_output_tokens", "unknown")
        supports_vision = litellm_data.get("supports_vision", False)
        supports_function_calling = litellm_data.get("supports_function_calling", False)
        supports_json = litellm_data.get("supports_response_schema", False)

        provider = "unknown"
        if "gpt" in model_name or "openai" in model_name:
            provider = "openai"
        elif "claude" in model_name or "anthropic" in model_name:
            provider = "anthropic"
        elif "gemini" in model_name or "google" in model_name:
            provider = "google"

        return f"""
Model: {model_name}
LiteLLM Data:
- Input cost/token: {litellm_data.get("input_cost_per_token", "unknown")}
- Output cost/token: {litellm_data.get("output_cost_per_token", "unknown")}
- Max input tokens: {max_input_tokens}
- Max output tokens: {max_output_tokens}
- Supports vision: {supports_vision}
- Supports function calling: {supports_function_calling}
- Supports JSON: {supports_json}

Task: Return enriched metadata in JSON matching schema.
Provider guess: {provider}
Effective date: {datetime.now().strftime("%Y-%m-%d")}
"""

    def _clean_llm_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up None/null values from the LLM response."""
        def clean(v):
            if v is None:
                return False
            if isinstance(v, dict):
                return {k: clean(val) for k, val in v.items()}
            if isinstance(v, list):
                return [clean(x) for x in v]
            return v
        return clean(data)

    def _save_enriched_metadata(self, enriched_models: Dict[str, Any]):
        """Save enriched metadata to JSON file."""
        metadata_dir = Path(__file__).parent / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        metadata_file = metadata_dir / "inference_model_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(enriched_models, f, indent=2, ensure_ascii=False)

    def _load_curated_models(self) -> List[str]:
        """Load curated model names from curated_llm_models.py."""
        metadata_dir = Path(__file__).parent / "metadata"
        curated_file = metadata_dir / "curated_llm_models.py"
        if not curated_file.exists():
            return []
        local_vars = {}
        exec(curated_file.read_text(), {}, local_vars)
        return local_vars.get("curated_llm_models", [])


# --------------------------
# Demo/Test
# --------------------------
def main():
    import os
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY missing")
        return

    manager = InferenceModelManager(auto_discover=False)
    enricher = InferenceModelEnricher(manager, reasoning_model="gpt-4o")

    results = enricher.enrich_curated_models()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
