#!/usr/bin/env python3
"""
PromptRoutingManager - Clean Prompt Execution Engine

Handles prompt execution using PromptManager for template management
and Promptify/OpenAI for actual LLM execution. Completely agnostic to domains.
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

# from promptify import Promptify  # Temporarily disabled due to build issues
from dotenv import load_dotenv
import openai

from ..prompt_manager import PromptManager

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent / '.env'
load_dotenv(env_path)


class PromptRoutingManager:
    """
    Clean prompt execution engine.
    
    Uses PromptManager for template management and executes prompts
    via Promptify or direct OpenAI calls. Domain-agnostic.
    """
    
    def __init__(self, 
                 model: str = "gpt-4o-mini", 
                 temperature: float = 0.1):
        """
        Initialize the prompt routing manager.
        
        Args:
            model: LLM model to use for execution
            temperature: Temperature for LLM calls
        """
        self.model = model
        self.temperature = temperature
        self.prompt_manager = PromptManager()
        
        # Initialize LLM clients
        self.promptify_client = None
        self.openai_client = None
        self._init_llm_clients()
        
        # Auto-load templates on initialization
        self.prompt_manager.auto_load_templates()
    
    def set_prompt_manager(self, prompt_manager: PromptManager) -> None:
        """
        Replace the current prompt manager with a new one.
        
        Args:
            prompt_manager: New PromptManager instance to use
        """
        self.prompt_manager = prompt_manager
    
    def _init_llm_clients(self):
        """Initialize Promptify and OpenAI clients."""
        api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key:
            # Try Promptify first (disabled due to build issues)
            # try:
            #     self.promptify_client = Promptify(
            #         model=self.model,
            #         api_key=api_key
            #     )
            # except:
            #     pass
            self.promptify_client = None  # Disabled
            
            # Initialize OpenAI client as fallback
            try:
                self.openai_client = openai.OpenAI(api_key=api_key)
            except:
                pass
    
    def execute_prompt(self, 
                      prompt_name: str,
                      variables: Dict[str, Any] = None,
                      fallback_to_general: bool = True) -> Dict[str, Any]:
        """
        Execute a registered prompt with variables.
        
        Args:
            prompt_name: Name of registered prompt
            variables: Variables to pass to template
            fallback_to_general: Whether to fallback to general_analysis if prompt not found
            
        Returns:
            Execution result with structured output
        """
        variables = variables or {}
        
        # Get prompt from manager
        prompt_data = self.prompt_manager.get_prompt(prompt_name)
        
        if not prompt_data:
            if fallback_to_general and prompt_name != "general_analysis":
                return self.execute_prompt("general_analysis", variables, fallback_to_general=False)
            else:
                return {
                    "success": False,
                    "error": f"Prompt '{prompt_name}' not found",
                    "available_prompts": self.prompt_manager.list_prompts()
                }
        
        # Execute the prompt
        return self._execute_prompt_data(prompt_data, variables, prompt_name)
    
    def execute_direct(self,
                      template: str,
                      output_schema: Dict[str, Any],
                      variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a prompt directly without registration.
        
        Args:
            template: Jinja2 template string
            output_schema: Expected output schema
            variables: Variables for template
            
        Returns:
            Execution result
        """
        variables = variables or {}
        
        prompt_data = {
            "template": template,
            "output_schema": output_schema,
            "template_type": "direct"
        }
        
        return self._execute_prompt_data(prompt_data, variables, "direct_execution")
    
    def _execute_prompt_data(self, 
                           prompt_data: Dict[str, Any], 
                           variables: Dict[str, Any],
                           prompt_name: str) -> Dict[str, Any]:
        """Execute prompt data with the available LLM client."""
        
        template = prompt_data["template"]
        output_schema = prompt_data["output_schema"]
        
        # Render Jinja2 template
        try:
            from jinja2 import Template
            jinja_template = Template(template)
            rendered_prompt = jinja_template.render(**variables)
        except Exception as e:
            return {
                "success": False,
                "error": f"Template rendering failed: {str(e)}",
                "template_error": True
            }
        
        # Execute with available client
        if self.promptify_client:
            return self._execute_with_promptify(rendered_prompt, output_schema, prompt_name)
        elif self.openai_client:
            return self._execute_with_openai(rendered_prompt, output_schema, prompt_name)
        else:
            return self._execute_fallback(rendered_prompt, output_schema, prompt_name)
    
    def _execute_with_promptify(self, 
                               rendered_prompt: str, 
                               output_schema: Dict[str, Any],
                               prompt_name: str) -> Dict[str, Any]:
        """Execute using Promptify client."""
        try:
            result = self.promptify_client.run(rendered_prompt, output_schema)
            
            return {
                "success": True,
                "result": result,
                "method": "promptify",
                "prompt_name": prompt_name,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model,
                    "temperature": self.temperature
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Promptify execution failed: {str(e)}",
                "method": "promptify",
                "prompt_name": prompt_name
            }
    
    def _execute_with_openai(self, 
                           rendered_prompt: str, 
                           output_schema: Dict[str, Any],
                           prompt_name: str) -> Dict[str, Any]:
        """Execute using direct OpenAI client."""
        try:
            system_prompt = f"""
You are a structured analysis assistant. Respond with JSON matching this schema:

{json.dumps(output_schema, indent=2)}

Ensure your response is valid JSON that matches the schema exactly.
"""
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": rendered_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=self.temperature
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return {
                "success": True,
                "result": result,
                "method": "openai_direct",
                "prompt_name": prompt_name,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model,
                    "temperature": self.temperature
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"OpenAI execution failed: {str(e)}",
                "method": "openai_direct",
                "prompt_name": prompt_name
            }
    
    def _execute_fallback(self, 
                         rendered_prompt: str, 
                         output_schema: Dict[str, Any],
                         prompt_name: str) -> Dict[str, Any]:
        """Fallback when no LLM client is available."""
        return {
            "success": False,
            "error": "No LLM client available",
            "fallback_result": {
                "rendered_prompt": rendered_prompt[:200] + "..." if len(rendered_prompt) > 200 else rendered_prompt,
                "expected_schema": output_schema,
                "suggestion": "Configure OpenAI API key or install promptify"
            },
            "method": "fallback",
            "prompt_name": prompt_name
        }
    
    def list_available_prompts(self) -> List[Dict[str, Any]]:
        """Get list of all available prompts with metadata."""
        prompts = []
        for prompt_name in self.prompt_manager.list_prompts():
            prompt_info = self.prompt_manager.get_prompt_info(prompt_name)
            if prompt_info:
                prompts.append(prompt_info)
        return prompts
    
    def get_prompt_info(self, prompt_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific prompt."""
        return self.prompt_manager.get_prompt_info(prompt_name)
    
    def reload_templates(self) -> Dict[str, Any]:
        """Reload all templates from the templates directory."""
        return self.prompt_manager.auto_load_templates()
    
    def is_available(self) -> bool:
        """Check if the manager has working LLM access."""
        return self.promptify_client is not None or self.openai_client is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the prompt routing manager."""
        return {
            "llm_available": self.is_available(),
            "promptify_available": self.promptify_client is not None,
            "openai_available": self.openai_client is not None,
            "model": self.model,
            "temperature": self.temperature,
            "total_prompts": len(self.prompt_manager.list_prompts()),
            "available_prompts": self.prompt_manager.list_prompts(),
            "templates_dir": str(self.prompt_manager.templates_dir),
            "usage_stats": self.prompt_manager.get_usage_stats()
        }
