"""Strategic Enhancement Agent for the MLE-STAR Meta-Learning Framework."""

import json
from typing import Any, Dict, Optional

from google.adk import agents
from google.adk.agents import callback_context as callback_context_module
from google.adk.models import llm_response as llm_response_module
from google.genai import types

from machine_learning_engineering.shared_libraries import config, common_util
from machine_learning_engineering.sub_agents.enhancer import prompt


def get_enhancer_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Dynamically builds the instruction for the Enhancer agent."""
    last_run_id = len(context.state.get("run_history_summary", [])) - 1
    last_run_summary = context.state.get("run_history_summary", [])[last_run_id]
    
    return prompt.ENHANCER_AGENT_INSTR.format(
        last_run_id=last_run_id,
        prev_run_id=max(last_run_id - 1, 0),
        next_run_id=last_run_id + 1,
        last_run_final_state=json.dumps(context.state.get("last_run_final_state", {}), indent=2),
        run_history_summary=json.dumps(context.state.get("run_history_summary", []), indent=2),
        last_run_score=last_run_summary.get("best_score"),
        best_score_so_far=context.state.get("best_score_so_far"),
        last_run_time=last_run_summary.get("duration_seconds")
    )


def parse_enhancer_output(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
) -> Optional[llm_response_module.LlmResponse]:
    """Parses, validates, and stores the Enhancer's strategic JSON output."""
    response_text = common_util.get_text_from_response(llm_response)
    try:
        # Clean the response to extract only the JSON block
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            raise json.JSONDecodeError("No JSON object found.", response_text, 0)
        
        json_str = response_text[json_start:json_end]
        enhancer_output = json.loads(json_str)

        # Basic schema validation
        if "strategic_summary" not in enhancer_output or "config_overrides" not in enhancer_output:
            raise ValueError("Enhancer output missing required keys.")
            
        callback_context.state["enhancer_output"] = enhancer_output

    except (json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: Failed to parse Enhancer output. Error: {e}")
        # Fallback to a safe, non-changing strategy
        callback_context.state["enhancer_output"] = {
            "strategic_summary": "Error parsing previous output. Retrying with default configuration.",
            "config_overrides": {},
            "directives": []
        }
    return None


enhancer_agent = agents.Agent(
    model=config.CONFIG.agent_model,
    name="enhancer_agent",
    description="Analyzes past runs to generate a new strategy.",
    instruction=get_enhancer_instruction,
    after_model_callback=parse_enhancer_output,
    generate_content_config=types.GenerateContentConfig(temperature=0.7),
)


