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
    run_history_summary = context.state.get("run_history_summary")
    if not run_history_summary:
        history_json = context.state.get("run_history_summary_json")
        if history_json:
            try:
                run_history_summary = json.loads(history_json)
            except json.JSONDecodeError:
                run_history_summary = []
        else:
            run_history_summary = []

    last_run_id = len(run_history_summary) - 1
    last_run_summary = run_history_summary[last_run_id] if last_run_id >= 0 else {}

    last_run_final_state = context.state.get("last_run_final_state")
    if not last_run_final_state:
        final_state_json = context.state.get("last_run_final_state_json")
        if final_state_json:
            try:
                last_run_final_state = json.loads(final_state_json)
            except json.JSONDecodeError:
                last_run_final_state = {}
        else:
            last_run_final_state = {}

    return prompt.ENHANCER_AGENT_INSTR.format(
        last_run_id=last_run_id,
        prev_run_id=max(last_run_id - 1, 0),
        next_run_id=last_run_id + 1,
        last_run_final_state=json.dumps(last_run_final_state, indent=2),
        run_history_summary=json.dumps(run_history_summary, indent=2),
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
        required_keys = ["strategic_summary", "config_overrides", "strategic_goals"]
        if not all(key in enhancer_output for key in required_keys):
            raise ValueError(f"Enhancer output missing required keys: {required_keys}")

        # Validate and filter strategic goals by target_agent_phase
        VALID_PHASES = {"refinement", "ensemble", "submission"}
        strategic_goals = enhancer_output.get("strategic_goals", [])
        validated_goals = []

        for goal in strategic_goals:
            phase = goal.get("target_agent_phase", "")

            # Map deprecated "modelling" to "refinement" with warning
            if phase == "modelling":
                print(f"WARNING: Deprecated phase 'modelling' mapped to 'refinement' for goal: {goal.get('focus')}")
                goal["target_agent_phase"] = "refinement"
                validated_goals.append(goal)
            elif phase in VALID_PHASES:
                validated_goals.append(goal)
            else:
                print(f"ERROR: Invalid target_agent_phase '{phase}' - goal ignored: {goal.get('focus')}")

        enhancer_output["strategic_goals"] = validated_goals
        callback_context.state["enhancer_output"] = enhancer_output

    except (json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: Failed to parse Enhancer output. Error: {e}")
        # Fallback to a safe, non-changing strategy
        callback_context.state["enhancer_output"] = {
            "strategic_summary": "Error parsing previous output. Retrying with default configuration.",
            "config_overrides": {},
            "strategic_goals": []
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

