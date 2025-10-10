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

    # The enhancer ALWAYS uses single-improvement mode because it only runs AFTER Run 0, 1, 2...
    # to plan for Run 1, 2, 3... (never before Run 0)
    # Run 0 uses the full multi-step pipeline WITHOUT enhancer input
    prev_run_score = None
    if len(run_history_summary) >= 2:
        prev_run_score = run_history_summary[-2].get("best_score")
    
    current_score = last_run_summary.get("best_score")
    score_delta = (prev_run_score - current_score) if prev_run_score and current_score else 0
    
    return prompt.ENHANCER_SINGLE_IMPROVEMENT_INSTR.format(
        last_run_id=last_run_id,
        next_run_id=last_run_id + 1,
        prev_run_id=max(last_run_id - 1, 0),
        run_history_summary=json.dumps(run_history_summary, indent=2),
        last_run_final_state=json.dumps(last_run_final_state, indent=2),
        last_run_score=current_score,
        prev_run_score=prev_run_score if prev_run_score else "N/A",
        score_delta=score_delta,
        last_run_time=last_run_summary.get("duration_seconds")
    )


def parse_enhancer_output(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
) -> Optional[llm_response_module.LlmResponse]:
    """Parses, validates, and stores the Enhancer's strategic JSON output.
    
    Supports two schemas:
    1. Run 0: Multi-goal mode with strategic_goals array
    2. Run 1+: Single-improvement mode with next_improvement object
    """
    response_text = common_util.get_text_from_response(llm_response)
    try:
        # Clean the response to extract only the JSON block
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            raise json.JSONDecodeError("No JSON object found.", response_text, 0)
        
        json_str = response_text[json_start:json_end]
        enhancer_output = json.loads(json_str)

        # Detect which schema is being used
        has_next_improvement = "next_improvement" in enhancer_output
        has_strategic_goals = "strategic_goals" in enhancer_output
        
        if has_next_improvement:
            # Single-improvement mode (Run 1+)
            required_keys = ["strategic_summary", "next_improvement"]
            if not all(key in enhancer_output for key in required_keys):
                raise ValueError(f"Single-improvement enhancer output missing required keys: {required_keys}")
            
            # Validate next_improvement structure
            improvement = enhancer_output.get("next_improvement", {})
            improvement_keys = ["focus", "description", "rationale"]
            if not all(key in improvement for key in improvement_keys):
                raise ValueError(f"next_improvement missing required keys: {improvement_keys}")
            
            strategy_id = improvement.get('strategy_id', 'N/A')
            print(f"[Enhancer] Single-improvement mode: {improvement.get('focus')}")
            print(f"[Enhancer] Strategy ID: {strategy_id}")
            print(f"[Enhancer] Description: {improvement.get('description')}")
            
        elif has_strategic_goals:
            # Multi-goal mode (Run 0)
            required_keys = ["strategic_summary", "config_overrides", "strategic_goals"]
            if not all(key in enhancer_output for key in required_keys):
                raise ValueError(f"Multi-goal enhancer output missing required keys: {required_keys}")

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
            print(f"[Enhancer] Multi-goal mode: {len(validated_goals)} strategic goals validated")
        else:
            raise ValueError("Enhancer output must contain either 'next_improvement' or 'strategic_goals'")

        callback_context.state["enhancer_output"] = enhancer_output

    except (json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: Failed to parse Enhancer output. Error: {e}")
        # Fallback based on run history length
        run_history = callback_context.state.get("run_history_summary", [])
        if len(run_history) > 1:
            # Run 1+: Fallback to minimal improvement
            callback_context.state["enhancer_output"] = {
                "strategic_summary": "Error parsing previous output. Using minimal improvement strategy.",
                "next_improvement": {
                    "focus": "hyperparameter_tuning",
                    "description": "Perform minor hyperparameter adjustments to improve model performance",
                    "rationale": "Safe fallback strategy with low risk of breaking existing functionality"
                }
            }
        else:
            # Run 0: Fallback to safe multi-goal strategy
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

