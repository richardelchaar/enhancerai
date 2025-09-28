"""Ensemble agent for Machine Learning Engineering."""

from typing import Optional, List, Dict, Any
import json
import os

from google.adk import agents
from google.adk.agents import callback_context as callback_context_module
from google.adk.models import llm_response as llm_response_module
from google.adk.models import llm_request as llm_request_module
from google.genai import types

from machine_learning_engineering.sub_agents.ensemble import prompt
from machine_learning_engineering.shared_libraries import common_util
from machine_learning_engineering.shared_libraries import config
from machine_learning_engineering.shared_libraries import debug_util


def get_ensemble_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the ensemble agent instruction."""
    solution1_code = context.state.get("train_code_1_1", "")
    solution1_score = context.state.get("train_code_exec_result_1_1", {}).get("score", "N/A")
    solution2_code = context.state.get("train_code_1_2", "")
    solution2_score = context.state.get("train_code_exec_result_1_2", {}).get("score", "N/A")

    # --- FIX: Dynamically build strategic guidance string ---
    enhancer_output = context.state.get("enhancer_output", {})
    strategic_goals = enhancer_output.get("strategic_goals", [])

    ensemble_goals = [
        g for g in strategic_goals if g.get("target_agent_phase") == "ensemble"
    ]

    if ensemble_goals:
        # Use the highest priority goal if multiple exist
        primary_goal = sorted(ensemble_goals, key=lambda x: x.get("priority", 99))[0]
        ensemble_focus = primary_goal.get("focus", "default ensembling")
        ensemble_rationale = primary_goal.get("rationale", "No rationale provided.")
    else:
        ensemble_focus = "simple averaging"
        ensemble_rationale = "No specific strategic goal was provided. Start with a simple and robust averaging ensemble."
    # --- END FIX ---
    
    return prompt.ENSEMBLE_PLANNING_INSTR.format(
        solution1_code=solution1_code,
        solution1_score=solution1_score,
        solution2_code=solution2_code,
        solution2_score=solution2_score,
        ensemble_focus=ensemble_focus,
        ensemble_rationale=ensemble_rationale,
    )


def parse_and_store_ensemble_plan(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
) -> Optional[llm_response_module.LlmResponse]:
    """Parses the Planner's JSON output and stores it in the state."""
    response_text = common_util.get_text_from_response(llm_response)
    try:
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        json_str = response_text[json_start:json_end]
        plan: List[Dict[str, Any]] = json.loads(json_str)
        if not isinstance(plan, list) or not all("plan_step_description" in p for p in plan):
            raise ValueError("Plan is not a valid list of steps.")
        callback_context.state["ensemble_plan"] = plan
        callback_context.state["ensemble_plan_step_count"] = len(plan)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: Could not parse ensemble plan. Error: {e}")
        callback_context.state["ensemble_plan"] = []
        callback_context.state["ensemble_plan_step_count"] = 0
    return None


def load_ensemble_plan_step(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest
) -> Optional[llm_response_module.LlmResponse]:
    """Loads the current step of the ensemble plan into the context."""
    plan = callback_context.state.get("ensemble_plan", [])
    step_idx = callback_context.state.get("ensemble_iter", 0)

    if step_idx >= len(plan):
        return llm_response_module.LlmResponse()

    current_step = plan[step_idx]
    
    # In the first iteration, the code to refine is the entire previous solution.
    # In subsequent iterations, it's the output of the previous implementation step.
    if step_idx == 0:
        # For the first step, we might not need a "code to refine" as it builds from scratch.
        # We'll pass the first solution's code as a baseline reference.
        code_to_refine = callback_context.state.get("train_code_1_1", "")
    else:
        code_to_refine = callback_context.state.get(f"ensemble_code_{step_idx - 1}", "")

    callback_context.state["current_ensemble_plan_step"] = current_step.get("plan_step_description", "")
    callback_context.state["current_ensemble_code_to_implement"] = current_step.get("code_block_to_implement", "")
    return None


def get_ensemble_implement_instruction(context: callback_context_module.ReadonlyContext) -> str:
    """Gets instruction for implementing a single step of the ensemble plan."""
    # This function is now simpler as the logic is in the plan itself.
    # The agent just needs to combine the code blocks.
    plan = context.state.get("ensemble_plan", [])
    step_idx = context.state.get("ensemble_iter", 0)
    
    # Concatenate all code blocks up to the current step
    full_code = ""
    for i in range(step_idx + 1):
        if i < len(plan):
            full_code += f"# --- Plan Step {i+1}: {plan[i].get('plan_step_description', '')} ---\n"
            full_code += plan[i].get("code_block_to_implement", "")
            full_code += "\n\n"
            
    return f"""
# Your Task
You are an expert programmer. You have been given a multi-step plan to create an ensemble solution. Your task is to provide the complete, runnable Python script by combining all the code blocks from the plan up to the current step.

# Full Plan
{json.dumps(plan, indent=2)}

# Current Step
{step_idx + 1}

# Instructions
Combine all `code_block_to_implement` values from the beginning of the plan up to and including the current step into a single, cohesive, and runnable Python script. Ensure the final script is complete and syntactically correct.

# Required Output Format
You must provide your response as a single Python code block.
"""


def advance_ensemble_iterator(
    callback_context: callback_context_module.CallbackContext
) -> Optional[llm_response_module.LlmResponse]:
    """Advances the ensemble iteration counter."""
    ensemble_iter = callback_context.state.get("ensemble_iter", 0)
    callback_context.state["ensemble_iter"] = ensemble_iter + 1
    return None


def ensure_ensemble_iterator_initialized(
    callback_context: callback_context_module.CallbackContext,
) -> Optional[types.Content]:
    """Ensures the ensemble iterator exists before executing the loop."""
    if callback_context.state.get("ensemble_iter") is None:
        callback_context.state["ensemble_iter"] = 0
    return None


def create_ensemble_workspace(
    callback_context: callback_context_module.CallbackContext,
) -> Optional[types.Content]:
    """Creates the ensemble workspace directory."""
    workspace_dir = callback_context.state.get("workspace_dir", "")
    ensemble_dir = os.path.join(workspace_dir, "ensemble")
    
    # Create ensemble directory and subdirectories
    os.makedirs(ensemble_dir, exist_ok=True)
    os.makedirs(os.path.join(ensemble_dir, "input"), exist_ok=True)
    os.makedirs(os.path.join(ensemble_dir, "final"), exist_ok=True)
    
    # Copy input data from one of the task directories (use task 1 as source)
    task_1_input = os.path.join(workspace_dir, "1", "input")
    if os.path.exists(task_1_input):
        import shutil
        for item in os.listdir(task_1_input):
            src = os.path.join(task_1_input, item)
            dst = os.path.join(ensemble_dir, "input", item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    
    return None


# --- Agent Definitions ---

# Planner Agent to create the ensemble strategy
ensemble_planner_agent = agents.Agent(
    name="ensemble_planner",
    description="Creates a step-by-step plan to ensemble two solutions.",
    model=config.CONFIG.agent_model,
    instruction=get_ensemble_instruction,
    after_model_callback=parse_and_store_ensemble_plan,
    generate_content_config=types.GenerateContentConfig(temperature=0.0),
    include_contents="none",
)

# Implementer Agent (wrapped in a debug loop)
ensemble_implement_agent = debug_util.get_run_and_debug_agent(
    prefix="ensemble_plan_implement",
    suffix="",
    agent_description="Implements one step of the ensemble plan.",
    instruction_func=get_ensemble_implement_instruction,
    before_model_callback=load_ensemble_plan_step,
)

# Loop to execute all steps of the plan
ensemble_execution_loop = agents.LoopAgent(
    name="ensemble_execution_loop",
    description="Executes each step of the ensemble plan sequentially.",
    sub_agents=[ensemble_implement_agent],
    before_agent_callback=ensure_ensemble_iterator_initialized,
    after_agent_callback=advance_ensemble_iterator,
    max_iterations=config.CONFIG.ensemble_loop_round,
)

# Main Sequential Agent for the Ensemble Phase
ensemble_agent = agents.SequentialAgent(
    name="ensemble_agent",
    description="Generates and executes a plan to create a final ensemble solution.",
    sub_agents=[ensemble_planner_agent, ensemble_execution_loop],
    before_agent_callback=create_ensemble_workspace,
)
