"""Refinement agent v3.0 for MLE-STAR, featuring a Mission-Oriented workflow."""

import json
from typing import Optional, List, Dict, Any
import os
import shutil

from google.adk import agents
from google.adk.agents import callback_context as callback_context_module
from google.adk.models import llm_response as llm_response_module
from google.adk.models import llm_request as llm_request_module
from google.genai import types

from machine_learning_engineering.sub_agents.refinement import prompt
from machine_learning_engineering.shared_libraries import debug_util, common_util, config, code_util


# --- Instruction Provider Functions ---

def get_ablation_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the ablation agent instruction."""
    task_id = context.agent_name.split("_")[-1]
    step = context.state.get(f"refine_step_{task_id}", 0)
    code = context.state.get(f"train_code_{step}_{task_id}", "")
    prev_ablations = context.state.get(f"prev_ablations_{task_id}", [])

    prev_ablations_str = ""
    for i, ablation_result in enumerate(prev_ablations):
        prev_ablations_str += f"## Previous ablation study result {i+1}\n"
        prev_ablations_str += f"{ablation_result}\n\n"

    if prev_ablations_str:
        return prompt.ABLATION_SEQ_INSTR.format(code=code, prev_ablations=prev_ablations_str)
    return prompt.ABLATION_INSTR.format(code=code)


def get_ablation_summary_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the ablation summary agent instruction."""
    task_id = context.agent_name.split("_")[-1]
    step = context.state.get(f"refine_step_{task_id}", 0)
    code = context.state.get(f"ablation_code_{step}_{task_id}", "")
    result_dict = context.state.get(f"ablation_code_exec_result_{step}_{task_id}", {})
    return prompt.SUMMARIZE_ABLATION_INSTR.format(
        code=code,
        result=result_dict.get("ablation_result", ""),
    )


def get_plan_generation_instruction(context: callback_context_module.ReadonlyContext) -> str:
    """Gets the instruction for the new Planner agent."""
    task_id = context.agent_name.split("_")[-1]
    step = context.state.get(f"refine_step_{task_id}", 0)
    code = context.state.get(f"train_code_{step}_{task_id}", "")
    ablation_summary = context.state.get(f"ablation_summary_{step}_{task_id}", "")

    # --- FIX: Dynamically build strategic guidance string ---
    enhancer_output = context.state.get("enhancer_output", {})
    strategic_goals = enhancer_output.get("strategic_goals", [])

    refinement_goals = [
        f"- Focus Area: {g['focus']}. Rationale: {g['rationale']}"
        for g in strategic_goals if g.get("target_agent_phase") == "refinement"
    ]

    enhancer_goals_str = "\n".join(refinement_goals)
    if not enhancer_goals_str:
        enhancer_goals_str = "No specific strategic goals provided. Use the ablation summary to find the best area for improvement."
    # --- END FIX ---
    
    return prompt.PLAN_GENERATION_INSTR.format(
        enhancer_goals=enhancer_goals_str,
        ablation_summary=ablation_summary,
        code=code,
    )


def get_plan_step_implement_instruction(context: callback_context_module.ReadonlyContext) -> str:
    """Gets instruction for implementing a single step of a plan."""
    task_id = context.agent_name.split("_")[-1]
    plan_step_description = context.state.get(f"current_plan_step_description_{task_id}", "")
    code_block_to_refine = context.state.get(f"current_code_block_to_refine_{task_id}", "")

    return prompt.IMPLEMENT_PLAN_STEP_INSTR.format(
        code_block=code_block_to_refine,
        plan_step_description=plan_step_description,
    )


# --- Callbacks for State Management ---

def parse_and_store_plan(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
) -> Optional[llm_response_module.LlmResponse]:
    """Parses the Planner's JSON output and stores it in the state."""
    response_text = common_util.get_text_from_response(llm_response)
    task_id = callback_context.agent_name.split("_")[-1]
    try:
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        json_str = response_text[json_start:json_end]
        plan: List[Dict[str, Any]] = json.loads(json_str)
        if not isinstance(plan, list) or not all("plan_step_description" in p for p in plan):
            raise ValueError("Plan is not a valid list of steps.")
        callback_context.state[f"refinement_plan_{task_id}"] = plan
        callback_context.state[f"plan_step_count_{task_id}"] = len(plan)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: Could not parse refinement plan. Error: {e}")
        callback_context.state[f"refinement_plan_{task_id}"] = []
        callback_context.state[f"plan_step_count_{task_id}"] = 0
    return None


def store_ablation_summary(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
) -> Optional[llm_response_module.LlmResponse]:
    """Stores the ablation summary generated by the LLM."""
    task_id = callback_context.agent_name.split("_")[-1]
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    summary = common_util.get_text_from_response(llm_response)
    callback_context.state[f"ablation_summary_{step}_{task_id}"] = summary
    return None


def load_plan_step_for_execution(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest
) -> Optional[llm_response_module.LlmResponse]:
    """Loads the current step of the plan into the context for the implementer agent."""
    task_id = callback_context.agent_name.split("_")[-1]
    plan = callback_context.state.get(f"refinement_plan_{task_id}", [])
    step_idx = callback_context.state.get(f"plan_execution_step_{task_id}", 0)

    if step_idx >= len(plan):
        return llm_response_module.LlmResponse() # Skip if plan is finished

    current_step = plan[step_idx]
    callback_context.state[f"current_plan_step_description_{task_id}"] = current_step.get("plan_step_description", "")
    callback_context.state[f"current_code_block_to_refine_{task_id}"] = current_step.get("code_block_to_refine", "")

    # Help debug_util build the right suffix for eval
    callback_context.state[f"inner_iter_{task_id}"] = step_idx
    return None


def update_plan_execution_step(
    callback_context: callback_context_module.CallbackContext
) -> Optional[types.Content]:
    """Increments the counter for which plan step to execute next."""
    task_id = callback_context.agent_name.split("_")[-1]
    step_idx = callback_context.state.get(f"plan_execution_step_{task_id}", 0)
    callback_context.state[f"plan_execution_step_{task_id}"] = step_idx + 1
    return None


def ensure_plan_execution_step_initialized(
    callback_context: callback_context_module.CallbackContext,
) -> Optional[types.Content]:
    """Ensures the plan execution step counter exists before the loop runs."""
    task_id = callback_context.agent_name.split("_")[-1]
    key = f"plan_execution_step_{task_id}"
    if callback_context.state.get(key) is None:
        callback_context.state[key] = 0
    return None


def init_outer_loop_states(
    callback_context: callback_context_module.CallbackContext
) -> Optional[types.Content]:
    """Initializes outer loop states for mission-oriented refinement."""
    task_id = callback_context.agent_name.split("_")[-1]
    callback_context.state[f"refine_step_{task_id}"] = 0
    callback_context.state[f"prev_ablations_{task_id}"] = []
    callback_context.state[f"prev_code_blocks_{task_id}"] = []
    callback_context.state[f"plan_execution_step_{task_id}"] = 0
    return None


def update_outer_loop_states(
    callback_context: callback_context_module.CallbackContext,
) -> Optional[types.Content]:
    """Advances outer loop and records ablation/code history."""
    task_id = callback_context.agent_name.split("_")[-1]
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    workspace_dir = callback_context.state.get("workspace_dir", "")
    lower = callback_context.state.get("lower", True)
    inner_loop_round = callback_context.state.get("inner_loop_round", 2)
    
    # --- FIX: Correct the run CWD path construction. ---
    run_cwd = os.path.join(workspace_dir, task_id)
    # --- END FIX ---

    prev_solution = callback_context.state.get(
        f"train_code_{step}_{task_id}", ""
    )
    prev_exec_result = callback_context.state.get(
        f"train_code_exec_result_{step}_{task_id}", {}
    )
    improvements: List[float] = []
    for inner_iter in range(inner_loop_round):
        exec_result = callback_context.state.get(
            f"train_code_improve_exec_result_{inner_iter}_{step}_{task_id}", {}
        )
        if lower:
            improvement = prev_exec_result.get("score", float("inf")) - exec_result.get("score", float("inf"))
        else:
            improvement = exec_result.get("score", float("-inf")) - prev_exec_result.get("score", float("-inf"))
        improvements.append(improvement)
    
    best_improvement = max(improvements) if improvements else 0.0
    best_idx = improvements.index(best_improvement) if improvements and best_improvement > 0 else 0

    # --- FIX: Make the output filename consistent with the state key. ---
    output_filepath = os.path.join(run_cwd, f"train{step+1}_{task_id}.py")
    # --- END FIX ---

    if best_improvement <= 0.0:
        best_solution = prev_solution
        best_exec_result = prev_exec_result
    else:
        best_solution = callback_context.state.get(
            f"train_code_improve_{best_idx}_{step}_{task_id}", ""
        )
        best_exec_result = callback_context.state.get(
            f"train_code_improve_exec_result_{best_idx}_{step}_{task_id}", {}
        )

    callback_context.state[f"train_code_{step+1}_{task_id}"] = best_solution
    callback_context.state[f"train_code_exec_result_{step+1}_{task_id}"] = best_exec_result
    
    os.makedirs(run_cwd, exist_ok=True) # Ensure the directory exists before writing
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(best_solution)

    ablation_results = callback_context.state.get(
        f"ablation_summary_{step}_{task_id}", ""
    )
    code_block = callback_context.state.get(
        f"refine_code_block_{step}_{task_id}", ""
    )
    callback_context.state[f"prev_ablations_{task_id}"].append(ablation_results)
    callback_context.state[f"prev_code_blocks_{task_id}"].append(code_block)
    callback_context.state[f"refine_step_{task_id}"] += 1
    return None


# --- Agent Definitions ---

use_data_leakage_checker = config.CONFIG.use_data_leakage_checker
refinement_parallel_sub_agents: List[agents.BaseAgent] = []
for k in range(config.CONFIG.num_solutions):
    task_id = str(k + 1)

    # INTEL STAGE: ablation with debug loop + summarization
    ablation_agent = debug_util.get_run_and_debug_agent(
        prefix="ablation",
        suffix=task_id,
        agent_description="Perform ablation studies to identify sensitive components.",
        instruction_func=get_ablation_agent_instruction,
        before_model_callback=None,
    )
    
    ablation_summary_agent = agents.Agent(
        model=config.CONFIG.agent_model,
        name=f"ablation_summary_agent_{task_id}",
        description="Summarize the ablation study results.",
        after_model_callback=store_ablation_summary,
        instruction=get_ablation_summary_agent_instruction,
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
        include_contents="none",
    )

    # PLAN STAGE: planner
    plan_generation_agent = agents.Agent(
        name=f"plan_generation_agent_{task_id}",
        description="Synthesizes intelligence to create a multi-step improvement plan.",
        model=config.CONFIG.agent_model,
        instruction=get_plan_generation_instruction,
        after_model_callback=parse_and_store_plan,
        generate_content_config=types.GenerateContentConfig(temperature=1.0),
        include_contents="none",
    )

    # EXECUTE STAGE
    plan_step_implement_agent = debug_util.get_run_and_debug_agent(
        prefix="plan_implement",
        suffix=task_id,
        agent_description="Implements a single step of the refinement plan.",
        instruction_func=get_plan_step_implement_instruction,
        before_model_callback=load_plan_step_for_execution,
    )

    plan_execution_loop_agent = agents.LoopAgent(
        name=f"plan_execution_loop_agent_{task_id}",
        description="Executes each step of the generated refinement plan.",
        sub_agents=[plan_step_implement_agent],
        before_agent_callback=ensure_plan_execution_step_initialized,
        after_agent_callback=update_plan_execution_step,
        max_iterations=config.CONFIG.inner_loop_round,
    )

    # One full refinement iteration
    ablation_and_refine_agent = agents.SequentialAgent(
        name=f"ablation_and_refine_agent_{task_id}",
        description="Analyzes, plans, and executes a full refinement cycle.",
        sub_agents=[
            ablation_agent,
            ablation_summary_agent,
            plan_generation_agent,
            plan_execution_loop_agent,
        ],
        after_agent_callback=update_outer_loop_states,
    )

    # Outer loop
    ablation_and_refine_loop_agent = agents.LoopAgent(
        name=f"ablation_and_refine_loop_agent_{task_id}",
        description="Iteratively refines a solution over multiple cycles.",
        sub_agents=[ablation_and_refine_agent],
        before_agent_callback=init_outer_loop_states,
        max_iterations=config.CONFIG.outer_loop_round,
    )
    refinement_parallel_sub_agents.append(ablation_and_refine_loop_agent)

# The final top-level refinement agent
refinement_agent = agents.ParallelAgent(
    name="refinement_agent",
    description="Refine each solution in parallel using a mission-oriented approach.",
    sub_agents=refinement_parallel_sub_agents,
)
