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
    """Gets the instruction for the Planner agent.
    
    Conditionally selects between:
    - BASELINE prompt (Run 0): Simple, ablation-driven, avoids expensive operations
    - ENHANCED prompt (Run 1+): Strategic, enforces ALL enhancer goals
    """
    task_id = context.agent_name.split("_")[-1]
    step = context.state.get(f"refine_step_{task_id}", 0)
    code = context.state.get(f"train_code_{step}_{task_id}", "")
    ablation_summary = context.state.get(f"ablation_summary_{step}_{task_id}", "")
    num_steps_required = context.state.get("inner_loop_round", 1)
    run_id = context.state.get("run_id", 0)

    # Check if we have enhancer strategic guidance
    enhancer_output = context.state.get("enhancer_output", {})
    strategic_goals = enhancer_output.get("strategic_goals", [])
    
    # Filter goals for refinement phase only
    refinement_goals = [
        g for g in strategic_goals if g.get("target_agent_phase") == "refinement"
    ]
    
    # Decision: Use enhanced prompt if we have refinement goals from enhancer
    has_strategic_guidance = run_id > 0 and len(refinement_goals) > 0
    
    if has_strategic_guidance:
        # RUN 1+ WITH ENHANCER GUIDANCE - Use strategic enhanced prompt
        # Format goals with priority order
        sorted_goals = sorted(refinement_goals, key=lambda x: x.get('priority', 99))
        
        # DEBUG: Print what goals we're working with
        print(f"[Task {task_id}] DEBUG: Refinement goals received: {len(refinement_goals)} goals")
        for i, g in enumerate(sorted_goals):
            print(f"  [{i}] Priority {g.get('priority')}: {g.get('focus')} (target: {g.get('target_agent_phase')})")
        
        enhancer_goals_str = "\n".join([
            f"Priority {g.get('priority', i+1)}: {g.get('focus', 'Unknown')} - {g.get('rationale', 'No rationale provided')}"
            for i, g in enumerate(sorted_goals)
        ])
        
        # DEBUG: Show what string is being passed to LLM
        print(f"[Task {task_id}] DEBUG: Strategic goals string being sent to planner:")
        print(f"---\n{enhancer_goals_str}\n---")
        
        # Smart note: Handle mismatch between number of goals and steps
        num_goals = len(refinement_goals)
        if num_goals > num_steps_required:
            enhancer_goals_str += f"\n\nNote: You have {num_goals} goals but only {num_steps_required} steps. Prioritize the highest priority goals."
            print(f"[Task {task_id}] Using ENHANCED planning: {num_goals} goals > {num_steps_required} steps (will prioritize)")
        elif num_goals < num_steps_required:
            enhancer_goals_str += f"\n\nNote: You have {num_goals} mandatory goals and {num_steps_required} steps. Implement all goals first, then use remaining steps for ablation-driven improvements."
            print(f"[Task {task_id}] Using ENHANCED planning: {num_goals} goals < {num_steps_required} steps (will supplement)")
        else:
            print(f"[Task {task_id}] Using ENHANCED planning: {num_goals} goals = {num_steps_required} steps (perfect match)")
        
        return prompt.PLAN_GENERATION_ENHANCED_INSTR.format(
            enhancer_goals=enhancer_goals_str,
            ablation_summary=ablation_summary,
            code=code,
            num_steps_required=num_steps_required,
        )
    else:
        # RUN 0 WITHOUT ENHANCER GUIDANCE - Use baseline prompt (like original MLE-STAR)
        if run_id > 0:
            # Edge case: Run 1+ but no refinement goals (shouldn't happen but handle gracefully)
            print(f"[Task {task_id}] WARNING: Run {run_id} but no refinement goals found. Falling back to baseline planning.")
        else:
            print(f"[Task {task_id}] Using BASELINE planning prompt (ablation-driven, Run 0)")
        
        return prompt.PLAN_GENERATION_BASELINE_INSTR.format(
            ablation_summary=ablation_summary,
            code=code,
            num_steps_required=num_steps_required,
        )


def get_plan_step_implement_instruction(context: callback_context_module.ReadonlyContext) -> str:
    """Gets instruction for implementing a single step of a plan."""
    task_id = context.agent_name.split("_")[-1]
    plan_step_description = context.state.get(f"current_plan_step_description_{task_id}", "")
    code_block_to_refine = context.state.get(f"current_code_block_to_refine_{task_id}", "")
    feature_engineering_function = context.state.get(f"current_feature_engineering_function_{task_id}", "")
    
    # Check if this is a feature engineering step
    if feature_engineering_function:
        # Provide the entire script as context for feature engineering
        refine_step = context.state.get(f"refine_step_{task_id}", 0)
        full_code = context.state.get(f"train_code_{refine_step}_{task_id}", "")
        return prompt.IMPLEMENT_FEATURE_ENGINEERING_INSTR.format(
            full_code=full_code,
            feature_engineering_function=feature_engineering_function,
            plan_step_description=plan_step_description,
        )
    else:
        # Standard implementation for other steps
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
        
        # FIX: Add validation to ensure planner provides a valid code block to prevent infinite debug loops
        for i, step in enumerate(plan):
            # A valid step must have ONE of the two required keys, but not both.
            has_code_block = step.get("code_block_to_refine") and step["code_block_to_refine"].strip()
            has_feature_func = step.get("feature_engineering_function") and step["feature_engineering_function"].strip()
            
            if not (has_code_block or has_feature_func):
                raise ValueError(f"Plan step {i} is missing a required key ('code_block_to_refine' or 'feature_engineering_function').")
            if has_code_block and has_feature_func:
                raise ValueError(f"Plan step {i} has both 'code_block_to_refine' and 'feature_engineering_function'. Only one is allowed.")

        callback_context.state[f"refinement_plan_{task_id}"] = plan
        callback_context.state[f"plan_step_count_{task_id}"] = len(plan)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: Could not parse refinement plan. Error: {e}")
        callback_context.state[f"refinement_plan_{task_id}"] = []
        callback_context.state[f"plan_step_count_{task_id}"] = 0
    return None


def load_plan_step_for_execution(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest
) -> Optional[llm_response_module.LlmResponse]:
    """Loads the current step of the plan into the context for the implementer agent."""
    task_id = callback_context.agent_name.split("_")[-1]
    plan = callback_context.state.get(f"refinement_plan_{task_id}", [])
    step_idx = callback_context.state.get(f"plan_execution_step_{task_id}", 0)

    print(f"\n[LOOP DEBUG] Task {task_id}: load_plan_step_for_execution called")
    print(f"[LOOP DEBUG]   step_idx={step_idx}, len(plan)={len(plan)}")
    print(f"[LOOP DEBUG]   Agent name: {callback_context.agent_name}")

    # Check if previous step failed - log it but continue trying remaining steps
    # This ensures all strategic goals get attempted even if some fail
    if step_idx > 0:
        refine_step = callback_context.state.get(f"refine_step_{task_id}", 0)
        prev_result = callback_context.state.get(
            f"train_code_improve_exec_result_{step_idx-1}_{refine_step}_{task_id}",
            {}
        )
        if not prev_result or prev_result.get("returncode", -1) != 0:
            print(f"WARNING: Plan step {step_idx-1} for task {task_id} failed - continuing to next step...")
            # Continue to next step instead of halting - this ensures all strategic goals are attempted

    # PHASE 4: Self-terminating condition - stop when plan is complete
    if step_idx >= len(plan):
        print(f"INFO: Refinement plan for task {task_id} complete ({len(plan)} steps executed). Halting loop.")
        return llm_response_module.LlmResponse() # Skip if plan is finished

    current_step = plan[step_idx]
    refine_step = callback_context.state.get(f"refine_step_{task_id}", 0)
    callback_context.state[f"current_plan_step_description_{task_id}"] = current_step.get("plan_step_description", "")
    
    # Handle the two types of plan steps
    if "feature_engineering_function" in current_step:
        # New robust path for feature engineering
        callback_context.state[f"current_feature_engineering_function_{task_id}"] = current_step.get("feature_engineering_function", "")
        # Clear the old key to avoid confusion
        callback_context.state[f"current_code_block_to_refine_{task_id}"] = ""
    else:
        # Standard path for other refinements
        callback_context.state[f"current_code_block_to_refine_{task_id}"] = current_step.get("code_block_to_refine", "")
        # Clear the new key
        callback_context.state[f"current_feature_engineering_function_{task_id}"] = ""

    # FIX: Store the code block with the correct key that debug_util expects
    callback_context.state[f"refine_code_block_{refine_step}_{task_id}"] = current_step.get("code_block_to_refine", "")

    # Help debug_util build the right suffix for eval
    callback_context.state[f"inner_iter_{task_id}"] = step_idx
    print(f"[Plan Execution] Task {task_id} executing step index {step_idx}: {callback_context.state[f'current_plan_step_description_{task_id}']}")
    return None


def update_plan_execution_step(
    callback_context: callback_context_module.CallbackContext
) -> Optional[types.Content]:
    """Increments the counter for which plan step to execute next."""
    task_id = callback_context.agent_name.split("_")[-1]
    step_idx = callback_context.state.get(f"plan_execution_step_{task_id}", 0)
    refine_step = callback_context.state.get(f"refine_step_{task_id}", 0)

    print(f"\n[LOOP DEBUG] update_plan_execution_step called")
    print(f"[LOOP DEBUG]   step_idx BEFORE increment: {step_idx}")

    # Get the result of the step that just ran by reading from the state
    result = callback_context.state.get(
        f"train_code_improve_exec_result_{step_idx}_{refine_step}_{task_id}", {}
    )

    # Always advance to next step to ensure all strategic goals are attempted
    callback_context.state[f"plan_execution_step_{task_id}"] = step_idx + 1
    
    if not result or result.get("returncode") != 0:
        print(f"INFO: Step {step_idx} for task {task_id} failed, but advancing to next step to try remaining strategies.")
    else:
        print(f"INFO: Step {step_idx} for task {task_id} succeeded. Advancing to next step.")
    
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
    
    def make_ablation_summary_callback(tid):
        def callback(callback_context, llm_response):
            step = callback_context.state.get(f"refine_step_{tid}", 0)
            summary = common_util.get_text_from_response(llm_response)
            callback_context.state[f"ablation_summary_{step}_{tid}"] = summary
            return None
        return callback

    ablation_summary_agent = agents.Agent(
        model=config.CONFIG.agent_model,
        name=f"ablation_summary_agent_{task_id}",
        description="Summarize the ablation study results.",
        after_model_callback=make_ablation_summary_callback(task_id),
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
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
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

# --- NEW: Simplified Single-Step Refinement for Run 1+ ---

def get_single_improvement_instruction(context: callback_context_module.ReadonlyContext) -> str:
    """Gets instruction for implementing the enhancer's single improvement suggestion."""
    task_id = context.agent_name.split("_")[-1]

    # Get the improvement suggestion from enhancer
    improvement = context.state.get("improvement_to_implement", {})
    if not improvement:
        return "ERROR: No improvement suggestion found from enhancer"

    focus = improvement.get("focus", "")
    description = improvement.get("description", "")
    rationale = improvement.get("rationale", "")

    # Get current champion code
    current_code = context.state.get(f"train_code_0_{task_id}", "")
    current_score = context.state.get(f"train_code_exec_result_0_{task_id}", {}).get("score", "N/A")

    return prompt.IMPLEMENT_SINGLE_IMPROVEMENT_INSTR.format(
        current_code=current_code,
        current_score=current_score,
        improvement_focus=focus,
        improvement_description=description,
        improvement_rationale=rationale,
    )


# Build simplified single-step agent for Run 1+
single_improvement_agent_task_1 = debug_util.get_run_and_debug_agent(
    prefix="single_improvement",
    suffix="1",
    agent_description="Implements the enhancer's single improvement suggestion",
    instruction_func=get_single_improvement_instruction,
    before_model_callback=None,
)

# Wrap the simple agent to choose between Run 0 (complex) and Run 1+ (simple)
def choose_refinement_mode(
    callback_context: callback_context_module.CallbackContext
) -> Optional[types.Content]:
    """Routes to simple single-step mode for Run 1+, full pipeline for Run 0."""
    is_refinement_run = callback_context.state.get("is_refinement_run", False)

    if is_refinement_run:
        print("[Refinement Mode] Using simplified single-improvement mode for Run 1+")
        # Will use single_improvement_agent
    else:
        print("[Discovery Mode] Using full ablation-plan-execute pipeline for Run 0")
        # Will use the full ablation_and_refine_loop_agent

    return None


# --- Conditional Task Execution for Refinement Mode ---

def skip_if_refinement_mode_and_not_task_1(
    callback_context: callback_context_module.CallbackContext
) -> Optional[types.Content]:
    """Skips tasks 2+ in refinement-only mode (linear refinement uses only Task 1)."""
    is_refinement_run = callback_context.state.get("is_refinement_run", False)
    agent_name = callback_context.agent_name
    
    # Extract task_id from agent name (e.g., "task_2_wrapper" -> "2")
    # The agent name will be "task_X_wrapper" for wrapped agents
    if "_wrapper" in agent_name:
        task_id = agent_name.split("_")[1]
    else:
        # Fallback: try to extract from sub-agent name
        return None
    
    if is_refinement_run and task_id != "1":
        print(f"[Refinement Mode] Skipping Task {task_id} (linear refinement uses only Task 1)")
        return types.Content(parts=[types.Part(text="skipped")], role="model")
    return None

# Task 1 needs conditional execution: complex for Run 0, simple for Run 1+
# Simplest approach: SequentialAgent with both, each has a skip callback

def skip_if_not_refinement(callback_context: callback_context_module.CallbackContext) -> Optional[types.Content]:
    """Skip if NOT in refinement mode."""
    if not callback_context.state.get("is_refinement_run", False):
        return types.Content(parts=[types.Part(text="skipped")], role="model")
    return None

def skip_if_refinement(callback_context: callback_context_module.CallbackContext) -> Optional[types.Content]:
    """Skip if in refinement mode."""
    if callback_context.state.get("is_refinement_run", False):
        return types.Content(parts=[types.Part(text="skipped")], role="model")
    return None

# Wrap Task 1 complex agent to skip in refinement mode
task_1_complex = agents.SequentialAgent(
    name="task_1_complex",
    sub_agents=[refinement_parallel_sub_agents[0]],
    before_agent_callback=skip_if_refinement,
)

# Wrap Task 1 simple agent to skip in discovery mode
task_1_simple = agents.SequentialAgent(
    name="task_1_simple",
    sub_agents=[single_improvement_agent_task_1],
    before_agent_callback=skip_if_not_refinement,
)

# Task 1 router runs both sequentially (one skips, one runs)
task_1_router = agents.SequentialAgent(
    name="task_1_router",
    sub_agents=[task_1_complex, task_1_simple],
)

# Build wrapped agent list
wrapped_refinement_sub_agents: List[agents.BaseAgent] = [task_1_router]

# Add Task 2+ with skip wrappers
for i, task_agent in enumerate(refinement_parallel_sub_agents):
    task_id = str(i + 1)
    if task_id == "1":
        continue  # Already added
    else:
        # Task 2+: wrapped with conditional skip for refinement runs
        wrapper = agents.SequentialAgent(
            name=f"task_{task_id}_wrapper",
            sub_agents=[task_agent],
            description=f"Task {task_id} wrapper (skipped in refinement mode)",
            before_agent_callback=skip_if_refinement_mode_and_not_task_1,
        )
        wrapped_refinement_sub_agents.append(wrapper)

# The final top-level refinement agent
# Uses ParallelAgent to run all tasks concurrently in Run 0 (original behavior)
# In Run 0: executes all tasks in parallel (Task 1 and Task 2 simultaneously)
# In Run 1+: only Task 1 executes (Task 2+ are skipped via wrapper before they start)
refinement_agent = agents.ParallelAgent(
    name="refinement_agent",
    description="Refine solutions in parallel (all tasks for Run 0, only Task 1 for Run 1+).",
    sub_agents=wrapped_refinement_sub_agents,
)
