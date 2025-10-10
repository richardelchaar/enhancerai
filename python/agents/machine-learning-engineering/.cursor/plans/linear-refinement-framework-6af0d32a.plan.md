<!-- 6af0d32a-5e26-4c55-8004-0a460fe8fb4c c506f135-4c89-4d7f-9af1-d9d6fc322c24 -->
# Linear Refinement Framework Implementation Plan

## Overview

Transform MLE-STAR to support two execution modes:

- **Run 0 (Discovery Mode)**: Full pipeline unchanged - searches for models, creates initial solutions, produces the first "Champion"
- **Run 1+ (Refinement Mode)**: Loads the Champion from previous run, skips initialization and ensemble, executes all Enhancer suggestions via a comprehensive multi-step refinement plan

## Phase 1: Configuration Changes

### File: `machine_learning_engineering/shared_libraries/config.py`

**Change 1.1**: Increase `inner_loop_round` to enable multi-step execution

```python
# Line ~30
inner_loop_round: int = 3  # Changed from 1 to 3 to execute full multi-step plans
```

**Rationale**: This allows the refinement loop to execute up to 3 steps of a generated plan, ensuring all Enhancer suggestions can be attempted in a single run.

## Phase 2: Orchestrator State Management

### File: `run_meta.py`

**Change 2.1**: Add champion detection and state seeding logic

After the line where you set up the workspace directory for each run (around line ~50-80), add:

```python
import json
import dataclasses
from machine_learning_engineering.shared_libraries import config

# Determine if this is a refinement run
is_refinement_run = run_id > 0
initial_state = {}

if is_refinement_run:
    print(f"[Run {run_id}] Initializing Linear Refinement Mode")
    
    # Ensure the run directory exists
    run_dir = os.path.join(workspace_dir, task_name, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Load run history to find the champion
    run_history_path = os.path.join(workspace_dir, task_name, "run_history.json")
    with open(run_history_path, "r") as f:
        history = json.load(f)
    
    # Get the best solution from the most recent run
    last_run = history[-1]
    champion_relative_path = last_run["best_solution_path"]
    champion_full_path = os.path.join(workspace_dir, task_name, champion_relative_path)
    
    print(f"[Run {run_id}] Loading champion from: {champion_full_path}")
    
    # Read the champion code
    with open(champion_full_path, "r") as f:
        champion_code = f.read()
    
    # Load config values into state (normally done by initialization agent's prepare_task)
    config_dict = dataclasses.asdict(config.CONFIG)
    initial_state.update(config_dict)
    
    # Read task description
    task_desc_path = os.path.join(workspace_dir, task_name, "task_description.txt")
    if os.path.exists(task_desc_path):
        with open(task_desc_path, "r") as f:
            initial_state["task_description"] = f.read()
    
    # CRITICAL: Seed ALL task IDs with champion code
    # The refinement agent creates parallel agents at import time based on num_solutions
    # So if num_solutions=2, we'll have 2 parallel threads - both need the champion
    num_solutions = config.CONFIG.num_solutions
    for task_id in range(1, num_solutions + 1):
        initial_state[f"train_code_0_{task_id}"] = champion_code
        initial_state[f"train_code_exec_result_0_{task_id}"] = {
            "score": last_run["best_score"],
            "returncode": 0
        }
    
    # Set refinement-only mode flag
    initial_state["is_refinement_run"] = True
```

**Change 2.2**: Pass the initial_state to the agent invocation

Modify the agent invocation line to include the state:

```python
# Change from:
# final_state = run_agent(...)

# To:
final_state = run_agent(..., initial_state=initial_state)
```

## Phase 3: Conditional Agent Pipeline

### File: `machine_learning_engineering/agent.py`

**Change 3.1**: Import the necessary agent types

```python
# After line ~11
from google.adk.agents import ConditionalAgent
```

**Change 3.2**: Create the refinement-only pipeline

After the `save_state` function and before the original `mle_pipeline_agent` definition:

```python
# Original full pipeline for Run 0
full_discovery_pipeline = agents.SequentialAgent(
    name="full_discovery_pipeline",
    sub_agents=[
        initialization_agent_module.initialization_agent,
        refinement_agent_module.refinement_agent,
        ensemble_agent_module.ensemble_agent,
        submission_agent_module.submission_agent,
    ],
    description="Full discovery and refinement pipeline for Run 0",
    after_agent_callback=save_state,
)

# Stripped-down pipeline for Run 1+
refinement_only_pipeline = agents.SequentialAgent(
    name="refinement_only_pipeline",
    sub_agents=[
        # Skip initialization (no model search needed)
        refinement_agent_module.refinement_agent,
        # Skip ensemble (only one solution thread)
        submission_agent_module.submission_agent,
    ],
    description="Focused refinement pipeline for Run 1+",
    after_agent_callback=save_state,
)

# Controller that switches between pipelines
mle_pipeline_agent = ConditionalAgent(
    name="mle_pipeline_agent",
    description="Routes to discovery or refinement pipeline based on run type",
    condition=lambda context: context.state.get("is_refinement_run", False),
    true_agent=refinement_only_pipeline,
    false_agent=full_discovery_pipeline,
)
```

**Change 3.3**: The `root_agent` definition remains unchanged (it references `mle_pipeline_agent`)

## Phase 4: Refinement Agent - Dynamic Planning

### File: `machine_learning_engineering/sub_agents/refinement/agent.py`

**Change 4.1**: Make plan generation instruction dynamic

In the `get_plan_generation_instruction` function (around line ~53):

```python
def get_plan_generation_instruction(context: callback_context_module.ReadonlyContext) -> str:
    """Gets the instruction for the new Planner agent."""
    task_id = context.agent_name.split("_")[-1]
    step = context.state.get(f"refine_step_{task_id}", 0)
    code = context.state.get(f"train_code_{step}_{task_id}", "")
    ablation_summary = context.state.get(f"ablation_summary_{step}_{task_id}", "")
    
    # NEW: Get the required number of steps from config
    num_steps_required = context.state.get("inner_loop_round", 1)

    # ... existing enhancer_goals_str logic ...
    
    return prompt.PLAN_GENERATION_INSTR.format(
        enhancer_goals=enhancer_goals_str,
        ablation_summary=ablation_summary,
        code=code,
        num_steps_required=num_steps_required,  # Pass to prompt
    )
```

**Change 4.2**: Make execution loop self-terminating

In the `load_plan_step_for_execution` function (around line ~128), enhance the termination logic:

```python
def load_plan_step_for_execution(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest
) -> Optional[llm_response_module.LlmResponse]:
    """Loads the current step of the plan into the context for the implementer agent."""
    task_id = callback_context.agent_name.split("_")[-1]
    plan = callback_context.state.get(f"refinement_plan_{task_id}", [])
    step_idx = callback_context.state.get(f"plan_execution_step_{task_id}", 0)

    # Check if previous step failed before loading next step
    if step_idx > 0:
        refine_step = callback_context.state.get(f"refine_step_{task_id}", 0)
        prev_result = callback_context.state.get(
            f"train_code_improve_exec_result_{step_idx-1}_{refine_step}_{task_id}",
            {}
        )
        if not prev_result or prev_result.get("returncode", -1) != 0:
            print(f"WARNING: Plan execution halted for task {task_id} - step {step_idx-1} failed")
            return llm_response_module.LlmResponse()

    # NEW: Self-terminating condition - stop when plan is complete
    if step_idx >= len(plan):
        print(f"INFO: Refinement plan for task {task_id} complete ({len(plan)} steps executed). Halting loop.")
        return llm_response_module.LlmResponse()

    current_step = plan[step_idx]
    # ... rest of function unchanged ...
```

## Phase 5: Refinement Prompt - Forceful and Dynamic

### File: `machine_learning_engineering/sub_agents/refinement/prompt.py`

**Change 5.1**: Update the `PLAN_GENERATION_INSTR` prompt

Around line ~186, modify the "Your Task" section:

```python
# Your Task
1. **Synthesize:** Combine the insights from the Ablation Study Summary with the Strategic Guidance.
2. **Plan:** Your task is to identify and rank the top strategies for improving the code. You **MUST** provide a plan with **exactly {num_steps_required} distinct steps**, ordered from most to least impactful. Each step must target a different aspect of the code. If you cannot identify enough high-impact changes to fill all {num_steps_required} steps, use the remaining steps to propose smaller but still valuable refinements (e.g., regularization tuning, minor feature adjustments, code optimization).
3. **Identify Code:** For each step in your plan, you MUST identify the specific, contiguous block of code from the "Current Code" that needs to be modified.
```

**Rationale**: The `{num_steps_required}` placeholder will be filled with the value from `inner_loop_round`, making the prompt dynamically adapt to the configuration and compelling the agent to create a full, multi-step plan.

## Phase 6: Refinement Prompt - Stronger Enhancer Prioritization

### File: `machine_learning_engineering/sub_agents/refinement/prompt.py`

**Change 6.1**: Make the synthesis instruction more directive

Around line ~169-175, strengthen the guidance:

```python
# Persona
You are a world-class machine learning engineer, tasked with improving a model's performance by creating a multi-step plan for code refinement.

# Context
You have been given the results of an ablation study and a high-level strategic directive from your Research Lead. Your task is to synthesize this information into a concrete, actionable plan to modify the provided code.

**IMPORTANT**: The Strategic Guidance from the Research Lead represents priorities derived from empirical evidence across multiple runs. These strategic goals should be your PRIMARY focus when planning improvements. Use the Ablation Study Summary to inform HOW and WHERE to implement these strategic goals, not to override them.
```

## Testing & Verification Strategy

### Verification for Run 0 (Discovery Mode)

1. Execute a fresh Run 0 with no history file
2. Verify the full pipeline executes: initialization → refinement → ensemble → submission
3. Confirm `run_history.json` is created with the correct `best_solution_path`

### Verification for Run 1 (Refinement Mode)

1. Execute Run 1 after Run 0 completes
2. Verify the orchestrator logs show "Initializing Linear Refinement Mode"
3. Verify initialization and ensemble agents are skipped
4. Verify the refinement agent creates a multi-step plan (should see 3 steps in logs)
5. Verify all 3 steps are executed (check for "Refinement plan complete" message)
6. Verify the final solution builds upon the champion code from Run 0

### Key Log Messages to Monitor

- `[Run 1] Initializing Linear Refinement Mode`
- `[Run 1] Loading champion from: ...`
- `INFO: Refinement plan for task 1 complete (3 steps executed)`
- Plan generation should show exactly 3 steps in the JSON output

## Risk Mitigation

**Risk 1**: ConditionalAgent import may not exist in the ADK version

- **Mitigation**: If import fails, implement a simple wrapper function that checks the state flag and manually calls the appropriate sub-pipeline

**Risk 2**: State seeding may not persist correctly through agent boundaries

- **Mitigation**: Add debug logging at each agent's `before_agent_callback` to verify state values are accessible

**Risk 3**: The ablation agent might fail on champion code if it's already complex

- **Mitigation**: Ensure ablation agent has robust error handling; if ablation fails, the planning agent should still receive the enhancer goals and create a reasonable plan

## Success Criteria

1. Run 0 executes identically to current behavior (no regression)
2. Run 1 skips initialization and ensemble agents (observable in logs)
3. Run 1 loads and refines the champion code from Run 0 (verifiable by comparing code similarity)
4. Refinement agent in Run 1 generates and executes a 3-step plan (logs show all 3 steps)
5. All Enhancer strategic goals from Run 0 are represented in Run 1's refinement plan
6. Final solution from Run 1 has a better or equal score to Run 0's champion

### To-dos

- [ ] Update config.py to set inner_loop_round to 3
- [ ] Add run type detection logic to run_meta.py
- [ ] Implement champion loading and state seeding in run_meta.py
- [ ] Create conditional agent pipeline in agent.py with discovery and refinement modes
- [ ] Update refinement agent.py to pass num_steps_required to prompt
- [ ] Enhance load_plan_step_for_execution with self-termination logic
- [ ] Update PLAN_GENERATION_INSTR to accept and use num_steps_required
- [ ] Strengthen enhancer prioritization language in refinement prompt
- [ ] Test Run 0 to verify no regression in discovery mode
- [ ] Test Run 1 to verify refinement-only mode and multi-step execution