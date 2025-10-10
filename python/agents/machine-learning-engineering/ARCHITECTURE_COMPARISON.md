# Architecture Comparison: Run 0 vs Run 1+

## Visual Flow Comparison

### Run 0: Full Discovery Pipeline
```
┌─────────────────────────────────────────────────────────────────┐
│                         RUN 0 (Discovery)                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Initialization Agent  │
                    │ - Generates N=2       │
                    │   baseline solutions  │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Refinement Agent    │
                    │   (Complex Mode)      │
                    └───────────┬───────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
            Task 1          Task 2          Task N
                │               │               │
                ▼               ▼               ▼
        ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
        │ Ablation Loop │ │ Ablation Loop │ │ Ablation Loop │
        │ (outer_loop)  │ │ (outer_loop)  │ │ (outer_loop)  │
        └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
                │               │               │
                ▼               ▼               ▼
        ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
        │ Plan Generator│ │ Plan Generator│ │ Plan Generator│
        │ - Creates 3-  │ │ - Creates 3-  │ │ - Creates 3-  │
        │   step plan   │ │   step plan   │ │   step plan   │
        └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
                │               │               │
                ▼               ▼               ▼
        ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
        │ Plan Executor │ │ Plan Executor │ │ Plan Executor │
        │ Loop Agent    │ │ Loop Agent    │ │ Loop Agent    │
        │ (inner_loop)  │ │ (inner_loop)  │ │ (inner_loop)  │
        └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
                │               │               │
        ┌───────┼───────┐ ┌─────┼───────┐ ┌───────┼───────┐
        ▼       ▼       ▼ ▼     ▼       ▼ ▼       ▼       ▼
      Step0  Step1  Step2 Step0 Step1 Step2 Step0 Step1 Step2
        │       │       │   │     │       │   │     │       │
        └───────┴───────┴───┴─────┴───────┴───┴─────┴───────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Ensemble Agent      │
                    │ - Combines N solutions│
                    │ - Creates final blend │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Submission Agent     │
                    │ - Generates CSV       │
                    └───────────┬───────────┘
                                │
                                ▼
                        Run 0 Complete
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Enhancer Agent      │
                    │ - Analyzes Run 0      │
                    │ - Outputs ONE         │
                    │   improvement for R1  │
                    └───────────────────────┘
```

### Run 1+: Simplified Refinement Pipeline
```
┌─────────────────────────────────────────────────────────────────┐
│                    RUN 1+ (Simplified)                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │  Load Champion from Run 0     │
                │  Load Improvement from        │
                │  Enhancer                     │
                └───────────────┬───────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Refinement Agent    │
                    │   (Simple Mode)       │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Router Callback     │
                    │ Checks:               │
                    │ is_refinement_run?    │
                    └───────────┬───────────┘
                                │ YES
                                ▼
                    ┌───────────────────────┐
                    │ single_improvement_   │
                    │ agent_task_1          │
                    │                       │
                    │ - NO ablation         │
                    │ - NO planning         │
                    │ - NO loops            │
                    │ - Just implement ONE  │
                    │   focused change      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Debug Loop          │
                    │ - Only if errors      │
                    │ - Max 6 iterations    │
                    └───────────┬───────────┘
                                │
                                ▼
                        Improved Code
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Submission Agent     │
                    │ - Generates CSV       │
                    └───────────┬───────────┘
                                │
                                ▼
                        Run 1 Complete
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Enhancer Agent      │
                    │ - Analyzes Run 1      │
                    │ - Outputs ONE         │
                    │   improvement for R2  │
                    └───────────────────────┘
```

## Complexity Comparison

### Run 0: Nested Loops

```
Max Iterations Calculation:
- outer_loop_round (ablation cycles): 1
- inner_loop_round (plan steps): 3
- max_debug_round (per step): 6
- max_retry (per debug): 6
- num_solutions (parallel tasks): 2

Per Task: 1 × 3 × 6 × 6 = 108 iterations
Total: 108 × 2 = 216 iterations (worst case)

Actual typical: ~10-20 iterations per task
```

### Run 1+: Single Loop

```
Max Iterations Calculation:
- One implementation attempt
- max_debug_round (if errors): 6
- max_retry (per debug): 6
- num_solutions: 1

Total: 1 × 6 × 6 = 36 iterations (worst case)

Actual typical: 1-3 iterations
```

## Routing Logic

### Decision Point: Task 1 Execution in Refinement Agent

```python
def route_to_appropriate_mode(callback_context):
    is_refinement_run = callback_context.state.get("is_refinement_run", False)
    
    if is_refinement_run:  # Run 1+
        # Execute simple agent inline
        runner = Runner(single_improvement_agent_task_1)
        result = runner.run(callback_context.state)
        callback_context.state.update(result)
        return types.Content(
            parts=[types.Part(text="simple mode executed")], 
            role="model"
        )
    else:  # Run 0
        # Return None to continue with complex agent
        return None
```

### Flow Control:

**Run 0 Path:**
```
is_refinement_run = False
    ↓
Router returns None
    ↓
Agent continues with next sub-agent
    ↓
Ablation → Planning → Execution loops run normally
```

**Run 1+ Path:**
```
is_refinement_run = True
    ↓
Router executes single_improvement_agent_task_1
    ↓
Router updates state with results
    ↓
Router returns Content (model response)
    ↓
Agent interprets this as completion
    ↓
Remaining complex agents never execute
    ↓
Control returns to pipeline (moves to next agent)
```

## State Variables

### Run 0 State (Example)
```json
{
  "run_id": 0,
  "is_refinement_run": false,
  "num_solutions": 2,
  "task_description": "...",
  "train_code_0_1": "# Initial solution 1",
  "train_code_0_2": "# Initial solution 2",
  "train_code_exec_result_0_1": {"score": 0.25, "returncode": 0},
  "train_code_exec_result_0_2": {"score": 0.27, "returncode": 0},
  "ablation_summary_1": {...},
  "plan_1": [...],
  "refine_step_1": 2,
  "train_code_2_1": "# Final refined solution 1",
  ...
}
```

### Run 1 State (Example)
```json
{
  "run_id": 1,
  "is_refinement_run": true,
  "num_solutions": 1,
  "task_description": "...",
  "improvement_to_implement": {
    "focus": "feature_engineering",
    "description": "Add interaction terms between ocean_proximity and median_income",
    "rationale": "Run 0 showed median_income had high importance..."
  },
  "train_code_0_1": "# Champion from Run 0",
  "train_code_exec_result_0_1": {"score": 0.23, "returncode": 0},
  "single_improvement_code_1": "# Improved code",
  "single_improvement_exec_result_1": {"score": 0.21, "returncode": 0},
  ...
}
```

## Enhancer Output Schemas

### Single-Improvement Schema (Always Used)
```json
{
  "strategic_summary": "Run 0 achieved strong baseline with XGBoost. Feature importance analysis shows median_income and ocean_proximity are key predictors.",
  "next_improvement": {
    "focus": "feature_engineering",
    "description": "Create interaction terms between median_income and ocean_proximity categorical variables",
    "rationale": "These features showed high individual importance but no interaction was captured"
  }
}
```

### Legacy Multi-Goal Schema (Backwards Compatibility Only)
```json
{
  "strategic_summary": "...",
  "config_overrides": {
    "inner_loop_round": 4
  },
  "strategic_goals": [
    {
      "target_agent_phase": "refinement",
      "focus": "feature_engineering",
      "description": "...",
      "rationale": "..."
    }
  ]
}
```

## Prompt Templates

### Run 0: Standard Prompts (Unchanged)
- Initialization: `INITIALIZATION_INSTR`
- Ablation: `ABLATION_AGENT_INSTR`
- Planning: `PLAN_GENERATION_INSTR`
- Execution: `IMPLEMENT_PLAN_STEP_INSTR`
- Ensemble: `ENSEMBLE_AGENT_INSTR`

### Run 1+: New Prompts
- Enhancer: `ENHANCER_SINGLE_IMPROVEMENT_INSTR`
- Implementation: `IMPLEMENT_SINGLE_IMPROVEMENT_INSTR`

## Key Differences Summary

| Aspect | Run 0 | Run 1+ |
|--------|-------|--------|
| **Input** | Baseline config | Champion code + One improvement |
| **Initialization** | Creates N new solutions | Loads champion from previous run |
| **Ablation** | Yes (identifies sensitive parts) | No |
| **Planning** | Yes (multi-step plan) | No |
| **Execution** | Multi-step loop | Single implementation |
| **Ensemble** | Yes (combines N solutions) | No (only 1 solution) |
| **Loop Depth** | 3-4 levels deep | 1 level (debug only) |
| **Max Iterations** | ~200+ | ~36 |
| **Typical Runtime** | 10-20 min | 2-5 min |
| **Risk** | Discovery mode, safe to explore | Focused, must preserve functionality |
| **Output** | N solutions → ensemble | 1 refined solution |

## Configuration Impact

### Affects Run 0 Only:
- `inner_loop_round`: Plan execution steps
- `outer_loop_round`: Ablation cycles
- `num_solutions`: Parallel solutions

### Affects Both Runs:
- `max_debug_round`: Debug iterations per code execution
- `max_retry`: Retry attempts
- `workspace_dir`: Base directory for all runs

### Run 1+ Ignores:
- `inner_loop_round` (no plan execution)
- `outer_loop_round` (no ablation)
- `num_solutions` (hardcoded to 1)

## Meta-Level Iteration

```
Command: --num_runs 5

┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  Run 0  │────▶│  Run 1  │────▶│  Run 2  │────▶│  Run 3  │────▶│  Run 4  │
│(Complex)│     │(Simple) │     │(Simple) │     │(Simple) │     │(Simple) │
└────┬────┘     └────┬────┘     └────┬────┘     └────┬────┘     └────┬────┘
     │               │               │               │               │
     ▼               ▼               ▼               ▼               ▼
  Enhancer        Enhancer        Enhancer        Enhancer        (End)
     │               │               │               │
     └───────────────┴───────────────┴───────────────┘
             Outputs ONE improvement each time
```

Each enhancer analyzes:
- All previous runs (cumulative history)
- Score trends
- What worked / what didn't
- Proposes next focused improvement

## Benefits of This Architecture

1. **Predictability**: Run 0 has known complexity, Run 1+ is simple and fast
2. **Debuggability**: Easy to trace what changed between runs (one improvement)
3. **Safety**: Run 0 functionality preserved with zero code deletions
4. **Efficiency**: Iteration at meta-level, not nested within runs
5. **Clarity**: `--num_runs N` directly controls number of improvement attempts

