# Simplified Run 1+ Implementation Summary

## Overview

Successfully implemented an ultra-simplified architecture for Run 1+ while preserving Run 0's full multi-step pipeline. This eliminates nested loop complexity and implements one improvement per run.

## Architecture Changes

### Run 0 (Discovery Mode - UNCHANGED)
- Full pipeline: Initialization → Ablation → Plan Generation → Multi-step Execution → Ensemble → Submission
- Uses nested loop agents with all original functionality preserved
- No enhancer input (runs with baseline or user-provided config)

### Run 1+ (Simplified Refinement Mode - NEW)
- Ultra-simple: Load champion → Implement ONE enhancer suggestion → Done
- NO nested loops, NO ablation, NO multi-step planning
- Iteration happens at meta-level via `--num_runs` parameter
- Each run gets exactly ONE improvement from enhancer

## Files Modified

### 1. `machine_learning_engineering/sub_agents/refinement/agent.py`

**Added Functions:**
- `get_single_improvement_instruction(context)` - Formats instruction for implementing a single improvement
- `single_improvement_agent_task_1` - Simplified agent that implements one improvement using debug_util wrapper
- `route_to_appropriate_mode(callback_context)` - Router callback that:
  - Checks `is_refinement_run` flag
  - Run 1+: Executes simple agent inline and returns early
  - Run 0: Returns None to continue with full pipeline

**Modified:**
- Task wrapping logic: Task 1 now wrapped with router instead of direct append
- Preserves all original agent code (ablation, planning, execution loops)

### 2. `machine_learning_engineering/sub_agents/refinement/prompt.py`

**Added:**
- `IMPLEMENT_SINGLE_IMPROVEMENT_INSTR` - New prompt for implementing single improvements
  - Receives current code, score, and improvement details
  - Instructs to implement ONLY the specific improvement
  - Preserves all existing constraints (indentation, no try/except, XGBoost rules, etc.)

### 3. `machine_learning_engineering/sub_agents/enhancer/agent.py`

**Modified `get_enhancer_instruction()`:**
- Now ALWAYS uses single-improvement mode
- Rationale: Enhancer only runs AFTER Run 0 to plan Run 1+
- Calculates score delta between runs for context
- Removed conditional logic (originally had buggy if/else)

**Modified `parse_enhancer_output()`:**
- Now supports dual schema detection:
  1. **Single-improvement mode** (Run 1+): Expects `next_improvement` object with:
     - `focus`: Category of improvement
     - `description`: Specific action to take
     - `rationale`: Evidence-based justification
  2. **Legacy multi-goal mode**: Still parses old `strategic_goals` array for backwards compatibility
- Enhanced error handling with appropriate fallbacks for each mode

### 4. `machine_learning_engineering/sub_agents/enhancer/prompt.py`

**Added:**
- `ENHANCER_SINGLE_IMPROVEMENT_INSTR` - New prompt for single-improvement planning
  - Analyzes run history and final state
  - Outputs ONE focused, actionable improvement
  - Provides strategic summary and rationale
  - Clear JSON schema with focus categories

### 5. `run_meta.py`

**Modified `_execute_pipeline_run()`:**
- Added improvement seeding for Run 1+
- Reads `next_improvement` from enhancer output
- Seeds `improvement_to_implement` in initial_state
- Adds debug logging to show what improvement is being implemented

## Data Flow

### After Run 0 Completes:

1. **Enhancer Invocation:**
   ```
   run_meta.py → _invoke_enhancer(last_run_id=0)
   ↓
   enhancer/agent.py → get_enhancer_instruction()
   ↓
   Uses ENHANCER_SINGLE_IMPROVEMENT_INSTR prompt
   ↓
   LLM analyzes Run 0 results
   ↓
   parse_enhancer_output() validates JSON
   ↓
   Output: { "strategic_summary": "...", "next_improvement": {...} }
   ↓
   Saved to run_history.json
   ```

2. **Run 1 Initialization:**
   ```
   run_meta.py → _execute_pipeline_run(run_id=1)
   ↓
   Loads champion code from Run 0
   ↓
   Seeds initial_state with:
     - train_code_0_1 = champion_code
     - train_code_exec_result_0_1 = {score: ..., returncode: 0}
     - is_refinement_run = True
     - improvement_to_implement = enhancer_output["next_improvement"]
   ↓
   Pipeline starts
   ```

3. **Refinement Agent Execution:**
   ```
   refinement/agent.py → route_to_appropriate_mode()
   ↓
   Checks is_refinement_run → True
   ↓
   Runs single_improvement_agent_task_1
   ↓
   get_single_improvement_instruction() formats prompt
   ↓
   Uses IMPLEMENT_SINGLE_IMPROVEMENT_INSTR with:
     - current_code
     - current_score
     - improvement_focus
     - improvement_description
     - improvement_rationale
   ↓
   debug_util.get_run_and_debug_agent() handles:
     - Code execution
     - Debug loop if errors occur
     - Retry logic
   ↓
   Returns improved code
   ↓
   Returns early from router (skips complex agent)
   ```

4. **Remaining Pipeline:**
   ```
   Ensemble agent skipped (conditional in agent.py)
   ↓
   Submission agent runs (generates submission.csv)
   ↓
   Run 1 completes
   ↓
   Enhancer analyzes Run 1 to plan Run 2
   ↓
   Cycle repeats
   ```

## Key Design Decisions

### 1. Conditional Routing vs Code Duplication
- **Chosen:** Conditional routing via callback
- **Rationale:** Preserves all Run 0 code unchanged, no deletion risk
- **Implementation:** `route_to_appropriate_mode()` intercepts Task 1 execution

### 2. State-Based vs Parameter-Based Mode Detection
- **Chosen:** State-based via `is_refinement_run` flag
- **Rationale:** Consistent with existing codebase patterns
- **Set by:** `run_meta.py` based on `run_id > 0`

### 3. Schema Detection vs Version Flag
- **Chosen:** Schema detection in parser
- **Rationale:** Backwards compatible, no coordination needed
- **Implementation:** Check for `next_improvement` vs `strategic_goals`

### 4. Prompt Reuse vs New Prompts
- **Chosen:** Created new specialized prompts
- **Rationale:** Single-improvement mode has different requirements
- **Added:** `IMPLEMENT_SINGLE_IMPROVEMENT_INSTR` and `ENHANCER_SINGLE_IMPROVEMENT_INSTR`

## Testing Checklist

### Run 0 (Full Pipeline - No Regression)
- [ ] Initialization agent creates multiple solutions
- [ ] Ablation agent identifies sensitive components
- [ ] Plan generation creates multi-step plan
- [ ] Plan execution completes all steps (controlled by `inner_loop_round`)
- [ ] Ensemble agent combines solutions
- [ ] Submission agent generates final output
- [ ] No enhancer input used

### Run 1 (Simplified Mode)
- [ ] Champion code loaded from Run 0
- [ ] Enhancer's single improvement seeded in state
- [ ] Single improvement agent executes (not complex agent)
- [ ] Only ONE code modification made
- [ ] No ablation, no planning loops
- [ ] Debug loop works if code has errors
- [ ] Task 2+ skipped (linear refinement mode)
- [ ] Submission generated from refined code

### Enhancer Behavior
- [ ] After Run 0: Outputs `next_improvement` JSON
- [ ] After Run 1: Outputs `next_improvement` JSON
- [ ] Includes strategic summary analyzing previous run
- [ ] Focus area is one of valid categories
- [ ] Description is specific and actionable
- [ ] Rationale references empirical evidence

## Configuration Parameters (Unchanged)

These parameters still control Run 0 behavior:
- `inner_loop_round`: Number of plan steps to execute (default: 3)
- `outer_loop_round`: Ablation+plan+execute cycles (default: 1)
- `max_debug_round`: Max debug iterations per code execution (default: 6)
- `max_retry`: Max retry attempts (default: 6)
- `num_solutions`: Number of parallel solutions in Run 0 (default: 2)

Run 1+ ignores these loop parameters and always executes ONE improvement.

## Benefits Achieved

1. **Eliminated Nested Loop Complexity:**
   - Run 0: 3 × 1 × 6 × 6 = 108 max iterations (acceptable)
   - Run 1+: No nested loops, just debug/retry (6 max iterations)

2. **Clearer Mental Model:**
   - `--num_runs N`: Meta-level iteration (Run 0, Run 1, ..., Run N-1)
   - Each Run 1+ implements ONE focused improvement
   - Iteration happens across runs, not within runs

3. **Preserved Run 0 Functionality:**
   - Zero deletions from original complex agent code
   - Conditional routing ensures Run 0 path unchanged
   - All original prompts and logic intact

4. **Improved Debugging:**
   - Added logging for improvement seeding
   - Clear mode indicators in console output
   - Schema validation with helpful error messages

## Known Limitations

1. **Run 1+ Only Refines Task 1:**
   - Linear refinement mode sets `num_solutions = 1`
   - Task 2+ directories created for agent compatibility but skipped
   - This is intentional (focus refinement on champion only)

2. **Ensemble Agent Skipped in Run 1+:**
   - Conditional logic skips ensemble when `is_refinement_run = True`
   - Only one solution exists, so ensembling not applicable

3. **Config Overrides Limited:**
   - Run 1+ doesn't use `config_overrides` from enhancer
   - Could be added if needed for adaptive hyperparameters

## Future Enhancements (Not Implemented)

1. **Adaptive Loop Parameters:**
   - Enhancer could adjust `max_debug_round` based on run success
   - Currently only affects Run 0 (Run 1+ has fixed debug loop)

2. **Multi-Improvement Bundles:**
   - Could allow enhancer to output N improvements to try in parallel
   - Would require multiple task directories in Run 1+

3. **Rollback on Regression:**
   - If Run 1 score worse than Run 0, automatically revert
   - Would need score comparison logic in run_meta.py

4. **Improvement History Tracking:**
   - Track which improvements succeeded/failed
   - Feed back to enhancer to avoid repeating failed strategies

## Command Usage

### Run full meta-learning workflow with N runs:
```bash
python -m dotenv run -- python /Users/richard/Documents/adk-samples/python/agents/machine-learning-engineering/run_meta.py \
  --task_name california-housing-prices \
  --num_runs 3
```

- Run 0: Full discovery with multi-step refinement
- Run 1: Implement improvement #1 from enhancer
- Run 2: Implement improvement #2 from enhancer

### Resume from existing run history:
```bash
# If run_history.json exists, will skip completed runs and continue
python -m dotenv run -- python run_meta.py --task_name california-housing-prices --num_runs 5
```

## Implementation Status

✅ **COMPLETE** - All planned changes implemented and tested for linting errors
- [x] Created single-improvement instruction function
- [x] Created simplified agent (single_improvement_agent_task_1)
- [x] Added routing callback (route_to_appropriate_mode)
- [x] Modified task wrapping to use router
- [x] Added IMPLEMENT_SINGLE_IMPROVEMENT_INSTR prompt
- [x] Added ENHANCER_SINGLE_IMPROVEMENT_INSTR prompt
- [x] Updated enhancer instruction logic (always use single-improvement)
- [x] Updated enhancer output parser (dual schema support)
- [x] Added improvement seeding in run_meta.py
- [x] Verified no linting errors
- [x] Created summary documentation

Ready for testing with actual runs.

