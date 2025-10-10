# MLE-STAR File Naming Reference

## Complete File Naming Patterns

### Run 0 (Discovery Mode - Full Pipeline)

#### Initialization Phase
- **Files**: `init_code_1.py`, `init_code_2.py` (per task, 2 models per task)
- **State Keys**: `init_code_{task_id}_{model_id}`
  - Example: `init_code_1_1`, `init_code_1_2` for Task 1
- **Exec Result Keys**: `init_code_exec_result_{task_id}_{model_id}`

#### Merger Phase
- **Files**: `train0_{reference_idx}.py` (iterations), then `train0.py` (final merged)
- **State Keys**: `train_code_0_{task_id}` (step=0)
  - Example: `train_code_0_1` for Task 1
- **Exec Result Keys**: `train_code_exec_result_0_{task_id}`

#### Ablation Phase
- **Files**: `ablation_{step}.py`
  - Example: `ablation_0.py` (step=0)
- **State Keys**: `ablation_code_{step}_{task_id}`
  - Example: `ablation_code_0_1`
- **Exec Result Keys**: `ablation_code_exec_result_{step}_{task_id}`

#### Plan Execution Phase (Multi-step refinement)
- **Files**: `train{step}_improve{inner_iter}.py`
  - Example: `train0_improve0.py` (step=0, inner_iter=0)
- **State Keys**: `train_code_improve_{inner_iter}_{step}_{task_id}`
  - Example: `train_code_improve_0_0_1`
- **Exec Result Keys**: `train_code_improve_exec_result_{inner_iter}_{step}_{task_id}`
- **Note**: `inner_iter` comes from `inner_loop_round` config (default=3)

#### After Outer Loop (Best solution from refinement cycle)
- **Files**: `train{step}_{task_id}.py`
  - Example: `train1_1.py` (step=1, task_id=1) - THIS IS THE CHAMPION from Run 0
- **State Keys**: `train_code_{step}_{task_id}`
  - Example: `train_code_1_1`
- **Exec Result Keys**: `train_code_exec_result_{step}_{task_id}`
- **Note**: `step` increments after each outer loop iteration

#### Ensemble Phase
- **Files**: `ensemble{iter}.py`, `final_solution.py`
  - Example: `ensemble0.py`
- **State Keys**: `ensemble_code_{ensemble_iter}`
- **Exec Result Keys**: `ensemble_code_exec_result_{ensemble_iter}`

#### Final Submission
- **Files**: `final_solution.py` (in ensemble directory)
- **State Keys**: `submission_code`
- **Exec Result Keys**: `submission_code_exec_result`

---

### Run 1+ (Refinement Mode - Simplified Single Improvement)

#### Champion Loading
- **State Keys**: `train_code_0_{task_id}` (loaded from previous run's best file)
  - Example: `train_code_0_1` contains the champion code
- **Exec Result Keys**: `train_code_exec_result_0_{task_id}` (champion's score)

#### Single Improvement Phase
- **Files**: `train1_improve1.py` (ALWAYS this name for Run 1+)
- **State Keys**: `train_code_improve_1_{task_id}`
  - Example: `train_code_improve_1_1`
- **Exec Result Keys**: `train_code_improve_exec_result_1_{task_id}`
  - Example: `train_code_improve_exec_result_1_1`
- **Suffix Pattern**: `1_{task_id}` (hardcoded as step=1, improvement=1)

#### Final Submission (No ensemble in Run 1+)
- **Files**: `final_solution.py` (copies the improved code)
- **State Keys**: `submission_code`
- **Exec Result Keys**: `submission_code_exec_result`

---

## Critical Patterns to Remember

### Step Indexing
- **Step 0**: Initial code (after initialization/merger)
- **Step 1+**: After refinement cycles complete

### Inner Iteration Indexing
- **inner_iter**: 0, 1, 2, ... (up to `inner_loop_round - 1`)
- Used only in Run 0's plan execution phase

### Task Indexing
- **task_id**: 1, 2, ... (up to `num_solutions`)
- Run 0: typically 2 tasks (Task 1, Task 2)
- Run 1+: always 1 task (Task 1 only)

### Model Indexing (Initialization only)
- **model_id**: 1, 2 (two candidate models per task)

---

## State Key Construction Functions

### `get_updated_suffix()`
Constructs the suffix for state keys based on agent name:

| Agent Prefix | Suffix Pattern | Example |
|--------------|----------------|---------|
| `model_eval` | `{task_id}_{model_id}` | `1_1` |
| `merger` | `{task_id}_{reference_idx}` | `1_0` |
| `check_data_use` | `{task_id}` | `1` |
| `ablation` | `{step}_{task_id}` | `0_1` |
| `plan_implement` | `{inner_iter}_{step}_{task_id}` | `0_0_1` |
| `single_improvement` | `1_{task_id}` | `1_1` |
| `ensemble_plan_implement` | `{ensemble_iter}` | `0` |
| `submission` | `` (empty) | |

### `get_code_state_key()`
Maps agent name to state key pattern:

| Agent Prefix | State Key Pattern |
|--------------|-------------------|
| `model_eval` | `init_code_{suffix}` |
| `merger` | `merger_code_{suffix}` |
| `check_data_use` | `train_code_0_{suffix}` |
| `ablation` | `ablation_code_{suffix}` |
| `plan_implement` | `train_code_improve_{suffix}` |
| `single_improvement` | `train_code_improve_{suffix}` |
| `ensemble_plan_implement` | `ensemble_code_{suffix}` |
| `submission` | `submission_code` |

### File Path Construction in `evaluate_code()`

| Agent Prefix | File Path Pattern |
|--------------|-------------------|
| `model_eval` | `init_code_{model_id}.py` |
| `merger` | `train0_{reference_idx}.py` |
| `check_data_use` | `train0.py` |
| `ablation` | `ablation_{step}.py` |
| `plan_implement` | `train{step}_improve{inner_iter}.py` |
| `single_improvement` | `train1_improve1.py` |
| `ensemble_plan_implement` | `ensemble{suffix}.py` |
| `submission` | `final_solution.py` |

---

## Best Solution Path Construction (run_meta.py)

### Run 0 (Discovery Mode)
```python
last_step = final_state.get(f'refine_step_{task_id}', 0)
best_solution_path = f"run_{run_id}/{task_id}/train{last_step}_{task_id}.py"
# Example: run_0/1/train1_1.py (if refine_step_1 = 1)
```

### Run 1+ (Refinement Mode)
```python
best_solution_path = f"run_{run_id}/{task_id}/train1_improve1.py"
# Example: run_1/1/train1_improve1.py
```

### Ensemble (if best score comes from ensemble)
```python
best_solution_path = f"run_{run_id}/ensemble/final_solution.py"
# Example: run_0/ensemble/final_solution.py
```

---

## Common Pitfalls

### ❌ Wrong: Using step=0 for Run 1+ best solution
```python
# This is WRONG for Run 1+:
best_solution_path = f"run_1/1/train0_1.py"  # File doesn't exist!
```

### ✅ Correct: Using hardcoded train1_improve1.py for Run 1+
```python
# This is CORRECT for Run 1+:
best_solution_path = f"run_1/1/train1_improve1.py"  # File exists!
```

### ❌ Wrong: Assuming refine_step exists in Run 1+
```python
# Run 1+ doesn't increment refine_step, so this is wrong:
last_step = final_state.get(f'refine_step_1', 0)  # Returns 0 or doesn't exist
```

### ✅ Correct: Check is_refinement_run flag
```python
# Always check the run mode:
is_refinement_run = final_state.get("is_refinement_run", False)
if is_refinement_run:
    # Use hardcoded Run 1+ pattern
    best_solution_path = f"run_{run_id}/1/train1_improve1.py"
else:
    # Use dynamic Run 0 pattern
    last_step = final_state.get('refine_step_1', 0)
    best_solution_path = f"run_{run_id}/1/train{last_step}_1.py"
```

---

## Verification Checklist

- [ ] Run 0 creates `train1_1.py` as champion (step increments to 1 after outer loop)
- [ ] Run 0's run_history records `best_solution_path = "run_0/1/train1_1.py"`
- [ ] Run 1 loads champion from `run_0/1/train1_1.py`
- [ ] Run 1 creates `train1_improve1.py` as improved solution
- [ ] Run 1's run_history records `best_solution_path = "run_1/1/train1_improve1.py"`
- [ ] Run 2 loads champion from `run_1/1/train1_improve1.py`
- [ ] All state keys match their corresponding file paths
- [ ] No hardcoded step=0 assumptions for Run 1+

---

## File Locations Summary

### Run 0 Final Champion
- **File**: `run_0/1/train1_1.py` (or `run_0/2/train1_2.py` for Task 2)
- **How Created**: Best solution selected after outer loop completes (refine_step increments to 1)

### Run 1+ Improved Solution
- **File**: `run_1/1/train1_improve1.py`
- **How Created**: Single improvement agent writes to this hardcoded filename

### Ensemble Solutions
- **File**: `run_{id}/ensemble/final_solution.py`
- **How Created**: Submission agent copies best solution or creates ensemble
