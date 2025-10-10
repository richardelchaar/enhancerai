# Fix: FileNotFoundError for run_1/2/ Directory

## Problem

When running Run 1 (refinement mode), the system crashed with:
```
FileNotFoundError: [Errno 2] No such file or directory: 
'./machine_learning_engineering/workspace/california-housing-prices/run_1/2/ablation_0.py'
```

## Root Cause

The refinement agent is constructed at **module import time** (line 22 of `run_meta.py`):

```python
from machine_learning_engineering.agent import mle_pipeline_agent
```

When this import happens, the refinement agent's parallel sub-agents are created based on `config.CONFIG.num_solutions` (line 313 of `refinement/agent.py`):

```python
for k in range(config.CONFIG.num_solutions):
    task_id = str(k + 1)
    # Creates parallel agents for task 1, 2, 3, etc.
```

**The Problem:**
1. At import time, `CONFIG.num_solutions` = 2 (default) or 3 (from enhancer)
2. Parallel agents are created for task IDs 1, 2, (and possibly 3)
3. Later in `run_meta.py`, we set `num_solutions = 1` and only create directory `run_1/1/`
4. But agent 2 still tries to write to `run_1/2/` → **FileNotFoundError**

## Solution

Override `config.CONFIG.num_solutions = 1` **at runtime** before the refinement run starts, so that when agents try to access directories, they only look for task ID 1.

### Changes Made

**File:** `run_meta.py` (lines 113-118)

```python
if is_refinement_run:
    print(f"[Run {run_id}] Initializing Linear Refinement Mode")
    
    # CRITICAL: Override CONFIG.num_solutions BEFORE agents access directories
    # The agents were constructed at import time with the default/enhanced value,
    # but for linear refinement we only want 1 solution thread
    num_solutions = 1
    config.CONFIG.num_solutions = num_solutions
    print(f"[Run {run_id}] Forcing num_solutions = {num_solutions} for linear refinement")
```

**Also added safety reset for discovery runs** (lines 111-113):

```python
# Reset num_solutions for discovery runs (in case it was overridden before)
if not is_refinement_run:
    config.CONFIG.num_solutions = run_config.get("num_solutions", 2)
```

## Why This Works

Even though the parallel agents were already constructed at import time with multiple task IDs, they dynamically access `config.CONFIG.num_solutions` at runtime when determining which task directories to use. By overriding the CONFIG before the run starts, we ensure:

1. ✅ Only directory `run_1/1/` is created
2. ✅ Only task ID 1 is active during refinement
3. ✅ Agents don't try to access non-existent `run_1/2/` or `run_1/3/`
4. ✅ All computational budget focused on single linear refinement path

## Architecture Note

This is a limitation of the ADK's agent construction model:
- Agents are constructed at module import time (static)
- But we want dynamic behavior based on run context (refinement vs discovery)

The workaround is to override the global CONFIG at runtime, which the agents check when accessing directories.

## Verification

After this fix, Run 1 should:
1. Only show 1 solution thread (`[Run 1] Forcing num_solutions = 1`)
2. Only create/use `run_1/1/` directory
3. Not attempt to access `run_1/2/` or higher
4. Complete without FileNotFoundError

Run the pipeline and verify:
```bash
python -m dotenv run -- python run_meta.py --task_name california-housing-prices --num_runs 2
```

Expected log output:
```
[Run 1] Initializing Linear Refinement Mode
[Run 1] Forcing num_solutions = 1 for linear refinement
[Run 1] Loading champion from: ...
[Run 1] Created workspace structure and seeded 1 solution(s) with champion code
```

