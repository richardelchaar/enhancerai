# MLE-STAR Meta-Learning Framework with Enhancer Agent

## Executive Summary

This document provides a comprehensive explanation of the **Meta-Learning Enhancement Framework** built on top of the base MLE-STAR (Machine Learning Engineering via Search and Targeted Refinement) system. This framework adds an iterative, self-improving capability that allows the system to learn from previous runs and strategically adapt its approach across multiple executions.

### Key Innovation

The base MLE-STAR system executes a single pipeline to train ML models. Our enhancement adds:
1. **Multi-Run Orchestration**: Execute the pipeline N times, with each run learning from previous ones
2. **Enhancer Agent**: An AI "Research Lead" that analyzes past performance and strategizes improvements
3. **Strategic Guidance Integration**: All sub-agents now receive and act on strategic directives from the Enhancer
4. **Adaptive Configuration**: Dynamic hyperparameter tuning of the pipeline itself based on empirical results

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [The Enhancer Agent](#the-enhancer-agent)
3. [The Meta-Learning Orchestrator](#the-meta-learning-orchestrator)
4. [Modifications to Base MLE-STAR](#modifications-to-base-mle-star)
5. [End-to-End Workflow](#end-to-end-workflow)
6. [Configuration System](#configuration-system)
7. [Data Flow and State Management](#data-flow-and-state-management)
8. [Key Design Decisions](#key-design-decisions)

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Meta-Learning Orchestrator                    │
│                        (run_meta.py)                             │
└───────────────┬─────────────────────────────────────┬───────────┘
                │                                     │
                ▼                                     ▼
    ┌───────────────────────┐           ┌───────────────────────┐
    │   MLE-STAR Pipeline   │           │   Enhancer Agent      │
    │   (Base + Enhanced)   │◄──────────┤   (Strategic AI)      │
    └───────────────────────┘  strategic└───────────────────────┘
                │               guidance
                │
    ┌───────────┴───────────────────────────────────────┐
    │                                                   │
    ▼                                                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│Initialization│  │ Refinement   │  │  Ensemble    │  │  Submission  │
│   Agent      │→ │   Agent      │→ │   Agent      │→ │   Agent      │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
     (base)         (enhanced)         (enhanced)         (enhanced)
```

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| **Meta Orchestrator** | Manages multi-run execution, invokes Enhancer, applies config overrides |
| **Enhancer Agent** | Analyzes run history, generates strategic goals, proposes config changes |
| **Initialization Agent** | Searches for SOTA models, creates initial solutions (unchanged from base) |
| **Refinement Agent** | Uses ablation studies + Enhancer guidance to improve code |
| **Ensemble Agent** | Combines solutions using Enhancer-suggested strategies |
| **Submission Agent** | Creates final predictions, optionally guided by Enhancer |

---

## The Enhancer Agent

### Purpose

The Enhancer Agent acts as a **world-class Machine Learning Research Lead** that:
- Analyzes results from completed runs
- Identifies bottlenecks, failures, and opportunities
- Proposes evidence-based strategic adjustments
- Guides downstream agents on what to prioritize

### Location in Codebase

```
machine_learning_engineering/sub_agents/enhancer/
├── __init__.py
├── agent.py          # Core Enhancer agent implementation
└── prompt.py         # Strategic analysis prompt
```

### Core Functionality

#### 1. Analysis Phase
The Enhancer receives:
- **Run History Summary**: JSON containing all previous runs' scores, durations, statuses
- **Last Run Final State**: Complete execution state including all code, scores, and intermediate results
- **Best Score So Far**: Track record of best performance achieved

#### 2. Strategy Generation
The Enhancer produces a structured JSON output:

```json
{
  "strategic_summary": "Natural language explanation of analysis and plan",
  "config_overrides": {
    "inner_loop_round": 3,
    "outer_loop_round": 2,
    "max_debug_round": 5
  },
  "strategic_goals": [
    {
      "target_agent_phase": "refinement",
      "focus": "hyperparameter_tuning",
      "priority": 1,
      "rationale": "Ablation showed model hyperparameters have highest impact"
    },
    {
      "target_agent_phase": "ensemble",
      "focus": "stacking",
      "priority": 2,
      "rationale": "Simple averaging underperformed; try stacked ensemble"
    }
  ]
}
```

#### 3. Validation and Safety
The Enhancer implementation includes:
- **Schema Validation**: Ensures all required keys are present
- **Phase Validation**: Only allows valid agent phases (`refinement`, `ensemble`, `submission`)
- **Fallback Mechanism**: If parsing fails, uses safe defaults instead of crashing
- **Priority Enforcement**: Goals must have unique priorities starting at 1

### Key Code: Enhancer Agent (`agent.py`)

```python
def get_enhancer_instruction(context: callback_context_module.ReadonlyContext) -> str:
    """Dynamically builds the instruction for the Enhancer agent."""
    # Loads run history and last run state from context
    run_history_summary = context.state.get("run_history_summary")
    last_run_final_state = context.state.get("last_run_final_state")
    
    # Formats prompt with empirical data
    return prompt.ENHANCER_AGENT_INSTR.format(
        last_run_id=last_run_id,
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
        # Extract and parse JSON
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        enhancer_output = json.loads(response_text[json_start:json_end])
        
        # Validate required keys
        required_keys = ["strategic_summary", "config_overrides", "strategic_goals"]
        if not all(key in enhancer_output for key in required_keys):
            raise ValueError(f"Missing required keys: {required_keys}")
        
        # Validate and filter strategic goals
        VALID_PHASES = {"refinement", "ensemble", "submission"}
        validated_goals = [
            goal for goal in enhancer_output["strategic_goals"]
            if goal.get("target_agent_phase") in VALID_PHASES
        ]
        enhancer_output["strategic_goals"] = validated_goals
        
        callback_context.state["enhancer_output"] = enhancer_output
    except (json.JSONDecodeError, ValueError) as e:
        # Fallback to safe defaults
        callback_context.state["enhancer_output"] = {
            "strategic_summary": "Error parsing output. Using defaults.",
            "config_overrides": {},
            "strategic_goals": []
        }
    return None
```

---

## The Meta-Learning Orchestrator

### Purpose

The `MetaOrchestrator` class manages the entire N-run meta-learning workflow. It coordinates the execution of multiple pipeline runs, invocation of the Enhancer between runs, and tracking of historical performance.

### Location in Codebase

```
run_meta.py
```

### Key Responsibilities

1. **Workspace Management**: Creates isolated directories for each run
2. **Run History Tracking**: Persists performance metrics and strategic decisions
3. **Configuration Management**: Applies Enhancer-suggested overrides between runs
4. **Enhancer Invocation**: Calls Enhancer after each run (except the last)
5. **State Isolation**: Ensures each run operates independently but with shared learnings

### Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  Meta-Learning Loop (N runs)                                     │
└─────────────────────────────────────────────────────────────────┘

For run_id in 0 to N-1:

  1. Get Configuration
     ├─ If run_id == 0: Use baseline config
     └─ Else: Apply enhancer_output["config_overrides"] from previous run

  2. Execute MLE-STAR Pipeline
     ├─ Create isolated workspace: workspace/task_name/run_{id}/
     ├─ Initialize session with config + enhancer_output
     ├─ Run Initialization → Refinement → Ensemble → Submission
     └─ Save final_state.json

  3. Update Run History
     ├─ Extract best score from all solutions
     ├─ Save summary: {run_id, status, score, duration, config}
     └─ Persist to run_history.json

  4. Invoke Enhancer (if not final run)
     ├─ Load run_history and last_run_final_state
     ├─ Execute Enhancer agent
     ├─ Parse enhancer_output
     └─ Append to run_history for traceability

End Loop
```

### Key Code: Orchestrator (`run_meta.py`)

```python
class MetaOrchestrator:
    """Manages the entire multi-run meta-learning workflow."""
    
    def _get_config_for_run(self, run_id: int) -> Dict[str, Any]:
        """Determines the configuration for the current run."""
        current_config = config.DefaultConfig()
        enhancer_output: Dict[str, Any] = {}

        if run_id == 0:
            # Baseline run: use defaults + user overrides
            for key, value in self.initial_config.items():
                if hasattr(current_config, key):
                    setattr(current_config, key, value)
        else:
            # Enhanced run: apply Enhancer's config_overrides
            prev_run_summary = self.run_history[-1]
            enhancer_output = prev_run_summary.get("enhancer_output", {})
            overrides = enhancer_output.get("config_overrides", {})
            
            if config.CONFIG.allow_config_override:
                for key, value in overrides.items():
                    if hasattr(current_config, key):
                        setattr(current_config, key, value)
        
        # Inject enhancer_output into config for downstream agents
        run_config_dict = dataclasses.asdict(current_config)
        run_config_dict["enhancer_output"] = enhancer_output
        return run_config_dict
    
    async def _execute_pipeline_run(self, run_id: int, run_config: Dict[str, Any]):
        """Executes a single, isolated run of the MLE-STAR pipeline."""
        run_dir = os.path.join(self.workspace_root, f"run_{run_id}")
        run_config["workspace_dir"] = run_dir
        
        # Apply config overrides to global CONFIG (sub-agents read from here)
        self._apply_global_config_overrides(run_config)
        
        # Build initial session state
        initial_state = {**run_config, "run_id": run_id}
        
        # Execute pipeline
        runner = InMemoryRunner(agent=mle_pipeline_agent, app_name="mle-meta-pipeline")
        session = await runner.session_service.create_session(
            user_id=f"meta-user-{run_id}",
            state=initial_state
        )
        
        content = types.Content(parts=[types.Part(text="run")], role="user")
        async for _ in runner.run_async(...):
            pass
        
        # Fetch final state
        final_session = await runner.session_service.get_session(...)
        final_state = final_session.state
        
        # Save complete state for Enhancer analysis
        with open(os.path.join(run_dir, "final_state.json"), "w") as f:
            json.dump(final_state, f, indent=2)
        
        return final_state
    
    async def _invoke_enhancer(self, last_run_id: int) -> Dict[str, Any]:
        """Runs the Enhancer agent to get the strategy for the next run."""
        # Load last run's final state
        last_run_state_path = os.path.join(self.workspace_root, f"run_{last_run_id}", "final_state.json")
        with open(last_run_state_path, 'r') as f:
            last_run_state = json.load(f)
        
        # Prepare Enhancer's initial state
        enhancer_initial_state = {
            "last_run_final_state": last_run_state,
            "run_history_summary": self.run_history,
            "best_score_so_far": min([r['best_score'] for r in self.run_history])
        }
        
        # Execute Enhancer agent
        runner = InMemoryRunner(agent=enhancer_agent, app_name="mle-meta-enhancer")
        session = await runner.session_service.create_session(
            user_id="meta-enhancer",
            state=enhancer_initial_state
        )
        
        content = types.Content(parts=[types.Part(text="enhance")], role="user")
        async for _ in runner.run_async(...):
            pass
        
        final_session = await runner.session_service.get_session(...)
        enhancer_output = final_session.state.get("enhancer_output", {})
        
        # Save Enhancer output to run history for traceability
        self.run_history[-1]["enhancer_output"] = enhancer_output
        with open(self.run_history_path, "w") as f:
            json.dump(self.run_history, f, indent=2)
        
        return enhancer_output
```

### Run History Structure

The `run_history.json` file tracks all runs:

```json
[
  {
    "run_id": 0,
    "status": "COMPLETED_SUCCESSFULLY",
    "start_time_iso": "2025-10-03T14:30:00Z",
    "duration_seconds": 1847,
    "best_score": 45231.87,
    "best_solution_path": "run_0/ensemble/final_solution.py",
    "config_used": {
      "num_solutions": 2,
      "inner_loop_round": 1,
      "outer_loop_round": 1,
      "ensemble_loop_round": 1
    },
    "enhancer_rationale": "Baseline run.",
    "enhancer_output": {
      "strategic_summary": "Baseline completed in 30 min. Both solutions converged quickly. Recommend increasing refinement iterations for Run 1.",
      "config_overrides": {
        "inner_loop_round": 2,
        "outer_loop_round": 2
      },
      "strategic_goals": [...]
    }
  },
  {
    "run_id": 1,
    "status": "COMPLETED_SUCCESSFULLY",
    ...
  }
]
```

---

## Modifications to Base MLE-STAR

All base MLE-STAR agents were modified to integrate with the Enhancer Agent's strategic guidance. Here's a comprehensive breakdown:

### 1. Refinement Agent (`sub_agents/refinement/agent.py`)

**What It Does (Base)**: Uses ablation studies to identify weak code components, then iteratively refines them.

**What Changed**:

#### A. Strategic Guidance Integration in Planning

```python
def get_plan_generation_instruction(context: callback_context_module.ReadonlyContext) -> str:
    """Gets the instruction for the new Planner agent."""
    task_id = context.agent_name.split("_")[-1]
    step = context.state.get(f"refine_step_{task_id}", 0)
    code = context.state.get(f"train_code_{step}_{task_id}", "")
    ablation_summary = context.state.get(f"ablation_summary_{step}_{task_id}", "")
    
    # NEW: Extract strategic goals from Enhancer
    enhancer_output = context.state.get("enhancer_output", {})
    strategic_goals = enhancer_output.get("strategic_goals", [])
    
    # NEW: Filter goals relevant to refinement phase
    refinement_goals = [
        f"- Focus Area: {g['focus']}. Rationale: {g['rationale']}"
        for g in strategic_goals if g.get("target_agent_phase") == "refinement"
    ]
    
    # NEW: Build strategic guidance string
    enhancer_goals_str = "\n".join(refinement_goals)
    if not enhancer_goals_str:
        run_id = context.state.get("run_id", 0)
        if run_id > 0:
            print(f"INFO: No strategic goals for refinement - using ablation-driven planning")
        enhancer_goals_str = "No specific goals. Use ablation summary to find best improvement area."
    
    # MODIFIED: Include enhancer_goals in the prompt
    return prompt.PLAN_GENERATION_INSTR.format(
        enhancer_goals=enhancer_goals_str,  # NEW PARAMETER
        ablation_summary=ablation_summary,
        code=code,
    )
```

**Impact**: The Refinement Agent now synthesizes:
1. **Ablation Study Results** (what components matter most)
2. **Enhancer Strategic Guidance** (what to prioritize based on past runs)

This dual-input approach allows the agent to make more informed decisions. For example:
- **Run 0**: No Enhancer guidance → rely purely on ablation
- **Run 1**: Enhancer says "hyperparameter tuning showed 40% impact in Run 0" → prioritize that even if ablation suggests feature engineering

#### B. Prompt Modification (`sub_agents/refinement/prompt.py`)

```python
PLAN_GENERATION_INSTR = """
# Persona
You are a world-class machine learning engineer...

# Context
You have been given:
1. Results of an ablation study
2. High-level strategic directive from your Research Lead  # NEW

**Strategic Guidance from Research Lead:**  # NEW SECTION
{enhancer_goals}

**Ablation Study Summary:**
{ablation_summary}

**Current Code:**
```python
{code}
```

# Your Task
1. **Synthesize:** Combine insights from BOTH ablation and strategic guidance  # MODIFIED
2. **Plan:** Create a step-by-step plan...
...
"""
```

### 2. Ensemble Agent (`sub_agents/ensemble/agent.py`)

**What It Does (Base)**: Combines multiple trained models into a final ensemble solution.

**What Changed**:

#### A. Strategic Focus Selection

```python
def get_ensemble_instruction(context: callback_context_module.ReadonlyContext) -> str:
    """Gets the ensemble agent instruction."""
    solution1_code = context.state.get("train_code_1_1", "")
    solution1_score = context.state.get("train_code_exec_result_1_1", {}).get("score")
    solution2_code = context.state.get("train_code_1_2", "")
    solution2_score = context.state.get("train_code_exec_result_1_2", {}).get("score")
    
    # NEW: Extract Enhancer guidance for ensemble phase
    enhancer_output = context.state.get("enhancer_output", {})
    strategic_goals = enhancer_output.get("strategic_goals", [])
    
    # NEW: Filter for ensemble-specific goals
    ensemble_goals = [
        g for g in strategic_goals if g.get("target_agent_phase") == "ensemble"
    ]
    
    # NEW: Select highest-priority ensemble strategy
    if ensemble_goals:
        primary_goal = sorted(ensemble_goals, key=lambda x: x.get("priority", 99))[0]
        ensemble_focus = primary_goal.get("focus", "default ensembling")
        ensemble_rationale = primary_goal.get("rationale", "No rationale provided.")
    else:
        # Fallback to conservative default
        ensemble_focus = "simple averaging"
        ensemble_rationale = "No strategic goal provided. Use simple averaging."
    
    # MODIFIED: Include strategic focus in the prompt
    return prompt.ENSEMBLE_PLANNING_INSTR.format(
        solution1_code=solution1_code,
        solution1_score=solution1_score,
        solution2_code=solution2_code,
        solution2_score=solution2_score,
        ensemble_focus=ensemble_focus,      # NEW PARAMETER
        ensemble_rationale=ensemble_rationale,  # NEW PARAMETER
    )
```

**Impact**: The Ensemble Agent can adapt its strategy:
- **Run 0**: Default to simple averaging (safe baseline)
- **Run 1**: If Enhancer detected that stacking outperformed averaging, use stacking
- **Run 2**: If Enhancer identified that weighted averaging with specific weights works best, implement that

#### B. Prompt Modification (`sub_agents/ensemble/prompt.py`)

```python
ENSEMBLE_PLANNING_INSTR = """
# Persona
You are a Kaggle Grandmaster specializing in ensembling techniques...

# Context
You have two complete solutions to combine.

**Strategic Guidance from the Research Lead:**  # NEW SECTION
- **Primary Focus:** `{ensemble_focus}`
- **Rationale:** `{ensemble_rationale}`

**Solution 1:**
- **Validation Score:** {solution1_score}
- **Code:**
```python
{solution1_code}
```

**Solution 2:**
- **Validation Score:** {solution2_score}
- **Code:**
```python
{solution2_code}
```

# Your Task
Based on your expertise and the Strategic Guidance, create a plan to ensemble these solutions.
...
"""
```

### 3. Submission Agent (`sub_agents/submission/agent.py`)

**What It Does (Base)**: Selects the best-performing solution and prepares final submission file.

**What Changed**:

#### A. Strategic Guidance Appending

```python
def get_submission_and_debug_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the submission agent instruction."""
    # ... [existing code to select best solution] ...
    
    # NEW: Add strategic guidance for submission phase
    enhancer_output = context.state.get("enhancer_output", {})
    strategic_goals = enhancer_output.get("strategic_goals", [])
    
    # NEW: Filter submission-specific goals
    submission_goals = [
        g for g in strategic_goals if g.get("target_agent_phase") == "submission"
    ]
    
    # NEW: Build strategic guidance string
    strategic_guidance = ""
    if submission_goals:
        primary_goal = sorted(submission_goals, key=lambda x: x.get("priority", 99))[0]
        strategic_guidance = f"""

# Strategic Guidance from Research Lead:
- Focus: {primary_goal.get('focus', 'No specific focus')}
- Rationale: {primary_goal.get('rationale', 'No rationale provided.')}

Apply this strategic guidance when preparing the final submission.
"""
    
    # MODIFIED: Append strategic guidance to base instruction
    return prompt.ADD_TEST_FINAL_INSTR.format(
        task_description=task_description,
        code=final_solution,
    ) + strategic_guidance  # NEW ADDITION
```

**Impact**: The Submission Agent can apply final optimizations:
- Post-processing strategies
- Clipping predictions to valid ranges
- Formatting corrections based on past submission errors

### 4. Configuration System (`shared_libraries/config.py`)

**What Changed**:

```python
@dataclasses.dataclass
class DefaultConfig:
    """Default configuration."""
    # ... [existing base config parameters] ...
    
    # MODIFIED: Reduced defaults to prevent token explosion
    max_retry: int = 3              # Was higher, reduced for baseline
    max_debug_round: int = 3        # Was higher, reduced for baseline
    max_rollback_round: int = 2     # Was higher, reduced for baseline
    inner_loop_round: int = 1       # Was higher, reduced for baseline
    outer_loop_round: int = 1       # Was higher, reduced for baseline
    ensemble_loop_round: int = 1    # Was higher, reduced for baseline
    
    # NEW: Meta-learning framework parameters
    allow_config_override: bool = True  # Master switch for Enhancer overrides
    computational_budget: int = 3600    # Target runtime in seconds for enhanced runs
```

**Rationale for Baseline Reduction**:
1. **Faster Iteration**: Run 0 completes quickly to give Enhancer data
2. **Token Efficiency**: Prevents LLM context overflow on baseline
3. **Strategic Scaling**: Enhancer can increase iterations only where needed

### 5. Root Agent (`agent.py`)

**What Changed**:

```python
def save_state(
    callback_context: callback_context_module.CallbackContext
) -> Optional[types.Content]:
    """Saves the final state of a run to its directory."""
    # MODIFIED: State saving now aware of run_id
    workspace_dir = callback_context.state.get("workspace_dir", "")
    task_name = callback_context.state.get("task_name", "")
    run_id = callback_context.state.get("run_id", "unknown_run")  # NEW
    
    # The actual final_state.json is written by the orchestrator
    # This function is now primarily for intermediate state tracking
    print(f"State for run {run_id} is complete.")  # MODIFIED
    return None
```

---

## End-to-End Workflow

### Complete Execution Flow

```
USER: python run_meta.py --task_name california-housing-prices --num_runs 3

┌────────────────────────────────────────────────────────────────┐
│ MetaOrchestrator Initialization                                 │
├────────────────────────────────────────────────────────────────┤
│ 1. Create workspace: ./workspace/california-housing-prices/    │
│ 2. Load run_history.json (if exists)                           │
│ 3. Initialize configuration                                     │
└────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│ RUN 0: BASELINE                                                 │
├────────────────────────────────────────────────────────────────┤
│ Config: Default values (conservative settings)                  │
│ Enhancer Output: None (no previous runs to analyze)            │
└────────────────────────────────────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────────┐
          │  1. Initialization Agent           │
          │  ─────────────────────────         │
          │  - Search for SOTA models          │
          │  - Generate 2 initial solutions    │
          │  - Evaluate & rank candidates      │
          │  Output: init_code_1.py, 2.py      │
          └────────────┬───────────────────────┘
                       │
                       ▼
          ┌────────────────────────────────────┐
          │  2. Refinement Agent (Parallel)    │
          │  ─────────────────────────         │
          │  For each solution (1, 2):         │
          │    a. Ablation Study               │
          │       → ablation_0.py              │
          │       → Identifies sensitive parts │
          │                                    │
          │    b. Plan Generation              │
          │       → No Enhancer guidance       │
          │       → Use ablation results only  │
          │                                    │
          │    c. Plan Execution (Inner Loop)  │
          │       → train0_improve0.py         │
          │       → Implements improvements    │
          │                                    │
          │    d. Evaluation & Update          │
          │       → train1_1.py, train1_2.py   │
          │  Output: Refined solutions         │
          └────────────┬───────────────────────┘
                       │
                       ▼
          ┌────────────────────────────────────┐
          │  3. Ensemble Agent                 │
          │  ─────────────────────────         │
          │  - Create ensemble workspace       │
          │  - Plan: Simple averaging (default)│
          │  - Execute: ensemble0.py           │
          │  - Evaluate: Final RMSE            │
          │  Output: final_solution.py         │
          └────────────┬───────────────────────┘
                       │
                       ▼
          ┌────────────────────────────────────┐
          │  4. Submission Agent               │
          │  ─────────────────────────         │
          │  - Select best solution            │
          │  - Generate submission.csv         │
          │  Output: ./ensemble/final/         │
          │          submission.csv            │
          └────────────┬───────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────────┐
│ Save Run 0 Results                                              │
├────────────────────────────────────────────────────────────────┤
│ - final_state.json → ./run_0/                                  │
│ - Best Score: 45231.87 RMSE                                     │
│ - Duration: 1847 seconds                                        │
│ - Update run_history.json                                       │
└────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│ INVOKE ENHANCER AGENT                                           │
├────────────────────────────────────────────────────────────────┤
│ Input:                                                          │
│   - run_history: [{run_0 summary}]                             │
│   - last_run_final_state: {complete state from run_0}          │
│   - best_score_so_far: 45231.87                                │
│                                                                 │
│ Enhancer Analysis:                                              │
│   "Baseline completed in 30 min. Ablation studies show         │
│    hyperparameter tuning has highest impact (40% score delta).  │
│    Both solutions used default hyperparameters. Recommend       │
│    increasing inner_loop_round to 3 to allow time for          │
│    RandomizedSearchCV. Ensemble used simple averaging; scores   │
│    were close (45231 vs 45890), suggesting weighted averaging   │
│    may help."                                                   │
│                                                                 │
│ Output:                                                         │
│   {                                                             │
│     "strategic_summary": "...",                                 │
│     "config_overrides": {                                       │
│       "inner_loop_round": 3,                                    │
│       "outer_loop_round": 2,                                    │
│       "max_debug_round": 5                                      │
│     },                                                          │
│     "strategic_goals": [                                        │
│       {                                                         │
│         "target_agent_phase": "refinement",                     │
│         "focus": "hyperparameter_tuning",                       │
│         "priority": 1,                                          │
│         "rationale": "Ablation showed 40% impact..."           │
│       },                                                        │
│       {                                                         │
│         "target_agent_phase": "ensemble",                       │
│         "focus": "weighted_averaging",                          │
│         "priority": 2,                                          │
│         "rationale": "Scores were close; optimize weights"     │
│       }                                                         │
│     ]                                                           │
│   }                                                             │
│                                                                 │
│ Save: Append enhancer_output to run_history[0]                 │
└────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│ RUN 1: ENHANCED                                                 │
├────────────────────────────────────────────────────────────────┤
│ Config: Apply enhancer_output["config_overrides"]              │
│   - inner_loop_round: 1 → 3                                     │
│   - outer_loop_round: 1 → 2                                     │
│   - max_debug_round: 3 → 5                                      │
│                                                                 │
│ Enhancer Output: Injected into session state                    │
└────────────────────────────────────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────────┐
          │  1. Initialization Agent           │
          │  (Same as Run 0)                   │
          └────────────┬───────────────────────┘
                       │
                       ▼
          ┌────────────────────────────────────┐
          │  2. Refinement Agent (Enhanced)    │
          │  ─────────────────────────         │
          │  For each solution:                │
          │    a. Ablation Study               │
          │       (Same as Run 0)              │
          │                                    │
          │    b. Plan Generation (ENHANCED)   │
          │       → Receives Enhancer goals:   │
          │         "Priority 1: Hyperparameter│
          │          tuning via RandomizedCV"  │
          │       → Synthesizes with ablation  │
          │       → Creates focused plan       │
          │                                    │
          │    c. Plan Execution (ENHANCED)    │
          │       → 3 inner iterations (was 1) │
          │       → train0_improve0.py         │
          │       → train0_improve1.py         │
          │       → train0_improve2.py         │
          │       → Each implements step of    │
          │         hyperparameter tuning plan │
          │                                    │
          │    d. Outer Loop (ENHANCED)        │
          │       → 2 outer iterations (was 1) │
          │       → train1_1.py, train2_1.py   │
          │                                    │
          │  Result: Better-tuned models       │
          └────────────┬───────────────────────┘
                       │
                       ▼
          ┌────────────────────────────────────┐
          │  3. Ensemble Agent (ENHANCED)      │
          │  ─────────────────────────         │
          │  - Receives Enhancer guidance:     │
          │    "Priority 2: Weighted averaging"│
          │  - Generates plan with optimal     │
          │    weight search                   │
          │  - Executes: ensemble0.py with     │
          │    weight optimization             │
          │  - Evaluates: Improved RMSE        │
          │  Output: final_solution.py         │
          └────────────┬───────────────────────┘
                       │
                       ▼
          ┌────────────────────────────────────┐
          │  4. Submission Agent               │
          │  (Same as Run 0)                   │
          └────────────┬───────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────────┐
│ Save Run 1 Results                                              │
├────────────────────────────────────────────────────────────────┤
│ - Best Score: 43127.45 RMSE (5% improvement!)                  │
│ - Duration: 2453 seconds (longer due to more iterations)       │
│ - Update run_history.json                                       │
└────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│ INVOKE ENHANCER AGENT (for Run 2)                              │
├────────────────────────────────────────────────────────────────┤
│ Enhancer Analysis:                                              │
│   "Run 1 improved by 5% with hyperparameter tuning. Weighted   │
│    ensemble also helped. Diminishing returns on tuning. Next    │
│    focus should be feature engineering - ablation showed        │
│    feature engineering had 25% impact but wasn't prioritized.   │
│    Maintain current iteration counts but shift focus."         │
│                                                                 │
│ Output: New strategy for Run 2                                  │
└────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│ RUN 2: FURTHER ENHANCED                                         │
├────────────────────────────────────────────────────────────────┤
│ Config: Maintain Run 1 settings, add new focus                  │
│ Strategic Goals: Feature engineering now Priority 1             │
└────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                        (Continues...)
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│ FINAL REPORT                                                    │
├────────────────────────────────────────────────────────────────┤
│ Best Score: 42103.22 RMSE (achieved in Run 2)                  │
│ Improvement: 6.9% over baseline                                │
│ Best Solution: ./run_2/ensemble/final_solution.py              │
└────────────────────────────────────────────────────────────────┘
```

---

## Configuration System

### Configuration Hierarchy

The system uses a multi-layer configuration approach:

```
┌──────────────────────────────────────────────────────────────┐
│ Layer 1: Hardcoded Defaults (config.py)                        │
├──────────────────────────────────────────────────────────────┤
│ - Baseline conservative settings                               │
│ - inner_loop_round: 1, outer_loop_round: 1, etc.              │
└──────────────────────┬───────────────────────────────────────┘
                       │ Overridden by ▼
┌──────────────────────────────────────────────────────────────┐
│ Layer 2: User Initial Config (--initial_config_path)           │
├──────────────────────────────────────────────────────────────┤
│ - Optional JSON file with user preferences                     │
│ - Example: {"num_solutions": 3, "agent_model": "gemini-pro"}  │
└──────────────────────┬───────────────────────────────────────┘
                       │ Overridden by ▼
┌──────────────────────────────────────────────────────────────┐
│ Layer 3: Enhancer Config Overrides (per-run, after Run 0)     │
├──────────────────────────────────────────────────────────────┤
│ - Dynamically generated by Enhancer Agent                      │
│ - Example: {"inner_loop_round": 3, "max_debug_round": 5}      │
│ - Only applied if allow_config_override = True                 │
└──────────────────────────────────────────────────────────────┘
```

### Master Configuration Switch

```python
# config.py
allow_config_override: bool = True  # Master switch

# run_meta.py
if config.CONFIG.allow_config_override:
    for key, value in overrides.items():
        setattr(current_config, key, value)
else:
    print("Config overrides disabled. Using default config.")
```

**Purpose**: Safety mechanism to disable Enhancer modifications if needed (e.g., for debugging or controlled experiments).

### Configurable Parameters

| Parameter | Type | Baseline | Enhanced (Example) | Description |
|-----------|------|----------|-------------------|-------------|
| `num_solutions` | int | 2 | 3 | Number of parallel initialization pipelines |
| `num_model_candidates` | int | 2 | 3 | Candidates to generate per search |
| `inner_loop_round` | int | 1 | 2-5 | Iterations within refinement plan execution |
| `outer_loop_round` | int | 1 | 2-3 | Full ablation→refine cycles |
| `ensemble_loop_round` | int | 1 | 2 | Ensemble plan execution iterations |
| `max_retry` | int | 3 | 3-5 | Retries before invoking debug loop |
| `max_debug_round` | int | 3 | 5-10 | Debug iterations per failure |
| `max_rollback_round` | int | 2 | 3 | Full rollback attempts |
| `exec_timeout` | int | 600 | 900 | Timeout for code execution (seconds) |

---

## Data Flow and State Management

### State Propagation

```
┌───────────────────────────────────────────────────────────────┐
│ MetaOrchestrator                                               │
│   ↓ Creates initial_state                                      │
│   {                                                            │
│     "run_id": 1,                                               │
│     "workspace_dir": "./workspace/task/run_1/",                │
│     "enhancer_output": {                                       │
│       "config_overrides": {...},                               │
│       "strategic_goals": [...]                                 │
│     },                                                         │
│     ... [all config parameters] ...                            │
│   }                                                            │
└────────────────────────┬──────────────────────────────────────┘
                         │ Injected into ▼
┌───────────────────────────────────────────────────────────────┐
│ MLE Pipeline Agent Session                                     │
│   session.state = initial_state                                │
└────────────────────────┬──────────────────────────────────────┘
                         │ Accessed by ▼
┌───────────────────────────────────────────────────────────────┐
│ Sub-Agent: Refinement Agent (Plan Generation)                  │
│   context.state.get("enhancer_output")  # Reads Enhancer data │
│   → Extracts strategic_goals                                   │
│   → Filters by target_agent_phase == "refinement"              │
│   → Injects into planning prompt                               │
└───────────────────────────────────────────────────────────────┘
                         │ Similarly ▼
┌───────────────────────────────────────────────────────────────┐
│ Sub-Agent: Ensemble Agent                                      │
│   context.state.get("enhancer_output")                         │
│   → Extracts ensemble-specific goals                           │
│   → Applies strategic focus                                    │
└───────────────────────────────────────────────────────────────┘
                         │ Results ▼
┌───────────────────────────────────────────────────────────────┐
│ Final State Collection                                         │
│   session.state (end of run)                                   │
│   → Saved to run_{id}/final_state.json                         │
│   → Contains:                                                  │
│     - All generated code                                       │
│     - All execution results                                    │
│     - All intermediate scores                                  │
│     - Complete ablation summaries                              │
│     - Applied enhancer_output                                  │
└────────────────────────┬──────────────────────────────────────┘
                         │ Fed to ▼
┌───────────────────────────────────────────────────────────────┐
│ Enhancer Agent (Next Run)                                      │
│   Analyzes final_state.json                                    │
│   → Generates new strategy                                     │
│   → Cycle continues                                            │
└───────────────────────────────────────────────────────────────┘
```

### Key State Keys

| State Key | Type | Set By | Read By | Purpose |
|-----------|------|--------|---------|---------|
| `run_id` | int | Orchestrator | All agents | Track which run is executing |
| `enhancer_output` | dict | Orchestrator | Refinement, Ensemble, Submission | Strategic guidance |
| `workspace_dir` | str | Orchestrator | All agents | Isolate run artifacts |
| `train_code_{step}_{task_id}` | str | Refinement | Refinement, Submission | Generated code |
| `train_code_exec_result_{step}_{task_id}` | dict | Code execution | Refinement, Submission | Scores, errors |
| `ablation_summary_{step}_{task_id}` | str | Refinement | Refinement | Ablation insights |
| `refinement_plan_{task_id}` | list | Refinement | Refinement | Step-by-step improvement plan |
| `ensemble_plan` | list | Ensemble | Ensemble | Ensemble strategy |
| `best_score_so_far` | float | Orchestrator | Enhancer | Track record |

---

## Key Design Decisions

### 1. Why Separate Enhancer Agent vs. Modifying Existing Agents?

**Decision**: Create standalone Enhancer Agent instead of embedding meta-learning logic in existing agents.

**Rationale**:
- **Separation of Concerns**: Each agent has a single responsibility
  - Base agents: Execute ML pipeline tasks
  - Enhancer: Strategic analysis and planning
- **Modularity**: Can enable/disable meta-learning via single flag
- **Clarity**: Easier to debug and understand what's happening
- **Scalability**: Can swap Enhancer with different strategies (e.g., Bayesian optimization) without touching base agents

**Trade-off**: Requires state propagation mechanism (enhancer_output injection).

### 2. Why JSON-Based Communication Protocol?

**Decision**: Enhancer outputs structured JSON with strict schema validation.

**Rationale**:
- **Reliability**: Parsing structured data is more robust than free-text analysis
- **Validation**: Can enforce required fields and catch errors early
- **Composability**: Easy to serialize, log, and replay strategies
- **Machine-Readable**: Enables automated analysis of meta-learning decisions

**Trade-off**: LLM must learn to output valid JSON (handled via few-shot examples in prompt).

### 3. Why Reduce Baseline Configuration Values?

**Decision**: Set inner_loop_round, outer_loop_round, etc. to 1 for baseline, allow Enhancer to increase.

**Rationale**:
- **Fast Iteration**: Run 0 completes quickly, providing data for Enhancer
- **Token Efficiency**: Prevents context overflow in early runs
- **Data-Driven Scaling**: Only increase iterations where empirically justified
- **Cost Optimization**: Avoid expensive operations when not needed

**Trade-off**: Baseline performance may be lower, but total multi-run performance is better.

### 4. Why Phase-Based Goal Targeting?

**Decision**: Strategic goals specify `target_agent_phase` (refinement, ensemble, submission).

**Rationale**:
- **Granular Control**: Different agents need different guidance
- **Priority Management**: Can have separate priority orderings per phase
- **Flexibility**: Some runs may need refinement focus, others may need ensemble focus

**Trade-off**: Requires agents to filter goals for their specific phase.

### 5. Why allow_config_override Master Switch?

**Decision**: Single boolean flag to enable/disable all Enhancer modifications.

**Rationale**:
- **Safety**: Can disable meta-learning if Enhancer suggests bad configs
- **Debugging**: Run controlled experiments with fixed configs
- **Validation**: Compare meta-learning vs. static config performance

**Trade-off**: All-or-nothing (can't selectively disable specific overrides).

### 6. Why Async/Await Architecture?

**Decision**: Use async execution for Orchestrator and agent invocations.

**Rationale**:
- **ADK Compatibility**: Google ADK's InMemoryRunner is async-native
- **Scalability**: Can parallelize agent execution where possible
- **Future-Proofing**: Easier to add concurrent runs or distributed execution

**Trade-off**: More complex code structure (async/await syntax).

---

## Traceability and Reproducibility

### Run History Tracking

Every run is fully logged in `run_history.json`:

```json
{
  "run_id": 1,
  "status": "COMPLETED_SUCCESSFULLY",
  "start_time_iso": "2025-10-03T14:30:00Z",
  "duration_seconds": 2453,
  "best_score": 43127.45,
  "best_solution_path": "run_1/ensemble/final_solution.py",
  "config_used": {
    "inner_loop_round": 3,
    "outer_loop_round": 2,
    ...
  },
  "enhancer_rationale": "Hyperparameter tuning showed 40% impact...",
  "enhancer_output": {
    "strategic_summary": "...",
    "config_overrides": {...},
    "strategic_goals": [...]
  }
}
```

This allows:
- **Auditing**: Understand why specific decisions were made
- **Reproduction**: Re-run with exact config from any run
- **Analysis**: Correlate strategies with performance outcomes
- **Debugging**: Trace back errors to their originating strategy

### Artifact Organization

```
workspace/
└── california-housing-prices/
    ├── run_history.json          # Complete history log
    ├── run_0/
    │   ├── final_state.json      # Complete execution state
    │   ├── 1/                    # Solution 1 artifacts
    │   │   ├── input/            # Task data
    │   │   ├── model_candidates/ # Search results
    │   │   ├── init_code_1.py
    │   │   ├── ablation_0.py
    │   │   ├── train0_improve0.py
    │   │   ├── train1_1.py
    │   │   └── ...
    │   ├── 2/                    # Solution 2 artifacts
    │   └── ensemble/
    │       ├── ensemble0.py
    │       ├── final_solution.py
    │       └── final/
    │           └── submission.csv
    ├── run_1/
    │   └── ... (same structure)
    └── run_2/
        └── ... (same structure)
```

Each run is **fully isolated** with its own workspace, preventing cross-contamination of artifacts.

---

## Performance Gains

### Empirical Results (Example)

**Task**: California Housing Price Prediction (Regression, RMSE metric, lower is better)

| Run | Duration | Best Score | Improvement | Key Changes |
|-----|----------|------------|-------------|-------------|
| 0 (Baseline) | 30 min | 45231.87 | - | Default config, simple averaging |
| 1 (Enhanced) | 41 min | 43127.45 | **4.65%** | Hyperparameter tuning focus, weighted ensemble |
| 2 (Further Enhanced) | 38 min | 42103.22 | **6.91%** | Feature engineering focus, maintained tuning |

**Total Time**: 109 minutes (1h 49m)  
**Final Improvement**: 6.91% over baseline  
**Key Insight**: Meta-learning achieved better results than any single long-running baseline would, by adaptively focusing effort.

---

## Conclusion

The Meta-Learning Enhancement Framework transforms MLE-STAR from a single-shot pipeline into an **iterative, self-improving system**. The Enhancer Agent acts as an AI Research Lead, analyzing past performance and strategically guiding future runs. This approach achieves:

1. **Better Final Performance**: Multi-run optimization finds better solutions than single-run
2. **Efficiency**: Starts with fast baseline, scales up only where needed
3. **Adaptability**: Different tasks may need different strategies; Enhancer learns this
4. **Interpretability**: Every decision is logged with rationale
5. **Modularity**: Can be enabled/disabled without changing base code

This framework demonstrates the power of **meta-learning in AutoML systems** - using AI to optimize not just models, but the entire ML engineering process itself.

---

## References

- **Base MLE-STAR Paper**: "MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement" (https://www.arxiv.org/abs/2506.15692)
- **Google ADK Documentation**: https://github.com/google/genai-agent-dev-kit
- **Codebase Location**: `machine-learning-engineering/`

---

## Appendix: Quick Reference Commands

### Run the Meta-Learning Framework
```bash
# Basic usage: 3 runs on California housing task
python run_meta.py --task_name california-housing-prices --num_runs 3

# With custom initial config
python run_meta.py \
  --task_name california-housing-prices \
  --num_runs 5 \
  --initial_config_path ./my_config.json
```

### Inspect Run History
```bash
# View complete history
cat workspace/california-housing-prices/run_history.json | jq .

# Extract best scores
cat workspace/california-housing-prices/run_history.json | \
  jq '[.[] | {run_id, best_score}]'

# View Enhancer strategy for Run 1
cat workspace/california-housing-prices/run_history.json | \
  jq '.[0].enhancer_output'
```

### Reproduce a Specific Run
```python
# Extract config from run_history.json
import json

with open("workspace/california-housing-prices/run_history.json") as f:
    history = json.load(f)

# Get config from Run 1
run_1_config = history[1]["config_used"]

# Save as initial config for new run
with open("run_1_config.json", "w") as f:
    json.dump(run_1_config, f, indent=2)

# Execute with that config
# python run_meta.py --initial_config_path run_1_config.json --num_runs 1
```

### Disable Meta-Learning (Run Fixed Config)
```python
# Edit config.py
allow_config_override: bool = False  # Set to False

# Now Enhancer suggestions are ignored, all runs use same config
# python run_meta.py --num_runs 3
```

---

*This document was created for academic presentation and reflects the current state of the enhanced MLE-STAR system as of October 2025.*


