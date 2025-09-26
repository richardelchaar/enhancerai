# Multi‑Run Machine Learning Engineering (MLE‑STAR) Strategy

This document explains, in depth, how the multi‑run strategy works, why it’s designed this way, how enhancement plans are produced and enforced, where outputs live, and how the main pieces of code fit together. A senior engineer should be able to rebuild the system from this spec.

## Executive Summary

- A standard MLE‑STAR pipeline (initialization → refinement → ensemble → submission) runs once to produce a solid baseline solution.
- A separate orchestrator then runs the entire pipeline N times. Between runs, an enhancement sub‑agent reads the previous run’s `final_state.json` and emits a structured `enhancement_plan.json` with actionable guidance and optional config updates.
- The next run consumes that plan and enforces the guidance as mandatory directives across all sub‑agents’ prompts. This closes the loop and drives targeted improvements across iterations.

## Architecture

### Root pipeline (ADK)

- Entry point wires the four phases and persists state:
  - `machine_learning_engineering/agent.py:31` builds the sequential pipeline.
  - `machine_learning_engineering/agent.py:19` saves `final_state.json` after the pipeline finishes.
  - `machine_learning_engineering/agent.py:44` defines `root_agent` with the global/system instructions.

### Sub‑agents (per phase)

- Initialization: retrieves model candidates, generates/evaluates initial codes, merges and ranks.
  - File: `machine_learning_engineering/sub_agents/initialization/agent.py:397` (agent graph)
  - Workspace setup: `create_workspace` (`:258`)
  - Task prep and guidance load: `prepare_task` (`:227`)
  - Prompt providers: model retrieval/eval/merge and data‑use checks (`:290`, `:319`, `:342`, `:373`)

- Refinement: ablation to identify impactful blocks, plan + targeted code‑block improvements, inner/outer loops.
  - File: `machine_learning_engineering/sub_agents/refinement/agent.py` (prompt providers at `:115`, `:149`, `:174`, `:207`)

- Ensemble: plans and implements ensembles over the latest solutions, then iteratively refines.
  - File: `machine_learning_engineering/sub_agents/ensemble/agent.py` (prompt providers near the top and reinforce guidance)

- Submission: adds test inference and writes submission.
  - File: `machine_learning_engineering/sub_agents/submission/agent.py:33`

### Debug/eval utilities (shared)

- Run‑and‑debug loop: wraps LLM codegen with execution, retry, rollback, and optional data‑leakage checks.
  - File: `machine_learning_engineering/shared_libraries/debug_util.py` (core build blocks `get_run_and_debug_agent` et al)

- Code execution + scoring: writes the generated Python file, executes it, parses “Final Validation Performance: <score>”, and records execution results into state keys per agent + suffix.
  - File: `machine_learning_engineering/shared_libraries/code_util.py:187` (`evaluate_code`)
  - Naming/suffixing rules for file names and state keys: `code_util.py:79`, `:114`

## Multi‑Run Orchestration

### Orchestrator

- File: `machine_learning_engineering/multi_run.py`
  - Loads `.env` early: `:91`.
  - Reads config JSON: `:22`.
  - Creates per‑run workspace dirs at `<workspace_dir>/runs/run_XX`: `:27`.
  - Ensures the requested model is set on every sub‑agent before each run: `_set_model_recursively` (`:34`) + call site (`:125`).
  - Executes the ADK pipeline (front‑door → full sub‑graph) and captures responses: `_execute_pipeline` (`:45`).
  - Resolves the previous run’s plan into an enhancement for the next run: `generate_enhancement_plan` call (`:137`).
  - Applies any `config_updates` returned by the plan to the overrides for subsequent runs: `:145`–`:149`.
  - Writes an overall `run_history.json` with paths, best scores, and responses: `:160`–`:167`.

### Run lifecycle

1. Orchestrator builds `runs/run_01/` and runs the entire agent pipeline.
2. Pipeline writes `runs/run_01/<task>/final_state.json` and artifacts.
3. Enhancer reads `final_state.json` and (optionally) the previous plan and outputs `runs/run_01/enhancement_plan.json`.
4. Orchestrator carries `run_01/enhancement_plan.json` forward via `run_guidance_path` and applies any config updates.
5. Steps (1)–(4) repeat for `run_02/`, `run_03/`, … until `num_runs`.

## Enhancement Sub‑Agent (the “new agent”)

### Purpose

Turn the previous run’s state into structured, actionable guidance for the next run, plus optional config overrides.

### Code path

- Agent and flow: `machine_learning_engineering/sub_agents/enhancement/agent.py`
  - Prompt assembly: `_build_user_prompt` (`:29`) formats a context with `current_run`, `remaining_runs`, `previous_guidance` and the full previous `final_state` as prettified JSON.
  - LLM call: `_run_enhancement_agent_async` (`:53`) creates a small ADK agent with `ENHANCEMENT_SYSTEM_INSTR` and executes it via `InMemoryRunner`.
  - Response parsing: `_extract_plan` (`:91`) strips any extra text, locates the outermost JSON braces, loads into a dict, and normalizes `config_updates` → a list of `{key, value}` pairs.
  - Entrypoint: `generate_enhancement_plan` (`:126`) reads `final_state.json`, includes previous guidance when available, runs the enhancement agent, writes `enhancement_plan.json`, and returns the dict.

- Prompts: `machine_learning_engineering/sub_agents/enhancement/prompt.py`
  - System instruction details the schema and insists on a single JSON object: `ENHANCEMENT_SYSTEM_INSTR`.
  - User context template injects run counters and the raw `final_state` JSON: `ENHANCEMENT_USER_PROMPT`.

### Enhancement schema

The enhancer always returns a single JSON object:

```json
{
  "global_notes": "...",
  "initialization": "...",
  "refinement": "...",
  "ensemble": "...",
  "submission": "...",
  "config_updates": [{"key": "exec_timeout", "value": 900}, ...]
}
```

## Translating Enhancements into the Next Run

### Loading guidance into state

- During the next run’s initialization, `prepare_task` copies the module config into state and loads the previous plan from `run_guidance_path` into `state["run_guidance"]`.
  - Code: `machine_learning_engineering/sub_agents/initialization/agent.py:227`.

### Guidance injection and enforcement

- Guidance text is appended to each prompt and converted into mandatory directives:
  - Utility to combine guidance and extract bullet points: `get_run_guidance` and `extract_guidance_requirements` in `machine_learning_engineering/shared_libraries/common_util.py:46`, `:70`.
  - Initialization prompt providers add “Previous run guidance (MANDATORY)” plus a “Mandatory directives” list, and require the agent to explain compliance: `initialization/agent.py:301`, `:325`, `:356`, `:380`.
  - Refinement prompt providers do the same across ablation, summary, plan, and implementation: `refinement/agent.py:136`, `:161`, `:194`, `:217`.
  - Ensemble prompt providers enforce the same: `ensemble/agent.py` (first three prompt providers near the top of the file—see updated blocks that mention “MANDATORY”).
  - Submission prompt provider requires documenting how the directives are fulfilled: `submission/agent.py:66`.

This turns soft guidance into explicit contractual requirements the LLM must address in its plans and code.

### Optional config updates

- The enhancer’s `config_updates` are merged into the overrides used for subsequent runs prior to pipeline execution: `machine_learning_engineering/multi_run.py:145`.

## Workspace & Outputs

- Root workspace: `machine_learning_engineering/workspace/`
- Per run: `workspace/runs/run_XX/`
  - Plan for next run: `runs/run_XX/enhancement_plan.json`
  - Consolidated history: `runs/run_history.json`
  - Task workspace: `runs/run_XX/<task>/`
    - Agent artifacts, generated code (`train0.py`, `train*_improve*.py`, `ensemble*.py`, `final_solution.py`)
    - `final_state.json` — the entire callback state tree including all scores, code blobs, and metadata
    - `ensemble/final/submission.csv` (when produced)

Note: When you re‑run the orchestrator with the same `num_runs`, the same `run_01`, `run_02`, … folders are reused; contents are overwritten.

## Data Flow and State Keys (abridged)

- Code blobs are stored in well‑defined keys:
  - `init_code_<taskId>_<modelId>` (e.g., `init_code_1_2`) for model eval
  - `train_code_<step>_<taskId>` for current best code at a refinement step
  - `train_code_improve_<innerIter>_<step>_<taskId>` for plan implementations
  - `ensemble_code_<iter>` for ensemble implementations
- Each has an execution result key with `_exec_result_…` suffix and a parsed `score` (lower is better if `state["lower"]` is true).
  - See mapping helpers in `machine_learning_engineering/shared_libraries/code_util.py:114` and `:142`.

## Why This Design

- Keep the one‑shot pipeline unmodified and composable; the orchestrator drives iterations externally without entangling per‑run state.
- Make improvements targeted by using the previous run’s full state; the enhancer’s JSON narrows attention to the most impactful changes.
- Enforce guidance in prompts so directives (e.g., “add hyperparameter tuning”) can’t be silently ignored by downstream agents.

## How to Run

1. Install deps (once):
   ```bash
   poetry install --with dev
   ```
2. Configure environment (Vertex or API key) and model (e.g., `ROOT_AGENT_MODEL=gemini-2.5-flash`). Use `.env` or `export`.
3. Prepare or tweak the multi‑run config; a sample lives at `machine_learning_engineering/multi_run_config.sample.json`.
4. Run:
   ```bash
   poetry run python -m machine_learning_engineering.multi_run \
     --config machine_learning_engineering/multi_run_config.sample.json
   ```

Outputs will appear under `workspace/runs/run_01/`, then `run_02/`, along with a `run_history.json` summary.

## Troubleshooting

- 404 model not found: ensure your Vertex region/project has the specified model; change `ROOT_AGENT_MODEL` or `base_config.agent_model` to an available ID.
- Guidance ignored: the enforcement hooks are already in place; verify that `enhancement_plan.json` contains the directives you expect and that they appear in prompts (search for “Previous run guidance (MANDATORY)” in generated logs or in `final_state.json` prompt echoes if you persist them).
- Slow/timeout: raise `exec_timeout` via `config_updates` in your enhancement plan or set a higher default in the base config.

## Extending the System

- Stronger enforcement: add a validation pass that asserts presence of expected code artifacts (e.g., `RandomizedSearchCV` in code) before accepting a plan as “satisfied”. If absent, keep iterating the inner loop.
- Richer enhancer: compute diffs between previous best code and new code; ask the enhancer to suggest precise patch hunks.
- New robustness checks: plug additional checkers into `debug_util.get_run_and_debug_agent` (e.g., fairness checks) in the same vein as the data‑leakage tool.

## Key Code References

- Root agent and pipeline wiring: `machine_learning_engineering/agent.py:31`, `:44`
- Orchestrator core: `machine_learning_engineering/multi_run.py:27`, `:34`, `:45`, `:88`, `:137`
- Enhancement agent: `machine_learning_engineering/sub_agents/enhancement/agent.py:29`, `:53`, `:91`, `:126`
- Prompt guidance utilities: `machine_learning_engineering/shared_libraries/common_util.py:46`, `:70`
- Initialization prompt providers (with guidance enforcement): `machine_learning_engineering/sub_agents/initialization/agent.py:290`, `:319`, `:342`, `:373`
- Refinement prompt providers (with guidance enforcement): `machine_learning_engineering/sub_agents/refinement/agent.py:115`, `:149`, `:174`, `:207`
- Ensemble prompt providers (with guidance enforcement): `machine_learning_engineering/sub_agents/ensemble/agent.py`
- Submission prompt provider (with guidance enforcement): `machine_learning_engineering/sub_agents/submission/agent.py:33`
- Run‑and‑debug core: `machine_learning_engineering/shared_libraries/debug_util.py`
- Code execution + scoring: `machine_learning_engineering/shared_libraries/code_util.py:187`

---

This document reflects the current implementation and conventions in this repo and is intended to be comprehensive enough to re‑implement the system from scratch.

