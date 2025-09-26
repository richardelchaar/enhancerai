# Machine Learning Engineering Agent — Structural Revamp and Enforcement Guide

This document captures the architectural upgrades and behavioural changes introduced in the "Dynamic Enforcement" refactor.  It is intended for engineers who need to understand how the new multi-run workflow is assembled, what kinds of guidance are enforced automatically, and how to extend or operate the system safely.

---

## 1. Goals of This Refactor

1. **Rebuild agents per run.**  The previous design instantiated the initialization, refinement, ensemble, and submission agents exactly once during module import.  Configuration updates produced by the enhancement plan therefore had no effect on subsequent runs.  We now rebuild the entire agent hierarchy on every run, ensuring that overrides such as `num_solutions = 3` or additional ensemble passes actually materialise.
2. **Enforce guidance.**  Enhancement plans are no longer advisory.  A new validator inspects every generated script and raises hard failures when mandatory directives (e.g. “use RandomizedSearchCV” or “stop using `fillna(..., inplace=True)`”) are ignored.
3. **Adapt budgets automatically.**  When the enhancement plan requests heavier work—hyperparameter searches, CatBoost integration, or multi-layer stacking—we automatically raise the relevant loop counts and execution timeouts so that the next run has the resources it needs.
4. **Document the new contract.**  This README explains how the new pieces fit together and what developers must do when introducing new guidance keywords or sub-agents.

---

## 2. Dynamic Agent Construction

### 2.1 Builder Functions

Each sub-agent module now exports a `build_agent()` factory:

- `machine_learning_engineering/sub_agents/initialization/agent.py:build_agent`
- `machine_learning_engineering/sub_agents/refinement/agent.py:build_agent`
- `machine_learning_engineering/sub_agents/ensemble/agent.py:build_agent`
- `machine_learning_engineering/sub_agents/submission/agent.py:build_agent`

These functions consume the current `config.CONFIG` values at call-time, meaning that changes to `num_solutions`, `num_model_candidates`, loop budgets, or optional checkers all take effect immediately.  The helper callbacks (workspace creation, leak checking, ranking, etc.) are unchanged, so downstream behaviour remains familiar—each call simply receives fresh loop agents sized to the most recent configuration.

### 2.2 Root Agent Factory

`machine_learning_engineering/agent.py` introduces two utilities:

- `build_pipeline_agent()` rebuilds the sequential agent that chains initialization → refinement → ensemble → submission.
- `build_root_agent()` wraps the pipeline in the front-door agent, accepting an optional `agent_model` so callers can override the LLM per run.

For single-run usage the module still exposes `root_agent = build_root_agent()`, preserving backwards compatibility with notebooks or quick demos.

### 2.3 Multi-Run Integration

`machine_learning_engineering/multi_run.py` now calls `build_root_agent(run_config.agent_model)` for every run.  The orchestrator no longer mutates a shared agent tree; instead, it constructs a new one after applying config overrides and before executing the pipeline.  This guarantees that:

- Added solution tracks really exist (e.g., `num_solutions = 3`).
- Additional model candidates are requested and evaluated.
- Ensemble loop counts, submission stages, and data-use checkers reflect the newest instructions.

---

## 3. Guidance Enforcement

### 3.1 Validator Overview

A new module, `machine_learning_engineering/shared_libraries/guidance_validator.py`, defines `validate_code(context, code)`.  `machine_learning_engineering/shared_libraries/code_util.py:evaluate_code` calls this validator every time an executable script is produced (except for ablation artefacts).  If violations are detected, the execution result is marked as failed and the score is set to a worst-case sentinel.  The validator currently enforces the following rules:

| Guidance Signal | Enforced Requirement |
|-----------------|----------------------|
| “RandomizedSearchCV”, “GridSearchCV”, “Optuna”, or “Hyperopt” mentioned | The generated code must use at least one of these tools. |
| “CatBoost” mentioned | The code must reference CatBoost (e.g., `CatBoostRegressor`). |
| “inplace=True” mentioned | No generated code may contain `.fillna(..., inplace=True)`. |
| Ensemble guidance mentions “regression” or “RMSE” | The ensemble script must avoid classifiers / ROC AUC tooling, must contain regression metrics or regressors, and must not fabricate synthetic data. |

Violations are appended to the execution stderr so that the LLM receives explicit feedback during the retry loop.

### 3.2 Adding New Rules

To add your own directives:

1. Update `guidance_validator.validate_code` with the new pattern checks.
2. Update `README3.md` (this document) to describe the rule for future developers.
3. Optionally, extend `_apply_implicit_overrides` (see §4) if the directive implies new runtime budgets.

---

## 4. Adaptive Overrides

Enhancement plans often ask for more substantial exploration but forget to raise the relevant loop counts.  `_apply_implicit_overrides(plan, overrides)` in `machine_learning_engineering/multi_run.py` now inspects the free-form notes and guarantees minimum runtime budgets:

- **Hyperparameter search keywords** ⇒ `inner_loop_round ≥ 2`, `outer_loop_round ≥ 2`, `exec_timeout ≥ 1200` seconds.
- **CatBoost mentions** ⇒ `num_solutions ≥ 3`, `num_model_candidates ≥ 3`.
- **Multi-layer stacking mentions** ⇒ `ensemble_loop_round ≥ 2`.

These adjustments are applied *after* explicit `config_updates` from the enhancement plan, so hard overrides always win but never prevent necessary headroom.

---

## 5. Execution Flow Recap

1. **Configuration load:** `multi_run.py` reads the multi-run JSON, applies base overrides, and prepares per-run workspaces.
2. **Per-run setup:** `config_module.set_config` stores the current run’s settings.  `build_root_agent` constructs a fresh agent tree using those settings.
3. **Pipeline execution:** `_execute_pipeline` runs the new agent via `InMemoryRunner`.
4. **State capture:** `agent.save_state` persists `final_state.json` for downstream inspection.
5. **Plan generation:** `generate_enhancement_plan` produces the next guidance JSON.
6. **Adaptive overrides:** Explicit `config_updates` are applied, then `_apply_implicit_overrides` injects any derived overrides.
7. **Guidance enforcement:** Every generated script of the following run is validated against the recorded guidance before it can advance.

---

## 6. Developer Checklist

- **Adding new sub-agents:** expose a `build_agent()` factory and have `machine_learning_engineering/agent.py` call it inside `build_pipeline_agent()`.
- **Extending guidance:** update the validator and (optionally) the implicit override helper.
- **Testing directives:** run the multi-run orchestrator with a guidance file that mentions your new keywords; the validator will fail the step if the code ignores them.
- **Single-run experiments:** you can still import `root_agent` directly, but for tests that mutate configuration call `build_root_agent()` yourself to obtain fresh agents.

---

## 7. File Reference Summary

- Dynamic builders: `machine_learning_engineering/sub_agents/*/agent.py`
- Root agent factory: `machine_learning_engineering/agent.py`
- Orchestrator overrides: `machine_learning_engineering/multi_run.py`
- Guidance validator: `machine_learning_engineering/shared_libraries/guidance_validator.py`
- Code evaluation hook: `machine_learning_engineering/shared_libraries/code_util.py`

---

## 8. Next Steps

- Layer more semantic understanding into `guidance_validator` (e.g., verifying that CatBoost predictions feed the ensemble rather than being defined and ignored).
- Track directive satisfaction in the state itself so the enhancer can see which items were completed versus failed across runs.
- Expand `_apply_implicit_overrides` to capture execution-time hints (e.g., toggling leakage checkers) directly from the plan.

This refactor ensures that enhancement plans now have real authority: if the plan says “run RandomizedSearchCV,” the next run will fail until the agent actually ships a script that does it.  Use the validator and the builder pattern as you extend the system so future directives remain enforceable.
