"""Multi-run orchestrator for the Machine Learning Engineering agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any, Dict, Iterable, List, Optional

from google.genai import types
from google.adk.runners import InMemoryRunner
from dotenv import load_dotenv

from machine_learning_engineering.agent import build_root_agent
from machine_learning_engineering.shared_libraries import config as config_module
from machine_learning_engineering.sub_agents.enhancement import (
    generate_enhancement_plan,
)


def _load_multi_run_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)




def _apply_implicit_overrides(plan: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Derives additional config overrides implied by textual guidance."""

    text_parts = [
        str(plan.get("global_notes", "")),
        str(plan.get("initialization", "")),
        str(plan.get("refinement", "")),
        str(plan.get("ensemble", "")),
        str(plan.get("submission", "")),
    ]
    combined = " ".join(text_parts).lower()

    def ensure_minimum(key: str, value: int) -> None:
        current = overrides.get(key)
        if current is None or int(current) < value:
            overrides[key] = value

    if any(keyword in combined for keyword in ["randomizedsearchcv", "gridsearchcv", "optuna", "hyperopt"]):
        ensure_minimum("inner_loop_round", 2)
        ensure_minimum("outer_loop_round", 2)
        ensure_minimum("exec_timeout", 1200)

    if "catboost" in combined:
        ensure_minimum("num_solutions", 3)
        ensure_minimum("num_model_candidates", 3)

    if "multi-layer stacking" in combined or "multi layer stacking" in combined:
        ensure_minimum("ensemble_loop_round", 2)

def _build_run_workspace(base_workspace: str, run_idx: int) -> str:
    base_workspace = os.path.normpath(base_workspace)
    run_dir = os.path.join(base_workspace, "runs", f"run_{run_idx:02d}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


async def _execute_pipeline(agent: Any, messages: Iterable[str]) -> List[str]:
    """Executes the supplied root agent with the provided user messages."""

    runner = InMemoryRunner(
        agent=agent,
        app_name="machine-learning-engineering",
    )
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="multi-runner",
    )
    responses: List[str] = []
    for message_text in messages:
        content = types.Content(parts=[types.Part(text=message_text)], role="user")
        last_response = ""
        async for event in runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        last_response += part.text
        responses.append(last_response.strip())
    await runner.session_service.delete_session(
        app_name=runner.app_name,
        user_id=session.user_id,
        session_id=session.id,
    )
    return responses


def _extract_best_score(final_state: Dict[str, Any]) -> Optional[float]:
    scores: List[float] = []
    for key, value in final_state.items():
        if key.startswith("best_score") and isinstance(value, (int, float)):
            scores.append(float(value))
    if scores:
        return min(scores)
    return None


def run_multi_pipeline(config_path: str) -> Dict[str, Any]:
    """Runs the pipeline multiple times according to the configuration."""

    load_dotenv()
    multi_run_cfg = _load_multi_run_config(config_path)
    base_overrides = multi_run_cfg.get("base_config", {})
    num_runs = int(multi_run_cfg.get("num_runs", base_overrides.get("num_runs", 1)))
    if num_runs < 1:
        raise ValueError("num_runs must be at least 1")
    conversation: Iterable[str]
    if "conversation" in multi_run_cfg:
        conversation = multi_run_cfg["conversation"]
    else:
        conversation = [
            multi_run_cfg.get(
                "run_prompt",
                "Please execute the configured machine learning pipeline for the provided task.",
            )
        ]
    if isinstance(conversation, str):
        conversation = [conversation]
    conversation = [str(message) for message in conversation]
    base_config = config_module.load_config_from_mapping(base_overrides)
    base_workspace = base_config.workspace_dir
    history: List[Dict[str, Any]] = []
    previous_plan_path = multi_run_cfg.get("initial_guidance_path", "")
    adaptive_overrides = dict(base_overrides)
    for run_idx in range(1, num_runs + 1):
        run_workspace = _build_run_workspace(base_workspace, run_idx)
        run_overrides = dict(adaptive_overrides)
        run_overrides["workspace_dir"] = run_workspace
        run_overrides["run_guidance_path"] = previous_plan_path
        run_overrides["run_id"] = run_idx
        run_overrides["num_runs"] = num_runs
        run_config = config_module.load_config_from_mapping(run_overrides)
        config_module.set_config(run_config)
        current_root_agent = build_root_agent(run_config.agent_model)
        responses = asyncio.run(_execute_pipeline(current_root_agent, conversation))
        final_state_dir = os.path.join(run_workspace, run_config.task_name)
        final_state_path = os.path.join(final_state_dir, "final_state.json")
        if not os.path.exists(final_state_path):
            raise FileNotFoundError(
                f"Run {run_idx}: final_state.json not found at {final_state_path}"
            )
        with open(final_state_path, "r", encoding="utf-8") as f:
            final_state = json.load(f)
        best_score = _extract_best_score(final_state)
        plan_output_path = os.path.join(run_workspace, "enhancement_plan.json")
        enhancement_plan = generate_enhancement_plan(
            final_state_path=final_state_path,
            output_path=plan_output_path,
            current_run=run_idx,
            total_runs=num_runs,
            previous_guidance_path=previous_plan_path or None,
            model_name=run_config.agent_model,
        )
        for update in enhancement_plan.get("config_updates", []):
            key = update.get("key")
            if key:
                adaptive_overrides[key] = update.get("value")
        _apply_implicit_overrides(enhancement_plan, adaptive_overrides)
        history.append(
            {
                "run": run_idx,
                "workspace": run_workspace,
                "final_state_path": final_state_path,
                "enhancement_plan_path": plan_output_path,
                "best_score": best_score,
                "responses": responses,
            }
        )
        previous_plan_path = plan_output_path
    history_path = os.path.join(base_workspace, "runs", "run_history.json")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    return {
        "history_path": history_path,
        "runs": history,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-run pipeline orchestrator")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the multi-run configuration JSON file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = run_multi_pipeline(args.config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
