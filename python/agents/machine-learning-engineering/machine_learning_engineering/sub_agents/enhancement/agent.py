"""Enhancement agent that generates guidance for subsequent runs."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional

from google.genai import types
from google.adk import agents
from google.adk.runners import InMemoryRunner

from machine_learning_engineering.shared_libraries import config

from . import prompt


_DEFAULT_PLAN: Dict[str, Any] = {
    "global_notes": "",
    "initialization": "",
    "refinement": "",
    "ensemble": "",
    "submission": "",
    "config_updates": [],
}


def _build_user_prompt(
    final_state: Dict[str, Any],
    previous_guidance: Optional[Dict[str, Any]],
    current_run: int,
    total_runs: int,
) -> str:
    """Formats the user prompt supplied to the enhancement agent."""

    remaining_runs = max(total_runs - current_run, 0)
    previous_guidance_str = (
        json.dumps(previous_guidance, indent=2, ensure_ascii=False)
        if previous_guidance
        else "{}"
    )
    final_state_str = json.dumps(final_state, indent=2, ensure_ascii=False)
    return prompt.ENHANCEMENT_USER_PROMPT.format(
        current_run=current_run,
        total_runs=total_runs,
        remaining_runs=remaining_runs,
        previous_guidance=previous_guidance_str,
        final_state=final_state_str,
    )


async def _run_enhancement_agent_async(user_prompt: str, model_name: str) -> str:
    """Runs the enhancement agent asynchronously and returns the raw text response."""

    enhancement_agent = agents.Agent(
        model=model_name,
        name="enhancement_agent",
        description="Generate improvement guidance for the next pipeline run.",
        instruction=prompt.ENHANCEMENT_SYSTEM_INSTR,
        include_contents="none",
        generate_content_config=types.GenerateContentConfig(temperature=0.2),
    )
    runner = InMemoryRunner(
        agent=enhancement_agent,
        app_name="machine-learning-enhancement",
    )
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="enhancement",
    )
    message = types.Content(parts=[types.Part(text=user_prompt)], role="user")
    response_text = ""
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    response_text += part.text
    await runner.session_service.delete_session(
        app_name=runner.app_name,
        user_id=session.user_id,
        session_id=session.id,
    )
    return response_text.strip()


def _extract_plan(response_text: str) -> Dict[str, Any]:
    """Extracts the guidance JSON from the model response."""

    if not response_text:
        return dict(_DEFAULT_PLAN)
    start_idx = response_text.find("{")
    end_idx = response_text.rfind("}")
    if start_idx == -1 or end_idx == -1:
        return dict(_DEFAULT_PLAN)
    candidate = response_text[start_idx : end_idx + 1]
    try:
        loaded = json.loads(candidate)
    except json.JSONDecodeError:
        return dict(_DEFAULT_PLAN)
    plan = dict(_DEFAULT_PLAN)
    if isinstance(loaded, dict):
        for key in plan:
            if key in loaded:
                plan[key] = loaded[key]
    # Normalise config_updates to a list of dicts with key/value pairs.
    updates = plan.get("config_updates", [])
    if not isinstance(updates, list):
        plan["config_updates"] = []
    else:
        normalised_updates = []
        for item in updates:
            if isinstance(item, dict) and "key" in item and "value" in item:
                normalised_updates.append({
                    "key": item["key"],
                    "value": item["value"],
                })
        plan["config_updates"] = normalised_updates
    return plan


def generate_enhancement_plan(
    final_state_path: str,
    output_path: str,
    current_run: int,
    total_runs: int,
    previous_guidance_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Generates enhancement guidance for the next run and writes it to disk."""

    if not os.path.exists(final_state_path):
        raise FileNotFoundError(f"Final state file not found: {final_state_path}")
    with open(final_state_path, "r", encoding="utf-8") as f:
        final_state = json.load(f)
    previous_guidance: Optional[Dict[str, Any]] = None
    if previous_guidance_path and os.path.exists(previous_guidance_path):
        try:
            with open(previous_guidance_path, "r", encoding="utf-8") as f:
                previous_guidance = json.load(f)
        except json.JSONDecodeError:
            previous_guidance = None
    user_prompt = _build_user_prompt(
        final_state=final_state,
        previous_guidance=previous_guidance,
        current_run=current_run,
        total_runs=total_runs,
    )
    model_to_use = model_name or config.CONFIG.agent_model
    response_text = asyncio.run(
        _run_enhancement_agent_async(user_prompt=user_prompt, model_name=model_to_use)
    )
    plan = _extract_plan(response_text)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    return plan


__all__ = ["generate_enhancement_plan"]
