"""Demonstration of Machine Learning Engineering Agent using Agent Development Kit"""

import os
import json
from typing import Optional
from google.genai import types
from google.adk.agents import callback_context as callback_context_module

from google.adk import agents
from machine_learning_engineering.sub_agents.initialization import agent as initialization_agent_module
from machine_learning_engineering.sub_agents.refinement import agent as refinement_agent_module
from machine_learning_engineering.sub_agents.ensemble import agent as ensemble_agent_module
from machine_learning_engineering.sub_agents.submission import agent as submission_agent_module

from machine_learning_engineering import prompt
from machine_learning_engineering.shared_libraries import config


def save_state(
    callback_context: callback_context_module.CallbackContext
) -> Optional[types.Content]:
    """Prints the current state of the callback context."""
    workspace_dir = callback_context.state.get("workspace_dir", "")
    task_name = callback_context.state.get("task_name", "")
    run_cwd = os.path.join(workspace_dir, task_name)
    with open(os.path.join(run_cwd, "final_state.json"), "w") as f:
        json.dump(callback_context.state.to_dict(), f, indent=2)
    return None




def build_pipeline_agent() -> agents.SequentialAgent:
    """Builds the sequential pipeline with fresh sub-agents."""

    initialization_agent = initialization_agent_module.build_agent()
    refinement_agent = refinement_agent_module.build_agent()
    ensemble_agent = ensemble_agent_module.build_agent()
    submission_agent = submission_agent_module.build_agent()

    return agents.SequentialAgent(
        name="mle_pipeline_agent",
        sub_agents=[
            initialization_agent,
            refinement_agent,
            ensemble_agent,
            submission_agent,
        ],
        description="Executes a sequence of sub-agents for solving the MLE task.",
        after_agent_callback=save_state,
    )


def build_root_agent(agent_model: Optional[str] = None) -> agents.Agent:
    """Builds the root agent, recreating the entire sub-agent graph."""

    pipeline = build_pipeline_agent()
    model_to_use = agent_model or os.getenv("ROOT_AGENT_MODEL", config.CONFIG.agent_model)
    return agents.Agent(
        model=model_to_use,
        name="mle_frontdoor_agent",
        instruction=prompt.FRONTDOOR_INSTRUCTION,
        global_instruction=prompt.SYSTEM_INSTRUCTION,
        sub_agents=[pipeline],
        generate_content_config=types.GenerateContentConfig(temperature=0.01),
    )


# Maintain the legacy module-level root agent for single-run scenarios.
root_agent = build_root_agent()
