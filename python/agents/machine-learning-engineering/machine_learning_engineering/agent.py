"""Demonstration of Machine Learning Engineering Agent using Agent Development Kit"""

# Import compatibility fixes first
from machine_learning_engineering.shared_libraries import aiohttp_compat

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
    """Saves the final state of a run to its directory."""
    # MODIFICATION: Now saves state to the specific run's directory.
    workspace_dir = callback_context.state.get("workspace_dir", "")
    task_name = callback_context.state.get("task_name", "")
    run_id = callback_context.state.get("run_id", "unknown_run")
    
    # This path is now relative to the root workspace, inside the specific run directory.
    # The actual final_state.json will be written by the orchestrator.
    # This function can be used for intermediate state-dumping if needed.
    # For now, its main purpose is fulfilled by the orchestrator.
    print(f"State for run {run_id} is complete.")
    return None


def skip_if_refinement_mode(
    callback_context: callback_context_module.CallbackContext
) -> Optional[types.Content]:
    """Skips initialization and ensemble in refinement-only mode."""
    if callback_context.state.get("is_refinement_run", False):
        print(f"[Refinement Mode] Skipping {callback_context.agent_name}")
        return types.Content(parts=[types.Part(text="skipped")], role="model")
    return None


# PHASE 3: CONDITIONAL PIPELINE ARCHITECTURE FOR LINEAR REFINEMENT

# Wrapper agents with conditional skip logic
initialization_agent_wrapper = agents.SequentialAgent(
    name="initialization_agent_wrapper",
    sub_agents=[initialization_agent_module.initialization_agent],
    description="Initialization agent - skipped in refinement mode",
    before_agent_callback=skip_if_refinement_mode,
)

ensemble_agent_wrapper = agents.SequentialAgent(
    name="ensemble_agent_wrapper",
    sub_agents=[ensemble_agent_module.ensemble_agent],
    description="Ensemble agent - skipped in refinement mode",
    before_agent_callback=skip_if_refinement_mode,
)

# Main pipeline with conditional execution
# - Run 0 (Discovery Mode): Executes all agents (initialization, refinement, ensemble, submission)
# - Run 1+ (Refinement Mode): Skips initialization and ensemble, only runs refinement and submission
mle_pipeline_agent = agents.SequentialAgent(
    name="mle_pipeline_agent",
    sub_agents=[
        initialization_agent_wrapper,  # Skipped in refinement mode
        refinement_agent_module.refinement_agent,  # Always runs
        ensemble_agent_wrapper,  # Skipped in refinement mode
        submission_agent_module.submission_agent,  # Always runs
    ],
    description="Adaptive pipeline: full discovery (Run 0) or focused refinement (Run 1+)",
    after_agent_callback=save_state,
)

# For ADK tools compatibility, the root agent must be named `root_agent`
# This agent remains for interactive, single-shot use via `adk run` or `adk web`.
root_agent = agents.Agent(
    model=config.CONFIG.agent_model,
    name="mle_frontdoor_agent",
    instruction=prompt.FRONTDOOR_INSTRUCTION,
    global_instruction=prompt.SYSTEM_INSTRUCTION,
    sub_agents=[mle_pipeline_agent],
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)
