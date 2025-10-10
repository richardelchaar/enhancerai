"""
Main entry point for the MLE-STAR Meta-Learning Framework.

This orchestrator manages the N-run process, invoking the MLE-STAR pipeline,
collecting results, and using the Enhancer agent to strategize for subsequent runs.
"""
# Import compatibility fixes first
from machine_learning_engineering.shared_libraries import aiohttp_compat

import argparse
import dataclasses
import json
import os
import shutil
import time
from typing import Any, Dict, List

from google.genai import types
from google.adk.runners import Runner
from google.adk.runners import InMemoryRunner

from machine_learning_engineering.agent import mle_pipeline_agent
from machine_learning_engineering.shared_libraries import config
from machine_learning_engineering.sub_agents.enhancer.agent import enhancer_agent


class MetaOrchestrator:
    """Manages the entire multi-run meta-learning workflow."""

    def __init__(self, task_name: str, num_runs: int, initial_config_path: str = None):
        self.task_name = task_name
        self.num_runs = num_runs
        self.workspace_root = os.path.join(
            config.CONFIG.workspace_dir, self.task_name
        )
        self.run_history_path = os.path.join(self.workspace_root, "run_history.json")
        self.run_history = []
        self.initial_config = self._load_initial_config(initial_config_path)

    def _load_initial_config(self, path: str) -> Dict[str, Any]:
        """Loads an initial config override from a JSON file."""
        if path and os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def _setup_workspace(self):
        """Creates the master workspace and loads the run history."""
        os.makedirs(self.workspace_root, exist_ok=True)
        if os.path.exists(self.run_history_path):
            with open(self.run_history_path, "r") as f:
                self.run_history = json.load(f)
        print(f"Workspace initialized at: {self.workspace_root}")
        print(f"Loaded {len(self.run_history)} previous run(s) from history.")

    def _get_config_for_run(self, run_id: int) -> Dict[str, Any]:
        """Determines the configuration for the current run."""
        # Start with a fresh copy of the defaults
        current_config = config.DefaultConfig()
        enhancer_output: Dict[str, Any] = {}

        if run_id == 0:
            # For the first run, apply any user-provided initial overrides
            for key, value in self.initial_config.items():
                if hasattr(current_config, key):
                    setattr(current_config, key, value)
            print("Run 0: Using baseline configuration with user overrides.")
        else:
            # For subsequent runs, use the strategy from the previous run's Enhancer
            prev_run_summary = self.run_history[-1]
            enhancer_output = prev_run_summary.get("enhancer_output", {})
            overrides = enhancer_output.get("config_overrides", {})
            if config.CONFIG.allow_config_override:
                for key, value in overrides.items():
                    if hasattr(current_config, key):
                        setattr(current_config, key, value)
                        print(f"  - Overriding config '{key}' with value '{value}'")
                print(f"Run {run_id}: Applying enhanced configuration from previous run.")
            else:
                print(f"Run {run_id}: allow_config_override is False. Using default config.")
        
        # Convert config to a plain dictionary (defensive copy) and inject enhancer output
        run_config_dict = dataclasses.asdict(current_config)
        run_config_dict["enhancer_output"] = enhancer_output
        run_config_dict["task_name"] = self.task_name
        return run_config_dict

    async def _execute_pipeline_run(self, run_id: int, run_config: Dict[str, Any]):
        """Executes a single, isolated run of the MLE-STAR pipeline."""
        print(f"\n--- Starting MLE-STAR Pipeline: Run {run_id} ---")
        run_dir = os.path.join(self.workspace_root, f"run_{run_id}")
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        os.makedirs(run_dir)
        
        # Override the default workspace to isolate this run
        # Ensures all artifacts for this run go under: workspace_root/run_{id}/
        run_config["workspace_dir"] = run_dir

        # Ensure sub-agents that consult the global CONFIG see the overrides for this run.
        overrides_snapshot = self._apply_global_config_overrides(run_config)

        # Build initial session state
        initial_state: Dict[str, Any] = {**run_config}
        initial_state["run_dir"] = run_dir
        initial_state["run_id"] = run_id

        # PHASE 2: LINEAR REFINEMENT MODE - Seed state with champion from previous run
        is_refinement_run = run_id > 0
        
        # Reset num_solutions for discovery runs (in case it was overridden before)
        if not is_refinement_run:
            config.CONFIG.num_solutions = run_config.get("num_solutions", 2)
        
        if is_refinement_run:
            print(f"[Run {run_id}] Initializing Linear Refinement Mode")
            
            # CRITICAL: Agents were constructed at import time with run_config's num_solutions
            # We need to create directories for ALL tasks the agents expect, even if we only use task 1
            num_solutions_from_agents = run_config.get("num_solutions", config.CONFIG.num_solutions)
            
            # Override CONFIG.num_solutions to 1 for linear refinement logic
            config.CONFIG.num_solutions = 1
            print(f"[Run {run_id}] Linear refinement mode: will only refine 1 solution (champion)")
            print(f"[Run {run_id}] But creating {num_solutions_from_agents} directories for agent compatibility")
            
            # Load run history to find the champion
            if not os.path.exists(self.run_history_path):
                raise FileNotFoundError(f"Run history not found at {self.run_history_path}. Cannot run refinement mode without previous run.")
            
            with open(self.run_history_path, "r") as f:
                history = json.load(f)
            
            # Get the best solution from the most recent run
            last_run = history[-1]
            champion_relative_path = last_run["best_solution_path"]
            champion_full_path = os.path.join(self.workspace_root, champion_relative_path)
            
            print(f"[Run {run_id}] Loading champion from: {champion_full_path}")
            
            # Read the champion code
            if not os.path.exists(champion_full_path):
                raise FileNotFoundError(f"Champion code not found at {champion_full_path}")
            
            with open(champion_full_path, "r") as f:
                champion_code = f.read()
            
            # Read task description (normally loaded by initialization agent's prepare_task)
            task_desc_path = os.path.join(config.CONFIG.data_dir, self.task_name, "task_description.txt")
            if os.path.exists(task_desc_path):
                with open(task_desc_path, "r") as f:
                    initial_state["task_description"] = f.read()
            
            # CRITICAL: Create task subdirectories for ALL tasks that agents expect
            # Agents were constructed at import time with num_solutions_from_agents
            # Even though only Task 1 will execute (Task 2+ are skipped via wrapper),
            # we create all directories to avoid FileNotFoundError if agents check for them
            for task_id in range(1, num_solutions_from_agents + 1):
                task_dir = os.path.join(run_dir, str(task_id))
                os.makedirs(task_dir, exist_ok=True)
                
                # Copy input data to each task directory (needed for code execution)
                input_dir = os.path.join(task_dir, "input")
                os.makedirs(input_dir, exist_ok=True)
                
                # Copy training data files
                source_data_dir = os.path.join(config.CONFIG.data_dir, self.task_name)
                for data_file in ["train.csv", "test.csv", "task_description.txt"]:
                    source_file = os.path.join(source_data_dir, data_file)
                    dest_file = os.path.join(input_dir, data_file)
                    if os.path.exists(source_file):
                        shutil.copy2(source_file, dest_file)
            
            # Create ensemble directory (needed for submission agent even though we skip ensemble agent)
            ensemble_dir = os.path.join(run_dir, "ensemble")
            os.makedirs(ensemble_dir, exist_ok=True)
            ensemble_input_dir = os.path.join(ensemble_dir, "input")
            os.makedirs(ensemble_input_dir, exist_ok=True)
            
            # Copy data to ensemble directory as well
            source_data_dir = os.path.join(config.CONFIG.data_dir, self.task_name)
            for data_file in ["train.csv", "test.csv", "task_description.txt"]:
                source_file = os.path.join(source_data_dir, data_file)
                dest_file = os.path.join(ensemble_input_dir, data_file)
                if os.path.exists(source_file):
                    shutil.copy2(source_file, dest_file)
            
            # CRITICAL: Seed ALL task IDs with champion code
            # The refinement agent created agents at import time based on num_solutions_from_agents
            # Even though only Task 1 executes (Task 2+ skipped), we seed all to avoid state errors
            for task_id in range(1, num_solutions_from_agents + 1):
                initial_state[f"train_code_0_{task_id}"] = champion_code
                initial_state[f"train_code_exec_result_0_{task_id}"] = {
                    "score": last_run["best_score"],
                    "returncode": 0
                }
            
            # Set refinement-only mode flag
            initial_state["is_refinement_run"] = True
            initial_state["num_solutions"] = 1  # LINEAR REFINEMENT: Only refine 1 solution (the champion)
            print(f"[Run {run_id}] Created workspace structure and seeded {num_solutions_from_agents} task(s) with champion code (score: {last_run['best_score']})")
            print(f"[Run {run_id}] LINEAR REFINEMENT MODE: Only Task 1 will execute (Task 2+ skipped)")
            print(f"[Run {run_id}] Note: All {num_solutions_from_agents} task directories created for agent compatibility")

        runner = InMemoryRunner(agent=mle_pipeline_agent, app_name="mle-meta-pipeline")
        try:
            session = await runner.session_service.create_session(
                app_name=runner.app_name,
                user_id=f"meta-user-{run_id}",
                state=initial_state,
            )

            # Kick off the run with a minimal user message
            content = types.Content(parts=[types.Part(text="run")], role="user")
            async for _ in runner.run_async(
                user_id=session.user_id,
                session_id=session.id,
                new_message=content,
            ):
                pass

            # Fetch final merged state from the session service
            final_session = await runner.session_service.get_session(
                app_name=runner.app_name,
                user_id=session.user_id,
                session_id=session.id,
            )
            final_state: Dict[str, Any] = final_session.state
        finally:
            # Restore global config and close runner
            try:
                await runner.close()
            finally:
                for key, value in overrides_snapshot.items():
                    setattr(config.CONFIG, key, value)

        # Save the complete state for this run for analysis by the Enhancer
        with open(os.path.join(run_dir, "final_state.json"), "w") as f:
            json.dump(final_state, f, indent=2)
            
        print(f"--- Finished MLE-STAR Pipeline: Run {run_id} ---")
        return final_state

    def _update_run_history(self, run_id: int, final_state: Dict[str, Any]):
        """Parses the final state and updates the persistent run history."""
        # Find the best score AND path from this run
        best_score = None
        best_solution_path = None
        lower_is_better = final_state.get("lower", True)
        
        # Check all potential final solutions from all parallel initializations
        for i in range(final_state.get("num_solutions", 1)):
            task_id = i + 1
            # Check final refined score
            last_step = final_state.get(f'refine_step_{task_id}', 0)
            refined_score = final_state.get(f"train_code_exec_result_{last_step}_{task_id}", {}).get("score")
            if refined_score is not None:
                if best_score is None or (lower_is_better and refined_score < best_score) or (not lower_is_better and refined_score > best_score):
                    best_score = refined_score
                    # Track the actual file path for this refined solution
                    best_solution_path = f"run_{run_id}/{task_id}/train{last_step}_{task_id}.py"

        # Check final ensembled score
        ensemble_iter = final_state.get("ensemble_iter", 0)
        ensembled_score = final_state.get(f"ensemble_code_exec_result_{ensemble_iter}", {}).get("score")
        if ensembled_score is not None:
            if best_score is None or (lower_is_better and ensembled_score < best_score) or (not lower_is_better and ensembled_score > best_score):
                best_score = ensembled_score
                best_solution_path = f"run_{run_id}/ensemble/final_solution.py"
        
        # Fallback if no valid solution found
        if best_solution_path is None:
            best_solution_path = f"run_{run_id}/ensemble/final_solution.py"
            print(f"WARNING: No valid solution found for run {run_id}, defaulting to ensemble path")
                    
        run_summary = {
            "run_id": run_id,
            "status": "COMPLETED_SUCCESSFULLY" if best_score is not None else "FAILED",
            "start_time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(final_state.get("start_time", time.time()))),
            "duration_seconds": round(time.time() - final_state.get("start_time", time.time())),
            "best_score": best_score,
            "best_solution_path": best_solution_path,
            "config_used": {k: v for k, v in final_state.items() if isinstance(v, (int, str, bool, float))},
            "enhancer_rationale": "Baseline run." if run_id == 0 else self.run_history[-1].get("enhancer_output", {}).get("strategic_summary", "N/A")
        }
        
        self.run_history.append(run_summary)
        with open(self.run_history_path, "w") as f:
            json.dump(self.run_history, f, indent=2)
        print(f"Run {run_id} summary saved. Best score: {best_score}")

    async def _invoke_enhancer(self, last_run_id: int) -> Dict[str, Any]:
        """Runs the Enhancer agent to get the strategy for the next run."""
        print(f"\n--- Invoking Enhancer Agent to strategize for Run {last_run_id + 1} ---")
        
        last_run_state_path = os.path.join(self.workspace_root, f"run_{last_run_id}", "final_state.json")
        with open(last_run_state_path, 'r') as f:
            last_run_state = json.load(f)
            
        enhancer_initial_state: Dict[str, Any] = {}
        enhancer_initial_state["last_run_final_state"] = last_run_state
        enhancer_initial_state["run_history_summary"] = self.run_history
        # Provide JSON-string copies as a fallback in case nested structures are
        # cleaned from the session state by downstream agents.
        enhancer_initial_state["last_run_final_state_json"] = json.dumps(last_run_state)
        enhancer_initial_state["run_history_summary_json"] = json.dumps(self.run_history)
        scores = [r['best_score'] for r in self.run_history if r['best_score'] is not None]
        enhancer_initial_state["best_score_so_far"] = min(scores) if scores else None

        runner = InMemoryRunner(agent=enhancer_agent, app_name="mle-meta-enhancer")
        try:
            session = await runner.session_service.create_session(
                app_name=runner.app_name,
                user_id="meta-enhancer",
                state=enhancer_initial_state,
            )
            content = types.Content(parts=[types.Part(text="enhance")], role="user")
            async for _ in runner.run_async(
                user_id=session.user_id,
                session_id=session.id,
                new_message=content,
            ):
                pass

            final_session = await runner.session_service.get_session(
                app_name=runner.app_name,
                user_id=session.user_id,
                session_id=session.id,
            )
            enhancer_output = final_session.state.get("enhancer_output", {})
        finally:
            await runner.close()
        
        # Append the enhancer's output to the history for the *current* run for traceability
        self.run_history[-1]["enhancer_output"] = enhancer_output
        with open(self.run_history_path, "w") as f:
            json.dump(self.run_history, f, indent=2)
        
        print(f"Enhancer Rationale: {enhancer_output.get('strategic_summary', 'No summary provided.')}")
        return enhancer_output

    def _apply_global_config_overrides(self, run_config: Dict[str, Any]) -> Dict[str, Any]:
        """Updates the shared CONFIG module state with run-specific overrides.

        Returns a snapshot of the original values so they can be restored after the run.
        """
        overrides_snapshot: Dict[str, Any] = {}
        for key, value in run_config.items():
            if hasattr(config.CONFIG, key):
                overrides_snapshot[key] = getattr(config.CONFIG, key)
        for key in overrides_snapshot:
            setattr(config.CONFIG, key, run_config[key])
        return overrides_snapshot

    async def run(self):
        """Executes the full meta-learning loop for N runs."""
        self._setup_workspace()
        
        for i in range(len(self.run_history), self.num_runs):
            current_config = self._get_config_for_run(i)
            
            # This is where the core pipeline is executed
            final_state = await self._execute_pipeline_run(run_id=i, run_config=current_config)
            
            # This function updates the history log
            self._update_run_history(run_id=i, final_state=final_state)
            
            # This invokes the enhancer to prepare for the *next* loop
            if i < self.num_runs - 1:
                await self._invoke_enhancer(last_run_id=i)

        print("\n--- Meta-Learning Framework Execution Complete ---")
        best_run = sorted([r for r in self.run_history if r['best_score'] is not None], key=lambda x: x['best_score'])[0]
        print(f"Best score of {best_run['best_score']} achieved in run {best_run['run_id']}.")
        print(f"Find the best solution at: {os.path.join(self.workspace_root, best_run['best_solution_path'])}")


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser(description="MLE-STAR Meta-Learning Framework Orchestrator")
    parser.add_argument("--task_name", type=str, default="california-housing-prices", help="Name of the task directory.")
    parser.add_argument("--num_runs", type=int, default=3, help="Total number of enhancement loops to run.")
    parser.add_argument("--initial_config_path", type=str, default=None, help="Path to a JSON file for initial config overrides.")
    
    args = parser.parse_args()

    orchestrator = MetaOrchestrator(
        task_name=args.task_name,
        num_runs=args.num_runs,
        initial_config_path=args.initial_config_path
    )
    asyncio.run(orchestrator.run())
