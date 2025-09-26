"""Configuration for Machine Learning Engineering Agent."""

import dataclasses
import os


@dataclasses.dataclass
class DefaultConfig:
    """Default configuration."""

    data_dir: str = "./machine_learning_engineering/tasks/"  # Directory containing the machine learning tasks and their data.
    task_name: str = "california-housing-prices"  # Name of the task to run.
    task_type: str = "Tabular Regression"  # Type of machine learning problem.
    lower: bool = True  # True if a lower value of the metric is better.
    workspace_dir: str = "./machine_learning_engineering/workspace/"  # Base directory for intermediate outputs and artifacts.
    agent_model: str = os.environ.get("ROOT_AGENT_MODEL", "gemini-2.0-flash-001")  # LLM model identifier used by the agents.
    task_description: str = ""  # Detailed description of the task.
    task_summary: str = ""  # Concise summary of the task.
    start_time: float = 0.0  # Timestamp marking the beginning of the task run.
    seed: int = 42  # Random seed for reproducibility.
    exec_timeout: int = 600  # Maximum execution time (seconds) per generated script.
    num_solutions: int = 2  # Number of distinct solution tracks to explore in parallel.
    num_model_candidates: int = 2  # Number of model candidates retrieved per solution track.
    max_retry: int = 10  # Maximum retries for agent loops.
    max_debug_round: int = 5  # Maximum debug iterations within the debug loop.
    max_rollback_round: int = 2  # Maximum rollback iterations for run-and-debug loops.
    inner_loop_round: int = 1  # Inner refinement iterations per outer loop.
    outer_loop_round: int = 1  # Number of outer refinement iterations.
    ensemble_loop_round: int = 1  # Number of ensemble refinement iterations.
    num_top_plans: int = 2  # Number of top plans to retain during plan refinement.
    use_data_leakage_checker: bool = False  # Toggle for the optional data leakage checker.
    use_data_usage_checker: bool = False  # Toggle for the optional data usage checker.
    run_guidance_path: str = ""  # Optional path to a JSON file containing guidance from a previous run.
    run_id: int = 0  # Identifier of the current run in a multi-run workflow.
    num_runs: int = 1  # Total number of runs in the multi-run workflow.


CONFIG = DefaultConfig()


def load_config_from_mapping(overrides: dict | None = None) -> DefaultConfig:
    """Creates a configuration instance from optional overrides."""

    base_config = DefaultConfig()
    if not overrides:
        return base_config
    valid_keys = {field.name for field in dataclasses.fields(DefaultConfig)}
    filtered_overrides = {k: v for k, v in overrides.items() if k in valid_keys}
    return dataclasses.replace(base_config, **filtered_overrides)


def set_config(new_config: DefaultConfig) -> None:
    """Updates the module-level configuration reference."""

    global CONFIG
    CONFIG = new_config
