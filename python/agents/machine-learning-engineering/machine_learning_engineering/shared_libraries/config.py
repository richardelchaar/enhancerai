"""Configuration for Machine Learning Engineering Agent."""

import dataclasses
import os


@dataclasses.dataclass
class DefaultConfig:
    """Default configuration."""
    data_dir: str = "./machine_learning_engineering/tasks/"
    task_name: str = "california-housing-prices"
    task_type: str = "Tabular Regression"
    lower: bool = True
    workspace_dir: str = "./machine_learning_engineering/workspace/"
    agent_model: str = os.environ.get("ROOT_AGENT_MODEL", "gemini-2.0-flash-001")
    task_description: str = ""
    task_summary: str = ""
    start_time: float = 0.0
    seed: int = 42
    exec_timeout: int = 600
    
    # --- MODIFIED FOR META-LEARNING ---
    # These values are now intended as the baseline for the first run.
    # The Enhancer agent will dynamically override them in subsequent runs.
    num_solutions: int = 2
    num_model_candidates: int = 2
    max_retry: int = 10
    max_debug_round: int = 5
    max_rollback_round: int = 2
    inner_loop_round: int = 1  # Changed from higher value to 1 for faster baseline
    outer_loop_round: int = 1  # Changed from higher value to 1 for faster baseline
    ensemble_loop_round: int = 1 # Changed from higher value to 1 for faster baseline
    num_top_plans: int = 2
    use_data_leakage_checker: bool = False
    use_data_usage_checker: bool = False
    
    # --- NEW FOR META-LEARNING FRAMEWORK ---
    allow_config_override: bool = True # Master switch to allow the orchestrator to apply Enhancer's strategy
    computational_budget: int = 3600 # Target runtime in seconds for an enhanced run


CONFIG = DefaultConfig()
