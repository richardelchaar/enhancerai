"""Code related utility functions."""

from typing import Any, Optional
import subprocess
import os
import time
import autopep8

from google.adk.agents import callback_context as callback_context_module


class Result:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def truncate_output(text: str, max_length: int = 5000) -> str:
    """Truncate output to prevent token explosion from verbose model training."""
    if len(text) <= max_length:
        return text
    
    # Keep the beginning and end, truncate the middle
    keep_start = max_length // 3
    keep_end = max_length // 3
    truncated = text[:keep_start] + f"\n\n... [TRUNCATED {len(text) - max_length} characters to prevent token explosion] ...\n\n" + text[-keep_end:]
    return truncated


def run_python_code(
    code_text: str,
    run_cwd: str,
    py_filepath: str,
    exec_timeout: int,
) -> dict[str, Any]:
    start_time = time.time()
    output_filepath = os.path.join(run_cwd, py_filepath)
    
    # Add code to suppress verbose model output
    suppression_code = """
# Suppress verbose model output to prevent token explosion
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
# Suppress LightGBM verbosity
os.environ['LIGHTGBM_VERBOSITY'] = '-1'
# Suppress XGBoost verbosity  
os.environ['XGBOOST_VERBOSITY'] = '0'
# Suppress sklearn warnings
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

"""
    
    # FIX: Apply autopep8 to fix indentation errors before execution
    # This is critical for ablation scripts where LLMs often generate inconsistent indentation
    try:
        code_text = autopep8.fix_code(
            code_text,
            options={
                'aggressive': 2,  # Level 2 aggressive fixes for indentation
                'max_line_length': 200,  # Allow longer lines (ML code often has long params)
            }
        )
    except Exception as e:
        # If autopep8 fails, continue with original code and log warning
        print(f"Warning: autopep8 formatting failed: {e}. Using original code.")

    # NEW: Validate syntax before execution to catch indentation errors early
    # This gives better error messages to the debug loop
    try:
        compile(code_text, '<string>', 'exec')
    except SyntaxError as e:
        # Syntax error detected - return it immediately so debug loop can fix it
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": f"SyntaxError before execution: {e}\nLine {e.lineno}: {e.text}\n{e.msg}",
            "execution_time": 0.0,
            "score": float("inf") if run_cwd else 0  # Need to infer from context
        }
    except IndentationError as e:
        # Indentation error detected - return it immediately
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": f"IndentationError before execution: {e}\nLine {e.lineno}: {e.text}\n{e.msg}\nPlease check that all lines use consistent 4-space indentation.",
            "execution_time": 0.0,
            "score": float("inf") if run_cwd else 0
        }

    enhanced_code = suppression_code + code_text
    
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(enhanced_code)
    try:
        result = subprocess.run(
            ["python", py_filepath],
            cwd=run_cwd,
            capture_output=True,
            text=True,
            timeout=exec_timeout,
            env={**os.environ, 'PYTHONWARNINGS': 'ignore', 'LIGHTGBM_VERBOSITY': '-1', 'XGBOOST_VERBOSITY': '0'}
        )
    except Exception as e:
        result = Result(returncode=1, stdout="", stderr=str(e))
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Truncate outputs to prevent token explosion
    truncated_stdout = truncate_output(result.stdout)
    truncated_stderr = truncate_output(result.stderr)
    
    result_dict = {
        "returncode": result.returncode,
        "stdout": truncated_stdout,
        "stderr": truncated_stderr,
        "execution_time": execution_time,
    }
    return result_dict


def extract_performance_from_text(text: str) -> float | None:
    """Extracts the final validation performance score from the text."""
    lines = text.splitlines()
    performance_value = None
    for line in lines:
        if "Final Validation Performance" in line:
            try:
                parts = line.split(":")
                # score_str = line.split("Final Validation Performance:")[-1].strip()
                score_str = parts[-1].strip()
                performance_value = float(score_str)
            except ValueError:
                pass
    return performance_value


def get_name_with_prefix_and_suffix(
    base_name: str,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Gets the name with the specified prefix and suffix."""
    new_name = base_name
    if prefix:
        new_name = prefix + "_" + new_name
    if suffix:
        new_name = new_name + "_" + suffix
    return new_name


def get_updated_suffix(
    callback_context: callback_context_module.CallbackContext,
) -> str:
    """Gets the suffix string."""
    agent_name = callback_context.agent_name
    if agent_name.startswith("model_eval"):
        model_id = agent_name.split("_")[-1]
        task_id = agent_name.split("_")[-2]
        suffix = f"{task_id}_{model_id}"
    elif agent_name.startswith("merger"):
        reference_idx = agent_name.split("_")[-1]
        task_id = agent_name.split("_")[-2]
        suffix = f"{task_id}_{reference_idx}"
    elif agent_name.startswith("check_data_use"):
        task_id = agent_name.split("_")[-1]
        suffix = f"{task_id}"
    elif agent_name.startswith("ablation"):
        task_id = agent_name.split("_")[-1]
        step = callback_context.state.get(f"refine_step_{task_id}", 0)
        suffix = f"{step}_{task_id}"
    elif agent_name.startswith("plan_implement"):
        task_id = callback_context.agent_name.split("_")[-1]
        step = callback_context.state.get(f"refine_step_{task_id}", 0)
        inner_iter = callback_context.state.get(f"inner_iter_{task_id}", 0)
        suffix = f"{inner_iter}_{step}_{task_id}"
    elif agent_name.startswith("ensemble_plan_implement"):
        ensemble_iter = callback_context.state.get("ensemble_iter", 0)
        suffix = f"{ensemble_iter}"
    elif agent_name.startswith("submission"):
        suffix = ""
    else:
        raise ValueError(f"Unexpected agent name: {agent_name}.")
    return suffix


def get_code_state_key(
    agent_name: str,
    suffix: str,
) -> str:
    """Gets the state key for the code."""
    if agent_name.startswith("model_eval"):
        key = f"init_code_{suffix}"
    elif agent_name.startswith("merger"):
        key = f"merger_code_{suffix}"
    elif agent_name.startswith("check_data_use"):
        key = f"train_code_0_{suffix}"
    elif agent_name.startswith("ablation"):
        key = f"ablation_code_{suffix}"
    elif agent_name.startswith("plan_implement"):
        key = f"train_code_improve_{suffix}"
    elif agent_name.startswith("ensemble_plan_implement"):
        key = f"ensemble_code_{suffix}"
    elif agent_name.startswith("submission"):
        key = "submission_code"
    else:
        raise ValueError(f"Unexpected agent name: {agent_name}.")
    return key


def get_code_execution_result_state_key(
    agent_name: str,
    suffix: str,
) -> str:
    """Gets the state key for the code execution result."""
    if agent_name.startswith("model_eval"):
        key = f"init_code_exec_result_{suffix}"
    elif agent_name.startswith("merger"):
        key = f"merger_code_exec_result_{suffix}"
    elif agent_name.startswith("check_data_use"):
        key = f"train_code_exec_result_0_{suffix}"
    elif agent_name.startswith("ablation"):
        key = f"ablation_code_exec_result_{suffix}"
    elif agent_name.startswith("plan_implement"):
        key = f"train_code_improve_exec_result_{suffix}"
    elif agent_name.startswith("ensemble_plan_implement"):
        key = f"ensemble_code_exec_result_{suffix}"
    elif agent_name.startswith("submission"):
        key = "submission_code_exec_result"
    else:
        raise ValueError(f"Unexpected agent name: {agent_name}.")
    return key


def get_run_code_condition(
    agent_name: str,
    raw_code: str,
) -> tuple[bool, Optional[dict[str, str]]]:
    """Currently the evaluator allows all code to execute."""
    return True, None


def requires_final_validation_output(agent_name: str) -> bool:
    """Returns True when the agent must print the final validation score."""
    if agent_name.startswith("ablation"):
        return False
    if agent_name.startswith("submission"):
        return False
    return True


def evaluate_code(
    callback_context: callback_context_module.CallbackContext,
) -> None:
    """Evaluates the given code."""
    lower = callback_context.state.get("lower", True)
    exec_timeout = callback_context.state.get("exec_timeout", 1800)
    agent_name = callback_context.agent_name
    suffix = get_updated_suffix(callback_context=callback_context)
    code_state_key = get_code_state_key(
        agent_name=agent_name,
        suffix=suffix,
    )
    raw_code = callback_context.state.get(code_state_key, "")
    
    # FIX: This block logic needs to be more specific.
    # It was failing to correctly determine the `task_id` for the run directory,
    # and was putting submission artifacts in the wrong place.
    if agent_name.startswith("model_eval"):
        model_id = agent_name.split("_")[-1]
        task_id = agent_name.split("_")[-2]
        py_filepath = f"init_code_{model_id}.py"
    elif agent_name.startswith("merger"):
        reference_idx = agent_name.split("_")[-1]
        task_id = agent_name.split("_")[-2]
        py_filepath = f"train0_{reference_idx}.py"
    elif agent_name.startswith("check_data_use"):
        task_id = agent_name.split("_")[-1]
        py_filepath = "train0.py"
    elif agent_name.startswith("ablation"):
        task_id = agent_name.split("_")[-1]
        step = callback_context.state.get(f"refine_step_{task_id}", 0)
        py_filepath = f"ablation_{step}.py"
    elif agent_name.startswith("plan_implement"):
        task_id = agent_name.split("_")[-1]
        step = callback_context.state.get(f"refine_step_{task_id}", 0)
        inner_iter = callback_context.state.get(f"inner_iter_{task_id}", 0)
        py_filepath = f"train{step}_improve{inner_iter}.py"
    elif agent_name.startswith("ensemble_plan_implement"):
        task_id = "ensemble"
        py_filepath = f"ensemble{suffix}.py"
    elif agent_name.startswith("submission"):
        # Submission operates in the 'ensemble' directory.
        task_id = "ensemble"
        py_filepath = "final_solution.py"
    else:
        raise ValueError(f"Unexpected agent name: {agent_name}.")
    
    should_run, blocking_error = get_run_code_condition(
        agent_name=agent_name,
        raw_code=raw_code,
    )
    validation_errors: list[dict[str, str]] = []
    if blocking_error:
        validation_errors.append(blocking_error)

    if should_run:
        workspace_dir = callback_context.state.get("workspace_dir", "")
        run_cwd = os.path.join(workspace_dir, task_id)
        result_dict = run_python_code(
            code_text=raw_code,
            run_cwd=run_cwd,
            py_filepath=py_filepath,
            exec_timeout=exec_timeout,
        )
        if agent_name.startswith("ablation"):
            if result_dict["returncode"] == 0:
                ablation_result = result_dict.get("stdout", "None")
            else:
                ablation_result = "None"
            result_dict["ablation_result"] = ablation_result
        else:
            if result_dict.get("returncode", 1) == 0:
                try:
                    score = extract_performance_from_text(result_dict.get("stdout", ""))
                    score = float(score)
                except:
                    score = 1e9 if lower else 0
            else:
                score = 1e9 if lower else 0
            result_dict["score"] = score
    else:
        result_dict = {
            "error": blocking_error["message"] if blocking_error else "Code validation failed.",
            "returncode": 1,
            "stdout": "",
            "stderr": blocking_error["message"] if blocking_error else "",
            "execution_time": 0.0,
            "score": float("inf") if lower else float("-inf")
        }
    if should_run and requires_final_validation_output(agent_name):
        stdout_text = result_dict.get("stdout", "")
        if "Final Validation Performance" not in stdout_text:
            validation_errors.append({
                "code": "missing_final_validation_output",
                "message": (
                    'Execution must print "Final Validation Performance: <score>" '
                    "for the harness to read the metric."
                ),
            })

    if validation_errors:
        combined_message = " | ".join(err["message"] for err in validation_errors)
        stderr_text = result_dict.get("stderr", "")
        if stderr_text:
            if combined_message not in stderr_text:
                result_dict["stderr"] = f"{stderr_text.rstrip()} | {combined_message}"
        else:
            result_dict["stderr"] = combined_message

        error_text = result_dict.get("error", "")
        if error_text:
            if combined_message not in error_text:
                result_dict["error"] = f"{error_text.rstrip()} | {combined_message}"
        else:
            result_dict["error"] = combined_message

        result_dict["validation_errors"] = validation_errors

        if result_dict.get("returncode", 0) == 0:
            result_dict["returncode"] = 1

        if agent_name.startswith("ablation"):
            result_dict["ablation_result"] = "None"
        else:
            result_dict["score"] = 1e9 if lower else 0
    code_execution_result_state_key = get_code_execution_result_state_key(
        agent_name=agent_name,
        suffix=suffix,
    )
    callback_context.state[code_execution_result_state_key] = result_dict
    return None
