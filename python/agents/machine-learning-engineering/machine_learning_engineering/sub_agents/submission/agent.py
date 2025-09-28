"""Submission agent for Machine Learning Engineering."""

from typing import Optional

from google.adk.agents import callback_context as callback_context_module
from google.adk.models import llm_response as llm_response_module
from google.adk.models import llm_request as llm_request_module

from machine_learning_engineering.sub_agents.submission import prompt
from machine_learning_engineering.shared_libraries import debug_util


def check_submission_finish(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> Optional[llm_response_module.LlmResponse]:
    """Checks if adding codes for submission is finished."""
    result_dict = callback_context.state.get(
        "submission_code_exec_result", {}
    )
    callback_context.state[
        "submission_skip_data_leakage_check"
    ] = True
    if result_dict:
        return llm_response_module.LlmResponse()
    callback_context.state[
        "submission_skip_data_leakage_check"
    ] = False
    return None


def get_submission_and_debug_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the submission agent instruction."""
    num_solutions = context.state.get("num_solutions", 2)
    ensemble_loop_round = context.state.get("ensemble_loop_round", 2)
    task_description = context.state.get("task_description", "")
    lower = context.state.get("lower", True)
    final_solution = ""
    best_score = None

    def consider_candidate(code: str, exec_result) -> None:
        nonlocal final_solution, best_score
        if not isinstance(exec_result, dict):
            return
        curr_score = exec_result.get("score")
        if curr_score is None:
            return
        if (
            best_score is None
            or (lower and curr_score < best_score)
            or (not lower and curr_score > best_score)
        ):
            final_solution = code
            best_score = curr_score

    for task_id in range(1, num_solutions + 1):
        # use the last completed refinement step for this task if available
        last_step = context.state.get(f"refine_step_{task_id}", 0)
        for step in range(last_step, -1, -1):
            curr_code = context.state.get(
                f"train_code_{step}_{task_id}", ""
            )
            curr_exec_result = context.state.get(
                f"train_code_exec_result_{step}_{task_id}", {}
            )
            consider_candidate(curr_code, curr_exec_result)
            # once we find a valid score, stop scanning earlier steps
            if isinstance(curr_exec_result, dict) and curr_exec_result.get("score") is not None:
                break
    for ensemble_iter in range(ensemble_loop_round + 1):
        curr_code = context.state.get(
            f"ensemble_code_{ensemble_iter}", {}
        )
        curr_exec_result = context.state.get(
            f"ensemble_code_exec_result_{ensemble_iter}", {}
        )
        consider_candidate(curr_code, curr_exec_result)
    
    # --- Add strategic guidance for submission ---
    enhancer_output = context.state.get("enhancer_output", {})
    strategic_goals = enhancer_output.get("strategic_goals", [])
    
    submission_goals = [
        g for g in strategic_goals if g.get("target_agent_phase") == "submission"
    ]
    
    strategic_guidance = ""
    if submission_goals:
        primary_goal = sorted(submission_goals, key=lambda x: x.get("priority", 99))[0]
        strategic_guidance = f"\n\n# Strategic Guidance from Research Lead:\n- Focus: {primary_goal.get('focus', 'No specific focus')}\n- Rationale: {primary_goal.get('rationale', 'No rationale provided.')}\n\nApply this strategic guidance when preparing the final submission."
    
    return prompt.ADD_TEST_FINAL_INSTR.format(
        task_description=task_description,
        code=final_solution,
    ) + strategic_guidance


submission_agent = debug_util.get_run_and_debug_agent(
    prefix="submission",
    suffix="",
    agent_description="Add codes for creating a submission file.",
    instruction_func=get_submission_and_debug_agent_instruction,
    before_model_callback=check_submission_finish,
)
