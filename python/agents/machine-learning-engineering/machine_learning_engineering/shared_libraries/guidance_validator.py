"""Validation helpers that enforce enhancement directives on generated code."""

from __future__ import annotations

from typing import List, Optional

from machine_learning_engineering.shared_libraries import common_util

_HPO_KEYWORDS = ["randomizedsearchcv", "gridsearchcv", "optuna", "hyperopt"]


def _section_for_agent(agent_name: str) -> Optional[str]:
    """Maps agent names to the guidance section that applies to them."""

    if agent_name.startswith(("model_eval", "merger", "check_data_use")):
        return "initialization"
    if agent_name.startswith("plan_implement"):
        return "refinement"
    if agent_name.startswith("ensemble_plan_implement"):
        return "ensemble"
    if agent_name.startswith("submission"):
        return "submission"
    return None


def validate_code(
    context,
    code: str,
) -> List[str]:
    """Returns human-readable validation errors for the given code."""

    agent_name = context.agent_name
    # Skip ablation artefacts – they are diagnostic only.
    if agent_name.startswith("ablation"):
        return []

    section = _section_for_agent(agent_name)
    if not section:
        return []

    guidance_text = common_util.get_run_guidance(context, section)
    if not guidance_text:
        return []

    guidance_lower = guidance_text.lower()
    code_lower = code.lower()
    errors: List[str] = []

    # Hyperparameter optimisation requirements.
    if any(keyword in guidance_lower for keyword in _HPO_KEYWORDS):
        if not any(keyword in code_lower for keyword in _HPO_KEYWORDS):
            errors.append(
                "Guidance requires hyperparameter search (RandomizedSearchCV / GridSearchCV / Optuna / Hyperopt)"
                " but the generated code does not include any of those tools."
            )

    # CatBoost integration requirement.
    if "catboost" in guidance_lower and "catboost" not in code_lower:
        errors.append(
            "Guidance calls for CatBoost integration, but the generated code does not reference CatBoost."
        )

    # inplace=True guidance.
    if "inplace=true" in guidance_lower and "inplace=true" in code_lower:
        errors.append(
            "Guidance forbids using pandas fillna with inplace=True, yet the generated code still does so."
        )

    # Ensemble-specific regression enforcement.
    if section == "ensemble" and ("regression" in guidance_lower or "rmse" in guidance_lower):
        if any(token in code_lower for token in ["classifier", "roc_auc", "make_classification"]):
            errors.append(
                "Ensemble guidance requires a regression workflow, but classification constructs remain in the code."
            )
        if not any(token in code_lower for token in ["regressor", "mean_squared_error", "rmse"]):
            errors.append(
                "Ensemble guidance expects regression metrics (e.g. RMSE), yet no regression metrics or regressors were detected."
            )
        if any(token in code_lower for token in ["create_synthetic_data", "make_classification"]):
            errors.append(
                "Ensemble guidance forbids synthetic data fallbacks, but the generated code creates synthetic datasets."
            )

    return errors


__all__ = ["validate_code"]
