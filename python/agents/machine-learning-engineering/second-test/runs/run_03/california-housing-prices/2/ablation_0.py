
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def run_experiment(model_type='ensemble', imputation_strategy='median'):
    """
    Runs a single experiment with specified model type and imputation strategy.

    Args:
        model_type (str): 'ensemble', 'lightgbm_only', or 'xgboost_only'.
        imputation_strategy (str): 'median' or 'mean' for 'total_bedrooms'.

    Returns:
        float: The Root Mean Squared Error (RMSE) on the validation set.
    """
    # Load the training dataset
    train_df = pd.read_csv("./input/train.csv")

    # Separate features (X) and the target variable (y)
    X = train_df.drop("median_house_value", axis=1)
    y = train_df["median_house_value"]

    # --- Preprocessing for Missing Values ---
    # Impute 'total_bedrooms' missing values based on the strategy
    if imputation_strategy == 'median':
        val_to_fill = X['total_bedrooms'].median()
    elif imputation_strategy == 'mean':
        val_to_fill = X['total_bedrooms'].mean()
    else:
        raise ValueError("Invalid imputation_strategy. Choose 'median' or 'mean'.")
    X['total_bedrooms'].fillna(val_to_fill, inplace=True)

    # --- Data Splitting ---
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred_lgb = None
    y_pred_xgb = None

    # --- Model Initialization and Training ---
    if model_type in ['ensemble', 'lightgbm_only']:
        model_lgb = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model_lgb.fit(X_train, y_train)
        y_pred_lgb = model_lgb.predict(X_val)

    if model_type in ['ensemble', 'xgboost_only']:
        model_xgb = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model_xgb.fit(X_train, y_train)
        y_pred_xgb = model_xgb.predict(X_val)

    # --- Model Prediction and Ensembling ---
    if model_type == 'ensemble':
        if y_pred_lgb is None or y_pred_xgb is None:
            raise ValueError("Both LightGBM and XGBoost predictions are required for ensembling.")
        y_pred_final = (y_pred_lgb + y_pred_xgb) / 2
    elif model_type == 'lightgbm_only':
        if y_pred_lgb is None:
            raise ValueError("LightGBM model was not trained for 'lightgbm_only' mode.")
        y_pred_final = y_pred_lgb
    elif model_type == 'xgboost_only':
        if y_pred_xgb is None:
            raise ValueError("XGBoost model was not trained for 'xgboost_only' mode.")
        y_pred_final = y_pred_xgb
    else:
        raise ValueError("Invalid model_type provided.")

    # --- Model Evaluation ---
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_final))
    return rmse

# --- Baseline Performance (Original Solution) ---
baseline_rmse = run_experiment(model_type='ensemble', imputation_strategy='median')
print(f"Original Solution RMSE (Ensemble + Median Imputation): {baseline_rmse:.4f}")

# --- Ablation 1: Only LightGBM Model (removing XGBoost and ensembling) ---
lgb_only_rmse = run_experiment(model_type='lightgbm_only', imputation_strategy='median')
print(f"Ablation 1 (LightGBM Only, Median Imputation) RMSE: {lgb_only_rmse:.4f}")

# --- Ablation 2: Only XGBoost Model (removing LightGBM and ensembling) ---
xgb_only_rmse = run_experiment(model_type='xgboost_only', imputation_strategy='median')
print(f"Ablation 2 (XGBoost Only, Median Imputation) RMSE: {xgb_only_rmse:.4f}")

# --- Ablation 3: Change Missing Value Imputation to Mean (keeping ensemble) ---
mean_imputation_ensemble_rmse = run_experiment(model_type='ensemble', imputation_strategy='mean')
print(f"Ablation 3 (Ensemble + Mean Imputation) RMSE: {mean_imputation_ensemble_rmse:.4f}")

# --- Ablation Study Conclusion ---
print("\n--- Contribution Analysis ---")

# Calculate the performance difference when a component is removed/changed from the baseline
# A positive difference indicates that the original component contributed positively (its removal/change degraded performance).
diff_lgb_only = lgb_only_rmse - baseline_rmse # Impact of including XGBoost in ensemble
diff_xgb_only = xgb_only_rmse - baseline_rmse # Impact of including LightGBM in ensemble
diff_mean_imputation = mean_imputation_ensemble_rmse - baseline_rmse # Impact of using Median vs. Mean Imputation

contributions = {
    "The inclusion of the XGBoost model (as part of the ensemble)": diff_lgb_only,
    "The inclusion of the LightGBM model (as part of the ensemble)": diff_xgb_only,
    "The choice of Median Imputation for 'total_bedrooms'": diff_mean_imputation
}

# Filter for positive contributions (where removing/changing the part degraded performance, i.e., original was better)
positive_contributions = {k: v for k, v in contributions.items() if v > 0}

if positive_contributions:
    most_contributing_desc = max(positive_contributions, key=positive_contributions.get)
    contribution_value = positive_contributions[most_contributing_desc]
    print(f"The part that contributes the most to the overall performance (achieving the lowest RMSE in the baseline) is **{most_contributing_desc}**. Removing or altering this part led to the largest degradation in performance, increasing RMSE by {contribution_value:.4f}.")
else:
    # This block handles cases where ablations either didn't degrade performance or even improved it.
    best_ablation_rmse = min(lgb_only_rmse, xgb_only_rmse, mean_imputation_ensemble_rmse)
    if best_ablation_rmse < baseline_rmse:
        if best_ablation_rmse == lgb_only_rmse:
            improvement = baseline_rmse - lgb_only_rmse
            print(f"Surprisingly, using only the LightGBM model performed better than the ensemble by {improvement:.4f} RMSE, suggesting the ensemble or XGBoost's contribution might be detrimental in this specific setup.")
        elif best_ablation_rmse == xgb_only_rmse:
            improvement = baseline_rmse - xgb_only_rmse
            print(f"Surprisingly, using only the XGBoost model performed better than the ensemble by {improvement:.4f} RMSE, suggesting the ensemble or LightGBM's contribution might be detrimental in this specific setup.")
        elif best_ablation_rmse == mean_imputation_ensemble_rmse:
            improvement = baseline_rmse - mean_imputation_ensemble_rmse
            print(f"Replacing Median Imputation with Mean Imputation *improved* performance by {improvement:.4f} RMSE, indicating Mean Imputation is a better choice for 'total_bedrooms' in this setup.")
    else:
        print("No single ablation led to a significant degradation or improvement; all components appear to contribute synergistically to the robust performance of the baseline solution.")
