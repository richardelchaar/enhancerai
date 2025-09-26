
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Load dataset
train_df = pd.read_csv("./input/train.csv")

# Separate target variable from features in the training data
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

def run_ablation_experiment(X_full, y_full, imputation_strategy='median', use_lgbm=True, use_xgb=True):
    """
    Runs a single ablation experiment based on specified parameters.

    Args:
        X_full (pd.DataFrame): Full feature set.
        y_full (pd.Series): Full target variable.
        imputation_strategy (str): 'median' or 'mean' for 'total_bedrooms' imputation.
        use_lgbm (bool): Whether to include LightGBM in the ensemble.
        use_xgb (bool): Whether to include XGBoost in the ensemble.

    Returns:
        float: The RMSE of the evaluated model/ensemble.
    """
    X = X_full.copy()
    y = y_full.copy()

    # Handle missing values for 'total_bedrooms' based on strategy
    if imputation_strategy == 'median':
        imputation_value = X['total_bedrooms'].median()
    elif imputation_strategy == 'mean':
        imputation_value = X['total_bedrooms'].mean()
    else:
        raise ValueError("Invalid imputation_strategy. Choose 'median' or 'mean'.")

    X['total_bedrooms'].fillna(imputation_value, inplace=True)

    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred_val_ensemble = np.zeros_like(y_val, dtype=float)
    model_count = 0

    # Train and predict with LightGBM if enabled
    if use_lgbm:
        lgbm_model = lgb.LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
        lgbm_model.fit(X_train, y_train)
        y_pred_val_ensemble += lgbm_model.predict(X_val)
        model_count += 1

    # Train and predict with XGBoost if enabled
    if use_xgb:
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        y_pred_val_ensemble += xgb_model.predict(X_val)
        model_count += 1

    # Handle case where no models are selected (though our scenarios will always have at least one)
    if model_count == 0:
        return float('inf') # Indicate an error or invalid configuration

    # Ensemble predictions
    y_pred_val_ensemble /= model_count

    # Evaluate the ensembled model
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble))
    return rmse

# --- Perform Ablation Study ---
results = {}

# 1. Baseline: Original Solution (LGBM + XGBoost Ensemble, Median Imputation)
baseline_rmse = run_ablation_experiment(X, y, imputation_strategy='median', use_lgbm=True, use_xgb=True)
results['Baseline (LGBM + XGBoost, Median Imputation)'] = baseline_rmse

# 2. Ablation: Only LightGBM (Median Imputation)
lgbm_only_rmse = run_ablation_experiment(X, y, imputation_strategy='median', use_lgbm=True, use_xgb=False)
results['Ablation 1: Only LightGBM (Median Imputation)'] = lgbm_only_rmse

# 3. Ablation: Only XGBoost (Median Imputation)
xgb_only_rmse = run_ablation_experiment(X, y, imputation_strategy='median', use_lgbm=False, use_xgb=True)
results['Ablation 2: Only XGBoost (Median Imputation)'] = xgb_only_rmse

# 4. Ablation: Ensemble with Mean Imputation instead of Median
mean_imputation_ensemble_rmse = run_ablation_experiment(X, y, imputation_strategy='mean', use_lgbm=True, use_xgb=True)
results['Ablation 3: Ensemble (LGBM + XGBoost), Mean Imputation'] = mean_imputation_ensemble_rmse

# --- Print Results ---
print("--- Ablation Study Results ---")
for scenario, rmse in results.items():
    print(f"- {scenario}: RMSE = {rmse:.4f}")

print("\n--- Analysis of Contributions ---")

# Determine the best performing scenario (lowest RMSE)
best_scenario_name = min(results, key=results.get)
best_rmse = results[best_scenario_name]

print(f"The baseline ensemble with median imputation achieved an RMSE of {baseline_rmse:.4f}.")

# Compare individual models to ensemble
if baseline_rmse < lgbm_only_rmse and baseline_rmse < xgb_only_rmse:
    print(f"The ensemble of LightGBM and XGBoost ({baseline_rmse:.4f} RMSE) outperforms using only LightGBM ({lgbm_only_rmse:.4f} RMSE) or only XGBoost ({xgb_only_rmse:.4f} RMSE).")
    print("This suggests that the ensembling strategy (combining both models) significantly contributes to performance.")
elif lgbm_only_rmse < baseline_rmse:
    print(f"Interestingly, using only LightGBM ({lgbm_only_rmse:.4f} RMSE) performed better than the ensemble. This suggests XGBoost might be degrading performance in the ensemble or the ensembling weight needs tuning.")
elif xgb_only_rmse < baseline_rmse:
    print(f"Interestingly, using only XGBoost ({xgb_only_rmse:.4f} RMSE) performed better than the ensemble. This suggests LightGBM might be degrading performance in the ensemble or the ensembling weight needs tuning.")
else:
    print("Individual models (LightGBM or XGBoost) perform similarly or slightly worse than the ensemble, indicating some benefit from ensembling.")

# Compare imputation strategies
if baseline_rmse < mean_imputation_ensemble_rmse:
    print(f"Using median imputation ({baseline_rmse:.4f} RMSE) resulted in better performance compared to mean imputation ({mean_imputation_ensemble_rmse:.4f} RMSE).")
    print("Therefore, median imputation for 'total_bedrooms' is a valuable preprocessing step contributing to the overall performance.")
elif mean_imputation_ensemble_rmse < baseline_rmse:
    print(f"Using mean imputation ({mean_imputation_ensemble_rmse:.4f} RMSE) performed slightly better than median imputation ({baseline_rmse:.4f} RMSE).")
    print("This suggests that mean imputation might be a marginally better preprocessing step in this specific scenario.")
else:
    print(f"Median and mean imputation strategies yield very similar performance ({baseline_rmse:.4f} vs {mean_imputation_ensemble_rmse:.4f} RMSE), indicating that the choice of median vs. mean for 'total_bedrooms' might not be a major differentiator in this setup.")

# Overall conclusion on the most contributing part based on degradation from baseline
contributions = {
    "Ensembling (removing one model)": max(lgbm_only_rmse, xgb_only_rmse) - baseline_rmse,
    "Median Imputation (vs. Mean)": mean_imputation_ensemble_rmse - baseline_rmse
}

if contributions["Ensembling (removing one model)"] > contributions["Median Imputation (vs. Mean)"]:
    print("\nBased on this ablation study, the most significant contribution to the overall performance comes from the ensembling of both LightGBM and XGBoost models. Removing either model individually leads to a larger drop in performance compared to changing the imputation strategy.")
elif contributions["Median Imputation (vs. Mean)"] > contributions["Ensembling (removing one model)"]:
    print("\nBased on this ablation study, the most significant contribution to the overall performance comes from using median imputation for 'total_bedrooms'. Changing to mean imputation leads to a larger drop in performance compared to running individual models instead of the ensemble.")
else:
    print("\nBased on this ablation study, both the ensembling strategy and the median imputation for 'total_bedrooms' provide similar levels of contribution to the overall performance, as their removal/modification leads to comparable changes in RMSE.")

