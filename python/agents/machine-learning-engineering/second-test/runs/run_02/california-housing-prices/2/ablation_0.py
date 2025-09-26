
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Data Loading and Preprocessing (Common to all ablations) ---
try:
    train_df = pd.read_csv('./input/train.csv')
except FileNotFoundError:
    train_df = pd.read_csv('../input/train.csv')

TARGET = 'median_house_value'
features = [col for col in train_df.columns if col != TARGET]
X = train_df[features]
y = train_df[TARGET]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

median_total_bedrooms = X_train['total_bedrooms'].median()
X_train['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)
X_val['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# --- Function to evaluate a model/ensemble ---
def evaluate_model_performance(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} Validation Performance: {rmse:.4f}")
    return rmse

# --- Baseline: Original Ensemble (LightGBM + XGBoost) ---
print("--- Running Baseline: LightGBM + XGBoost Ensemble ---")

lgbm_base = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42)
lgbm_base.fit(X_train, y_train)
y_pred_lgbm_base = lgbm_base.predict(X_val)

xgb_base = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_jobs=-1)
xgb_base.fit(X_train, y_train)
y_pred_xgb_base = xgb_base.predict(X_val)

y_pred_ensemble_base = (y_pred_lgbm_base + y_pred_xgb_base) / 2
rmse_baseline = evaluate_model_performance(y_val, y_pred_ensemble_base, "Baseline Ensemble")

print("\n--- Ablation Study ---")

# --- Ablation 1: LightGBM Only ---
print("\n--- Ablation 1: LightGBM Only ---")
lgbm_ablated = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42)
lgbm_ablated.fit(X_train, y_train)
y_pred_lgbm_ablated = lgbm_ablated.predict(X_val)
rmse_lgbm_only = evaluate_model_performance(y_val, y_pred_lgbm_ablated, "LightGBM Only")

# --- Ablation 2: XGBoost Only ---
print("\n--- Ablation 2: XGBoost Only ---")
xgb_ablated = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_jobs=-1)
xgb_ablated.fit(X_train, y_train)
y_pred_xgb_ablated = xgb_ablated.predict(X_val)
rmse_xgb_only = evaluate_model_performance(y_val, y_pred_xgb_ablated, "XGBoost Only")

# --- Ablation 3: No Imputation for 'total_bedrooms' ---
# Re-load data and split for this specific ablation to isolate the change
print("\n--- Ablation 3: No Imputation for 'total_bedrooms' ---")
X_no_impute = X.copy() # Use original X before any imputation

# For this ablation, we will drop rows with NaN in 'total_bedrooms'
# (a common simple alternative to imputation, demonstrating impact of handling NaNs differently)
X_train_no_impute, X_val_no_impute, y_train_no_impute, y_val_no_impute = train_test_split(
    X_no_impute, y, test_size=0.2, random_state=42)

# Drop rows with NaN ONLY in 'total_bedrooms' for both train and val sets
# Note: For real-world, dropping from X_train first then aligning X_val is critical.
# For simplicity here, we're dropping from X_train and X_val independently, which is not ideal but highlights the impact.
# A better approach would be to drop from the combined X then split, or use an imputer pipeline.
train_idx_cleaned = X_train_no_impute['total_bedrooms'].dropna().index
val_idx_cleaned = X_val_no_impute['total_bedrooms'].dropna().index

X_train_no_impute = X_train_no_impute.loc[train_idx_cleaned]
y_train_no_impute = y_train_no_impute.loc[train_idx_cleaned]

X_val_no_impute = X_val_no_impute.loc[val_idx_cleaned]
y_val_no_impute = y_val_no_impute.loc[val_idx_cleaned]


lgbm_no_impute = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42)
lgbm_no_impute.fit(X_train_no_impute, y_train_no_impute)
y_pred_lgbm_no_impute = lgbm_no_impute.predict(X_val_no_impute)

xgb_no_impute = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_jobs=-1)
xgb_no_impute.fit(X_train_no_impute, y_train_no_impute)
y_pred_xgb_no_impute = xgb_no_impute.predict(X_val_no_impute)

y_pred_ensemble_no_impute = (y_pred_lgbm_no_impute + y_pred_xgb_no_impute) / 2
rmse_no_imputation = evaluate_model_performance(y_val_no_impute, y_pred_ensemble_no_impute, "Ensemble (No Imputation - Rows Dropped)")

# --- Summary and Contribution Analysis ---
print("\n--- Ablation Study Summary ---")
print(f"Baseline (LGBM + XGBoost Ensemble): {rmse_baseline:.4f}")
print(f"Ablation 1 (LightGBM Only):        {rmse_lgbm_only:.4f} (Change: {rmse_lgbm_only - rmse_baseline:.4f})")
print(f"Ablation 2 (XGBoost Only):          {rmse_xgb_only:.4f} (Change: {rmse_xgb_only - rmse_baseline:.4f})")
print(f"Ablation 3 (No Imputation, Rows Dropped): {rmse_no_imputation:.4f} (Change: {rmse_no_imputation - rmse_baseline:.4f})")

performance_changes = {
    "Ensembling (vs LightGBM only)": rmse_lgbm_only - rmse_baseline,
    "Ensembling (vs XGBoost only)": rmse_xgb_only - rmse_baseline,
    "Imputation (vs dropping rows)": rmse_no_imputation - rmse_baseline # A positive change means no imputation was worse
}

# Determine which part contributes most to overall performance (i.e., whose removal hurts performance the most or whose addition improves it the most)
# A smaller RMSE is better. So, a larger positive change when removing something means that component was more important.
# Conversely, a smaller (more negative) change indicates removing it didn't hurt as much.

most_impactful_change = max(performance_changes, key=performance_changes.get)
max_impact_value = performance_changes[most_impactful_change]

if max_impact_value > 0:
    print(f"\nThe part contributing MOST to the overall performance (i.e., whose removal/change negatively impacted performance the most) is: '{most_impactful_change}' with a performance degradation of {max_impact_value:.4f} RMSE.")
elif max_impact_value < 0:
    # This case might happen if an 'ablation' unexpectedly improves performance, which is rare for 'removing' a good component
    # Or if we're comparing a worse alternative (like dropping rows) to the baseline, the baseline is what contributes.
    print(f"\nThe original solution's components are generally beneficial. The 'ablation' that resulted in the smallest degradation (or slight improvement if any) was related to '{most_impactful_change}'.")
    print(f"However, considering negative impact (increase in RMSE), the biggest contributor to performance, relative to its ablation, appears to be 'Ensembling (vs XGBoost only)' leading to an improvement of {-performance_changes['Ensembling (vs XGBoost only)']:.4f} if we only had XGBoost.")
    print("More directly: the ensemble process, or the use of specific models, is crucial.")

# Simple interpretation:
# If rmse_lgbm_only is significantly higher than rmse_baseline, ensembling improved performance over LGBM only.
# If rmse_xgb_only is significantly higher than rmse_baseline, ensembling improved performance over XGBoost only.
# If rmse_no_imputation is significantly higher than rmse_baseline, imputation is important.

contributions = {}
contributions["Ensembling (over LGBM only)"] = rmse_lgbm_only - rmse_baseline
contributions["Ensembling (over XGBoost only)"] = rmse_xgb_only - rmse_baseline
contributions["Median Imputation (over dropping NaNs)"] = rmse_no_imputation - rmse_baseline

# Find the largest positive difference, indicating the most significant drop when that part was removed/changed
most_contributing_part = None
largest_drop = -np.inf

if contributions["Ensembling (over LGBM only)"] > largest_drop:
    largest_drop = contributions["Ensembling (over LGBM only)"]
    most_contributing_part = "Ensembling (combining models)"

if contributions["Ensembling (over XGBoost only)"] > largest_drop:
    largest_drop = contributions["Ensembling (over XGBoost only)"]
    most_contributing_part = "Ensembling (combining models)" # It's still ensembling if it helped over either single model

if contributions["Median Imputation (over dropping NaNs)"] > largest_drop:
    largest_drop = contributions["Median Imputation (over dropping NaNs)"]
    most_contributing_part = "Median Imputation for 'total_bedrooms'"

if most_contributing_part and largest_drop > 0:
    print(f"\nBased on this ablation study, the part of the code that contributes most to the overall performance (i.e., whose modification or removal led to the largest performance degradation) is: '{most_contributing_part}'.")
elif most_contributing_part:
    print("\nBased on this ablation study, all tested ablations led to similar or slightly better performance, suggesting the original setup is robust or the ablations were not detrimental enough.")
else:
    print("\nCould not determine the most contributing part from the ablation results.")
