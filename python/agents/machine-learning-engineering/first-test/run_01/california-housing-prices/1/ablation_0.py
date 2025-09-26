
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load datasets
train_df = pd.read_csv("./input/train.csv")

# Preprocessing: Handle missing values (Baseline strategy for first two ablations)
# A copy of the DataFrame is made to allow for different imputation strategies in Ablation 3.
train_df_baseline_processed = train_df.copy()
train_df_baseline_processed['total_bedrooms'].fillna(train_df_baseline_processed['total_bedrooms'].median(), inplace=True)

# Define features (X) and target (y)
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income']
target = 'median_house_value'

X_baseline = train_df_baseline_processed[features]
y_baseline = train_df_baseline_processed[target]

# Split training data into a training set and a hold-out validation set for baseline and ablations 1 & 2
X_train_baseline, X_val_baseline, y_train_baseline, y_val_baseline = train_test_split(X_baseline, y_baseline, test_size=0.2, random_state=42)

print("--- Ablation Study ---")

# --- Baseline Performance (Original Ensemble with Median Imputation) ---
print("\n--- Baseline: LightGBM + XGBoost Ensemble (Median Imputation for total_bedrooms) ---")

# Initialize models
lgbm_model_baseline = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42)
xgb_model_baseline = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_jobs=-1)

# Train models
lgbm_model_baseline.fit(X_train_baseline, y_train_baseline)
xgb_model_baseline.fit(X_train_baseline, y_train_baseline)

# Predict and ensemble
y_pred_val_lgbm_baseline = lgbm_model_baseline.predict(X_val_baseline)
y_pred_val_xgb_baseline = xgb_model_baseline.predict(X_val_baseline)
y_pred_val_ensemble_baseline = (y_pred_val_lgbm_baseline + y_pred_val_xgb_baseline) / 2

# Evaluate
rmse_val_ensemble_baseline = np.sqrt(mean_squared_error(y_val_baseline, y_pred_val_ensemble_baseline))
print(f"1. Baseline (Ensemble with Median Imputation) Validation RMSE: {rmse_val_ensemble_baseline:.4f}")

# --- Ablation 1: LightGBM Only (with Median Imputation) ---
print("\n--- Ablation 1: LightGBM Only (Median Imputation for total_bedrooms) ---")

# Initialize and train LightGBM model
lgbm_model_ablation1 = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42)
lgbm_model_ablation1.fit(X_train_baseline, y_train_baseline) # Use baseline split with median imputation

# Predict and evaluate
y_pred_val_lgbm_ablation1 = lgbm_model_ablation1.predict(X_val_baseline)
rmse_val_lgbm_ablation1 = np.sqrt(mean_squared_error(y_val_baseline, y_pred_val_lgbm_ablation1))
print(f"2. Ablation 1 (LightGBM Only) Validation RMSE: {rmse_val_lgbm_ablation1:.4f}")

# --- Ablation 2: XGBoost Only (with Median Imputation) ---
print("\n--- Ablation 2: XGBoost Only (Median Imputation for total_bedrooms) ---")

# Initialize and train XGBoost model
xgb_model_ablation2 = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_jobs=-1)
xgb_model_ablation2.fit(X_train_baseline, y_train_baseline) # Use baseline split with median imputation

# Predict and evaluate
y_pred_val_xgb_ablation2 = xgb_model_ablation2.predict(X_val_baseline)
rmse_val_xgb_ablation2 = np.sqrt(mean_squared_error(y_val_baseline, y_pred_val_xgb_ablation2))
print(f"3. Ablation 2 (XGBoost Only) Validation RMSE: {rmse_val_xgb_ablation2:.4f}")

# --- Ablation 3: Different Missing Value Imputation (fill with 0 instead of median, with Ensemble) ---
print("\n--- Ablation 3: Ensemble with Missing Value Imputation (Fill total_bedrooms with 0) ---")

train_df_ablation3 = train_df.copy() # Start from original raw data
train_df_ablation3['total_bedrooms'].fillna(0, inplace=True) # Fill with 0 for this ablation
X_ablation3 = train_df_ablation3[features]
y_ablation3 = train_df_ablation3[target]
X_train_ablation3, X_val_ablation3, y_train_ablation3, y_val_ablation3 = train_test_split(X_ablation3, y_ablation3, test_size=0.2, random_state=42)

# Initialize and train ensemble models with the new data (0 imputation)
lgbm_model_ablation3 = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42)
xgb_model_ablation3 = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_jobs=-1)

lgbm_model_ablation3.fit(X_train_ablation3, y_train_ablation3)
xgb_model_ablation3.fit(X_train_ablation3, y_train_ablation3)

y_pred_val_lgbm_ablation3 = lgbm_model_ablation3.predict(X_val_ablation3)
y_pred_val_xgb_ablation3 = xgb_model_ablation3.predict(X_val_ablation3)
y_pred_val_ensemble_ablation3 = (y_pred_val_lgbm_ablation3 + y_pred_val_xgb_ablation3) / 2

rmse_val_ensemble_ablation3 = np.sqrt(mean_squared_error(y_val_ablation3, y_pred_val_ensemble_ablation3))
print(f"4. Ablation 3 (Ensemble with Imputation by 0) Validation RMSE: {rmse_val_ensemble_ablation3:.4f}")

# --- Conclusion on contribution ---
# Calculate the performance difference (degradation) when a part is removed/changed from the baseline.
# A higher positive difference indicates a stronger positive contribution of the original part.

degradation_from_removing_xgb = rmse_val_lgbm_ablation1 - rmse_val_ensemble_baseline
degradation_from_removing_lgbm = rmse_val_xgb_ablation2 - rmse_val_ensemble_baseline
degradation_from_changing_imputation = rmse_val_ensemble_ablation3 - rmse_val_ensemble_baseline

print("\n--- Analysis of Contributions ---")

contributions_details = {}

if degradation_from_removing_xgb > 0:
    contributions_details['XGBoost Model in Ensemble'] = degradation_from_removing_xgb
if degradation_from_removing_lgbm > 0:
    contributions_details['LightGBM Model in Ensemble'] = degradation_from_removing_lgbm
if degradation_from_changing_imputation > 0:
    contributions_details['Median Imputation for total_bedrooms'] = degradation_from_changing_imputation

if not contributions_details:
    # This scenario means baseline was not the best, or no significant degradation from ablations
    best_overall_rmse = min(rmse_val_ensemble_baseline, rmse_val_lgbm_ablation1, rmse_val_xgb_ablation2, rmse_val_ensemble_ablation3)
    if best_overall_rmse == rmse_val_lgbm_ablation1:
        print("Based on this ablation study: The LightGBM model alone performs best, outperforming the ensemble and XGBoost alone.")
        print("This suggests that the XGBoost model or the simple averaging might be negatively impacting the ensemble performance.")
    elif best_overall_rmse == rmse_val_xgb_ablation2:
        print("Based on this ablation study: The XGBoost model alone performs best, outperforming the ensemble and LightGBM alone.")
        print("This suggests that the LightGBM model or the simple averaging might be negatively impacting the ensemble performance.")
    elif best_overall_rmse == rmse_val_ensemble_ablation3:
        print("Based on this ablation study: The ensemble combined with filling 'total_bedrooms' with 0 performs best.")
        print("This indicates that filling missing 'total_bedrooms' values with 0 is a better imputation strategy than using the median for this dataset.")
    else: # Baseline is still the best or equal to the best, but no individual "part" caused a significant positive degradation when removed.
        print("The baseline solution (LightGBM + XGBoost ensemble with median imputation) performs optimally.")
        print("The current ablations suggest a balanced contribution from its components, as no single removed/modified part caused a significantly worse performance compared to others, nor did any ablation clearly improve performance.")
else:
    most_contributing_part_label = max(contributions_details, key=contributions_details.get)
    highest_degradation = contributions_details[most_contributing_part_label]

    print(f"\nWhich part of the code contributes the most to the overall performance:")
    print(f"The part that contributes most is: '{most_contributing_part_label}'.")
    print(f"Its presence (or specific strategy) in the baseline solution led to an improvement in RMSE by approximately {highest_degradation:.4f} compared to its ablated version.")

    if 'in Ensemble' in most_contributing_part_label:
        print("This highlights the effectiveness of the ensembling strategy, where the combination of models (or specifically that model's contribution to the ensemble) yields better results than individual models.")
    elif 'Imputation' in most_contributing_part_label:
        print("This highlights the importance of the chosen data preprocessing step (missing value imputation) for optimal model performance.")

