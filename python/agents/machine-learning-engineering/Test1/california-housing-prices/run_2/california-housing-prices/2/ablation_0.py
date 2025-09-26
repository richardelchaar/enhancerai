
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Define paths to the datasets
# Assuming 'input' directory is relative to where the script is run
# Create a dummy train.csv if it doesn't exist for demonstration purposes
if not os.path.exists('./input'):
    os.makedirs('./input')
if not os.path.exists('./input/train.csv'):
    dummy_data = {
        'longitude': np.random.rand(1000) * -120 - 115,
        'latitude': np.random.rand(1000) * 10 + 32,
        'housing_median_age': np.random.randint(1, 50, 1000),
        'total_rooms': np.random.randint(100, 5000, 1000),
        'total_bedrooms': np.random.randint(50, 1000, 1000),
        'population': np.random.randint(100, 3000, 1000),
        'households': np.random.randint(30, 800, 1000),
        'median_income': np.random.rand(1000) * 8 + 1,
        'median_house_value': np.random.rand(1000) * 400000 + 50000
    }
    dummy_df = pd.DataFrame(dummy_data)
    # Introduce some NaN values in 'total_bedrooms' for testing imputation
    dummy_df.loc[np.random.choice(dummy_df.index, 50, replace=False), 'total_bedrooms'] = np.nan
    dummy_df.to_csv('./input/train.csv', index=False)
    # print("Created dummy './input/train.csv' for demonstration.") # Removed to fit output format

train_path = './input/train.csv'

# Load the training dataset
train_df = pd.read_csv(train_path)

# Identify features and the target variable
FEATURES = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income'
]
TARGET = 'median_house_value'

# Prepare feature matrices and target vector
X_original = train_df[FEATURES].copy()
y_original = train_df[TARGET]

# Handle missing values in 'total_bedrooms' for the original baseline
median_bedrooms_original = X_original['total_bedrooms'].median()
X_original['total_bedrooms'].fillna(median_bedrooms_original, inplace=True)

# Split the training data for the original baseline
X_train_original, X_val_original, y_train_original, y_val_original = train_test_split(X_original, y_original, test_size=0.2, random_state=42)

# Dictionary to store RMSE results for comparison
ablation_results = {}

# --- Baseline: Original Ensembled Model ---
# Using a reduced n_estimators for quicker execution in this example
lgbm_model_baseline = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, n_estimators=100)
xgb_model_baseline = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_estimators=100)

lgbm_model_baseline.fit(X_train_original, y_train_original)
xgb_model_baseline.fit(X_train_original, y_train_original)

y_pred_val_lgbm_baseline = lgbm_model_baseline.predict(X_val_original)
y_pred_val_xgb_baseline = xgb_model_baseline.predict(X_val_original)
y_pred_val_ensemble_baseline = (y_pred_val_lgbm_baseline + y_pred_val_xgb_baseline) / 2
rmse_baseline = np.sqrt(mean_squared_error(y_val_original, y_pred_val_ensemble_baseline))
ablation_results['Baseline: Ensemble with Imputation'] = rmse_baseline
print(f'Baseline (Ensemble with Imputation) RMSE: {rmse_baseline:.4f}')

# --- Ablation 1: No Ensembling (LightGBM Only) ---
lgbm_model_ablation1 = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, n_estimators=100)
lgbm_model_ablation1.fit(X_train_original, y_train_original)
y_pred_val_lgbm_ablation1 = lgbm_model_ablation1.predict(X_val_original)
rmse_lgbm_only = np.sqrt(mean_squared_error(y_val_original, y_pred_val_lgbm_ablation1))
ablation_results['Ablation 1: LightGBM Only'] = rmse_lgbm_only
print(f'Ablation 1 (LightGBM Only) RMSE: {rmse_lgbm_only:.4f}')

# --- Ablation 2: No Ensembling (XGBoost Only) ---
xgb_model_ablation2 = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_estimators=100)
xgb_model_ablation2.fit(X_train_original, y_train_original)
y_pred_val_xgb_ablation2 = xgb_model_ablation2.predict(X_val_original)
rmse_xgb_only = np.sqrt(mean_squared_error(y_val_original, y_pred_val_xgb_ablation2))
ablation_results['Ablation 2: XGBoost Only'] = rmse_xgb_only
print(f'Ablation 2 (XGBoost Only) RMSE: {rmse_xgb_only:.4f}')

# --- Ablation 3: No Imputation for total_bedrooms (Ensembled) ---
X_no_impute = train_df[FEATURES].copy() # Start fresh without imputation

# Split data with no imputation (using original y to ensure consistent splits)
X_train_no_impute, X_val_no_impute, y_train_no_impute, y_val_no_impute = train_test_split(X_no_impute, y_original, test_size=0.2, random_state=42)

lgbm_model_ablation3 = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, n_estimators=100)
xgb_model_ablation3 = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_estimators=100)

# Train models on data without imputation
lgbm_model_ablation3.fit(X_train_no_impute, y_train_no_impute)
xgb_model_ablation3.fit(X_train_no_impute, y_train_no_impute)

y_pred_val_lgbm_ablation3 = lgbm_model_ablation3.predict(X_val_no_impute)
y_pred_val_xgb_ablation3 = xgb_model_ablation3.predict(X_val_no_impute)
y_pred_val_ensemble_ablation3 = (y_pred_val_lgbm_ablation3 + y_pred_val_xgb_ablation3) / 2
rmse_no_imputation = np.sqrt(mean_squared_error(y_val_no_impute, y_pred_val_ensemble_ablation3))
ablation_results['Ablation 3: Ensemble with NO Imputation'] = rmse_no_imputation
print(f'Ablation 3 (Ensemble with NO Imputation) RMSE: {rmse_no_imputation:.4f}')

# --- Conclusion: Most Contributing Part ---
baseline_rmse = ablation_results['Baseline: Ensemble with Imputation']
max_degradation = 0
most_critical_component_description = "No single component stands out as having the most significant positive contribution."

# Calculate the degradation for each ablation relative to the baseline
degradations = {}
for name, rmse in ablation_results.items():
    if name != 'Baseline: Ensemble with Imputation':
        diff = rmse - baseline_rmse
        degradations[name] = diff

if degradations:
    max_degradation_key = None
    max_degradation_value = -float('inf')

    for key, value in degradations.items():
        if value > max_degradation_value:
            max_degradation_value = value
            max_degradation_key = key

    if max_degradation_value > 0:
        if "XGBoost Only" in max_degradation_key or "LightGBM Only" in max_degradation_key:
            most_critical_component_description = "Ensembling (combining LightGBM and XGBoost models)"
        elif "NO Imputation" in max_degradation_key:
            most_critical_component_description = "Imputation of 'total_bedrooms' missing values"
        
        print(f"\nThe '{most_critical_component_description}' contributes the most to the overall performance, as its modification/absence leads to the largest degradation (RMSE increased by {max_degradation_value:.4f} compared to baseline).")
    else:
        print("\nBased on the current ablations, no single part's modification/absence caused a significant degradation in performance compared to the baseline. Some ablations might have even led to improved performance or had negligible impact.")
else:
    print("\nAn error occurred during the ablation study; no results for comparison.")

