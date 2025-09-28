
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


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Load data
try:
    train_df = pd.read_csv('./input/train.csv')
    test_df = pd.read_csv('./input/test.csv')
except FileNotFoundError:
    print("Error: Ensure 'train.csv' and 'test.csv' are located in the './input/' directory.")
    # In a real scenario, this might trigger a more robust error handling or exit.
    # For this exercise, we assume the files exist as per instructions.
    raise

# Separate target variable
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']
X_test = test_df.copy()

# Define feature engineering function
# This function creates new features based on existing ones.
# Imputation for 'total_bedrooms' is handled separately as it's part of the ablation.
def create_additional_features(df):
    df_copy = df.copy()
    df_copy['rooms_per_household'] = df_copy['total_rooms'] / df_copy['households']
    df_copy['population_per_household'] = df_copy['population'] / df_copy['households']
    # 'bedrooms_per_room' is created after 'total_bedrooms' is imputed
    return df_copy

# --- Ablation Study Setup ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {} # To store RMSE for each scenario

# --- BASELINE: Median Imputation, Ensemble Model ---
print("Running Baseline: Median Imputation, Ensemble Model...")
fold_rmses_baseline = []
test_preds_baseline_folds = [] # Store test predictions for each fold to average later

# Preprocessing for baseline (median imputation) - applied once for feature engineering preparation
X_baseline_fe = create_additional_features(X)
X_test_baseline_fe = create_additional_features(X_test)

# Impute 'total_bedrooms' with median before calculating 'bedrooms_per_room'
imputer_median_baseline = SimpleImputer(strategy='median')
X_baseline_fe['total_bedrooms'] = imputer_median_baseline.fit_transform(X_baseline_fe[['total_bedrooms']])
X_test_baseline_fe['total_bedrooms'] = imputer_median_baseline.transform(X_test_baseline_fe[['total_bedrooms']])
X_baseline_fe['bedrooms_per_room'] = X_baseline_fe['total_bedrooms'] / X_baseline_fe['total_rooms']
X_test_baseline_fe['bedrooms_per_room'] = X_test_baseline_fe['total_bedrooms'] / X_test_baseline_fe['total_rooms']

for fold, (train_idx, val_idx) in enumerate(kf.split(X_baseline_fe)):
    X_train, X_val = X_baseline_fe.iloc[train_idx], X_baseline_fe.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Scaling for current fold
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled_fold = scaler.transform(X_test_baseline_fe) # Scale test data using this fold's scaler

    # LightGBM model
    lgbm = lgb.LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=-1)
    lgbm.fit(X_train_scaled, y_train)

    # RandomForest model
    rf = RandomForestRegressor(random_state=42, n_estimators=1000, max_depth=10, n_jobs=-1, verbose=0)
    rf.fit(X_train_scaled, y_train)

    # Ensemble prediction (simple average)
    y_pred_lgbm = lgbm.predict(X_val_scaled)
    y_pred_rf = rf.predict(X_val_scaled)
    y_pred_val = (y_pred_lgbm + y_pred_rf) / 2

    fold_rmses_baseline.append(np.sqrt(mean_squared_error(y_val, y_pred_val)))

    # Predict on test data for this fold
    test_pred_lgbm = lgbm.predict(X_test_scaled_fold)
    test_pred_rf = rf.predict(X_test_scaled_fold)
    test_preds_baseline_folds.append((test_pred_lgbm + test_pred_rf) / 2)

mean_rmse_baseline = np.mean(fold_rmses_baseline)
results['baseline'] = mean_rmse_baseline
print(f"Baseline (Median Imputation, Ensemble) RMSE: {mean_rmse_baseline:.4f}")

# --- ABLATION 1: Mean Imputation, Ensemble Model ---
print("\nRunning Ablation 1: Mean Imputation, Ensemble Model...")
fold_rmses_ablation1 = []
test_preds_ablation1_folds = []

# Preprocessing for ablation 1 (mean imputation)
X_ablation1_fe = create_additional_features(X)
X_test_ablation1_fe = create_additional_features(X_test)

# Impute 'total_bedrooms' with mean
imputer_mean_ablation1 = SimpleImputer(strategy='mean')
X_ablation1_fe['total_bedrooms'] = imputer_mean_ablation1.fit_transform(X_ablation1_fe[['total_bedrooms']])
X_test_ablation1_fe['total_bedrooms'] = imputer_mean_ablation1.transform(X_test_ablation1_fe[['total_bedrooms']])
X_ablation1_fe['bedrooms_per_room'] = X_ablation1_fe['total_bedrooms'] / X_ablation1_fe['total_rooms']
X_test_ablation1_fe['bedrooms_per_room'] = X_test_ablation1_fe['total_bedrooms'] / X_test_ablation1_fe['total_rooms']

for fold, (train_idx, val_idx) in enumerate(kf.split(X_ablation1_fe)):
    X_train, X_val = X_ablation1_fe.iloc[train_idx], X_ablation1_fe.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Scaling for current fold
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled_fold = scaler.transform(X_test_ablation1_fe)

    # LightGBM model
    lgbm = lgb.LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=-1)
    lgbm.fit(X_train_scaled, y_train)

    # RandomForest model
    rf = RandomForestRegressor(random_state=42, n_estimators=1000, max_depth=10, n_jobs=-1, verbose=0)
    rf.fit(X_train_scaled, y_train)

    # Ensemble prediction (simple average)
    y_pred_lgbm = lgbm.predict(X_val_scaled)
    y_pred_rf = rf.predict(X_val_scaled)
    y_pred_val = (y_pred_lgbm + y_pred_rf) / 2

    fold_rmses_ablation1.append(np.sqrt(mean_squared_error(y_val, y_pred_val)))

    # Predict on test data for this fold
    test_pred_lgbm = lgbm.predict(X_test_scaled_fold)
    test_pred_rf = rf.predict(X_test_scaled_fold)
    test_preds_ablation1_folds.append((test_pred_lgbm + test_pred_rf) / 2)

mean_rmse_ablation1 = np.mean(fold_rmses_ablation1)
results['ablation1'] = mean_rmse_ablation1
print(f"Ablation 1 (Mean Imputation, Ensemble) RMSE: {mean_rmse_ablation1:.4f}")

# --- ABLATION 2: Median Imputation, LightGBM Only ---
print("\nRunning Ablation 2: Median Imputation, LightGBM Only...")
fold_rmses_ablation2 = []
test_preds_ablation2_folds = []

# Preprocessing for ablation 2 (median imputation)
X_ablation2_fe = create_additional_features(X)
X_test_ablation2_fe = create_additional_features(X_test)

# Impute 'total_bedrooms' with median
imputer_median_ablation2 = SimpleImputer(strategy='median')
X_ablation2_fe['total_bedrooms'] = imputer_median_ablation2.fit_transform(X_ablation2_fe[['total_bedrooms']])
X_test_ablation2_fe['total_bedrooms'] = imputer_median_ablation2.transform(X_test_ablation2_fe[['total_bedrooms']])
X_ablation2_fe['bedrooms_per_room'] = X_ablation2_fe['total_bedrooms'] / X_ablation2_fe['total_rooms']
X_test_ablation2_fe['bedrooms_per_room'] = X_test_ablation2_fe['total_bedrooms'] / X_test_ablation2_fe['total_rooms']

for fold, (train_idx, val_idx) in enumerate(kf.split(X_ablation2_fe)):
    X_train, X_val = X_ablation2_fe.iloc[train_idx], X_ablation2_fe.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Scaling for current fold
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled_fold = scaler.transform(X_test_ablation2_fe)

    # Only LightGBM model
    lgbm = lgb.LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=-1)
    lgbm.fit(X_train_scaled, y_train)

    y_pred_val = lgbm.predict(X_val_scaled)
    fold_rmses_ablation2.append(np.sqrt(mean_squared_error(y_val, y_pred_val)))

    # Predict on test data for this fold
    test_preds_ablation2_folds.append(lgbm.predict(X_test_scaled_fold))

mean_rmse_ablation2 = np.mean(fold_rmses_ablation2)
results['ablation2'] = mean_rmse_ablation2
print(f"Ablation 2 (Median Imputation, LightGBM Only) RMSE: {mean_rmse_ablation2:.4f}")

# --- Determine Largest Change and Final Performance ---
print("\n--- Ablation Results Summary ---")
print(f"Baseline RMSE: {results['baseline']:.4f}")
print(f"Ablation 1 (Mean Imputation vs. Median) RMSE: {results['ablation1']:.4f}")
print(f"Ablation 2 (LightGBM Only vs. Ensemble) RMSE: {results['ablation2']:.4f}")

performance_changes = {
    'Missing Value Imputation Strategy (Median vs. Mean)': abs(results['baseline'] - results['ablation1']),
    'Model Ensembling (Ensemble vs. LightGBM Only)': abs(results['baseline'] - results['ablation2'])
}

largest_change_component = max(performance_changes, key=performance_changes.get)
largest_change_value = performance_changes[largest_change_component]

print(f"\nThe component whose ablation resulted in the largest change in performance was: '{largest_change_component}' (Change: {largest_change_value:.4f})")

# Select the best performing scenario based on RMSE
best_scenario_name = 'baseline'
best_rmse = results['baseline']
final_test_preds = np.mean(test_preds_baseline_folds, axis=0) # Initialize with baseline predictions

if results['ablation1'] < best_rmse:
    best_rmse = results['ablation1']
    best_scenario_name = 'ablation1'
    final_test_preds = np.mean(test_preds_ablation1_folds, axis=0)
if results['ablation2'] < best_rmse:
    best_rmse = results['ablation2']
    best_scenario_name = 'ablation2'
    final_test_preds = np.mean(test_preds_ablation2_folds, axis=0)

print(f"\nBest performing scenario: '{best_scenario_name}' with RMSE: {best_rmse:.4f}")

# Final validation performance should be the RMSE of the best performing scenario
final_validation_score = best_rmse
print(f"Final Validation Performance: {final_validation_score}")

# Ensure predictions are non-negative
final_test_preds[final_test_preds < 0] = 0

# Create submission file
submission_df = pd.DataFrame({'median_house_value': final_test_preds})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created.")
