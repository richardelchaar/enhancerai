
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
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# --- Helper function for consistent RMSE calculation ---
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# --- Create a dummy train.csv if it doesn't exist for reproducibility ---
# This ensures the script is runnable even without the actual dataset.
# In a real scenario, you would have your 'train.csv' already in 'input/'
if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/train.csv"):
    print("Creating a dummy 'train.csv' for demonstration purposes.")
    num_samples = 200
    dummy_data = {
        'feature_A': np.random.rand(num_samples) * 100,
        'feature_B': np.random.rand(num_samples) * 50,
        'total_bedrooms': np.random.rand(num_samples) * 10,
        'rooms_per_household': np.random.rand(num_samples) * 5,
        'population': np.random.randint(100, 5000, num_samples),
        'households': np.random.randint(50, 1000, num_samples),
        'median_income': np.random.rand(num_samples) * 10,
        'median_house_value': np.random.rand(num_samples) * 100000 + 150000
    }
    # Introduce some NaNs in total_bedrooms for testing imputation
    nan_indices = np.random.choice(num_samples, int(num_samples * 0.05), replace=False)
    dummy_data['total_bedrooms'][nan_indices] = np.nan
    pd.DataFrame(dummy_data).to_csv("./input/train.csv", index=False)
else:
    print("Using existing 'train.csv'.")

# ==============================================================================
# ===== BASELINE: Original Code Performance =====
# ==============================================================================
print("Running Baseline: Original Code...")

# Load the training data
train_df_baseline = pd.read_csv("./input/train.csv")

# Handle missing values: Fill 'total_bedrooms' with the median (ORIGINAL STRATEGY)
if 'total_bedrooms' in train_df_baseline.columns:
    train_df_baseline['total_bedrooms'].fillna(train_df_baseline['total_bedrooms'].median(), inplace=True)

# Define features (X) and target (y)
X_baseline = train_df_baseline.drop('median_house_value', axis=1)
y_baseline = train_df_baseline['median_house_value']

# Split the training data into a training set and a hold-out validation set
X_train_baseline, X_val_baseline, y_train_baseline, y_val_baseline = train_test_split(X_baseline, y_baseline, test_size=0.2, random_state=42)

# --- Model Training: LightGBM ---
lgbm_baseline = lgb.LGBMRegressor(objective='regression',
                                  metric='rmse',
                                  random_state=42,
                                  n_jobs=-1,
                                  verbose=-1)
lgbm_baseline.fit(X_train_baseline, y_train_baseline)
y_pred_lgbm_baseline = lgbm_baseline.predict(X_val_baseline)

# --- Model Training: XGBoost ---
xgb_model_baseline = xgb.XGBRegressor(objective='reg:squarederror',
                                      eval_metric='rmse',
                                      random_state=42,
                                      verbosity=0)
xgb_model_baseline.fit(X_train_baseline, y_train_baseline)
y_pred_xgb_baseline = xgb_model_baseline.predict(X_val_baseline)

# --- Model Ensembling (ORIGINAL STRATEGY: Averaging LightGBM and XGBoost) ---
y_pred_ensemble_baseline = (y_pred_lgbm_baseline + y_pred_xgb_baseline) / 2

# Calculate RMSE
rmse_val_ensemble_baseline = calculate_rmse(y_val_baseline, y_pred_ensemble_baseline)
print(f"Baseline Performance (RMSE): {rmse_val_ensemble_baseline:.4f}")
baseline_score = rmse_val_ensemble_baseline

# ==============================================================================
# ===== ABLATION 1: Change Missing Value Imputation Strategy (Median -> Mean) =====
# ==============================================================================
print("\nRunning Ablation 1: Missing value imputation (total_bedrooms: Median -> Mean)...")

# Load the training data
train_df_ablation1 = pd.read_csv("./input/train.csv")

# Handle missing values: Fill 'total_bedrooms' with the MEAN (ABLATION)
if 'total_bedrooms' in train_df_ablation1.columns:
    train_df_ablation1['total_bedrooms'].fillna(train_df_ablation1['total_bedrooms'].mean(), inplace=True) # MODIFIED LINE

# Define features (X) and target (y)
X_ablation1 = train_df_ablation1.drop('median_house_value', axis=1)
y_ablation1 = train_df_ablation1['median_house_value']

# Split the training data into a training set and a hold-out validation set
X_train_ablation1, X_val_ablation1, y_train_ablation1, y_val_ablation1 = train_test_split(X_ablation1, y_ablation1, test_size=0.2, random_state=42)

# --- Model Training: LightGBM ---
lgbm_ablation1 = lgb.LGBMRegressor(objective='regression',
                                   metric='rmse',
                                   random_state=42,
                                   n_jobs=-1,
                                   verbose=-1)
lgbm_ablation1.fit(X_train_ablation1, y_train_ablation1)
y_pred_lgbm_ablation1 = lgbm_ablation1.predict(X_val_ablation1)

# --- Model Training: XGBoost ---
xgb_model_ablation1 = xgb.XGBRegressor(objective='reg:squarederror',
                                       eval_metric='rmse',
                                       random_state=42,
                                       verbosity=0)
xgb_model_ablation1.fit(X_train_ablation1, y_train_ablation1)
y_pred_xgb_ablation1 = xgb_model_ablation1.predict(X_val_ablation1)

# --- Model Ensembling (Original strategy: Averaging LightGBM and XGBoost) ---
y_pred_ensemble_ablation1 = (y_pred_lgbm_ablation1 + y_pred_xgb_ablation1) / 2

# Calculate RMSE
rmse_val_ensemble_ablation1 = calculate_rmse(y_val_ablation1, y_pred_ensemble_ablation1)
print(f"Ablation 1 Performance (RMSE): {rmse_val_ensemble_ablation1:.4f}")
ablation_1_score = rmse_val_ensemble_ablation1

# ==============================================================================
# ===== ABLATION 2: Remove XGBoost from Ensemble (Use LightGBM only) =====
# ==============================================================================
print("\nRunning Ablation 2: Remove XGBoost from ensemble (LightGBM only)...")

# Load the training data
train_df_ablation2 = pd.read_csv("./input/train.csv")

# Handle missing values: Fill 'total_bedrooms' with the median (ORIGINAL STRATEGY)
if 'total_bedrooms' in train_df_ablation2.columns:
    train_df_ablation2['total_bedrooms'].fillna(train_df_ablation2['total_bedrooms'].median(), inplace=True)

# Define features (X) and target (y)
X_ablation2 = train_df_ablation2.drop('median_house_value', axis=1)
y_ablation2 = train_df_ablation2['median_house_value']

# Split the training data into a training set and a hold-out validation set
X_train_ablation2, X_val_ablation2, y_train_ablation2, y_val_ablation2 = train_test_split(X_ablation2, y_ablation2, test_size=0.2, random_state=42)

# --- Model Training: LightGBM ---
lgbm_ablation2 = lgb.LGBMRegressor(objective='regression',
                                   metric='rmse',
                                   random_state=42,
                                   n_jobs=-1,
                                   verbose=-1)
lgbm_ablation2.fit(X_train_ablation2, y_train_ablation2)
y_pred_lgbm_ablation2 = lgbm_ablation2.predict(X_val_ablation2)

# --- Model Training: XGBoost ---
# XGBoost training is skipped for this ablation.
# xgb_model_ablation2 = xgb.XGBRegressor(...)
# xgb_model_ablation2.fit(...)
# y_pred_xgb_ablation2 = xgb_model_ablation2.predict(...)

# --- Model Ensembling (ABLATION: Use LightGBM predictions directly, no XGBoost) ---
y_pred_ensemble_ablation2 = y_pred_lgbm_ablation2 # MODIFIED LINE

# Calculate RMSE
rmse_val_ensemble_ablation2 = calculate_rmse(y_val_ablation2, y_pred_ensemble_ablation2)
print(f"Ablation 2 Performance (RMSE): {rmse_val_ensemble_ablation2:.4f}")
ablation_2_score = rmse_val_ensemble_ablation2

# ==============================================================================
# ===== SUMMARY =====
# ==============================================================================
print("\n===== ABLATION STUDY SUMMARY =====")
ablations = [
    ("Baseline (Original Code)", baseline_score),
    ("Ablation 1 (Impute with Mean)", ablation_1_score),
    ("Ablation 2 (LightGBM Only)", ablation_2_score),
]

print("\n--- Performance Results (RMSE) ---")
for name, score in ablations:
    print(f"{name}: {score:.4f}")

deltas = []
print("\n--- Impact Relative to Baseline ---")
for name, score in ablations[1:]: # Exclude baseline from delta calculation
    delta = abs(score - baseline_score)
    deltas.append((name, delta))
    print(f"{name}: Delta from Baseline = {delta:.4f}")

if deltas:
    most_impactful = max(deltas, key=lambda x: x[1])
    print(f"\nConclusion: The component whose ablation resulted in the largest change in performance was '{most_impactful[0]}' (delta: {most_impactful[1]:.4f}).")
else:
    print("No ablations were performed to compare.")