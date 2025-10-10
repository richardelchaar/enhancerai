
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

# Imports
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ===== Data Loading =====
# As per instructions, assuming the file exists at the specified path and not using try/except.
train_df = pd.read_csv("./input/train.csv")

# Separate features and target variable
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# ===== BASELINE: Original Code =====
print("Running Baseline (Original Code)...")

# Handle missing values: Impute 'total_bedrooms' with its median
# This is a common preprocessing step for this dataset
if X['total_bedrooms'].isnull().any():
    median_total_bedrooms_baseline = X['total_bedrooms'].median()
    X_baseline = X.copy()  # Use a copy for baseline to avoid modifying X for subsequent ablations
    X_baseline['total_bedrooms'].fillna(median_total_bedrooms_baseline, inplace=True)
else:
    X_baseline = X.copy()

# Split the data into training and validation sets
# Using a fixed random_state for reproducibility
X_train_baseline, X_val_baseline, y_train_baseline, y_val_baseline = train_test_split(X_baseline, y, test_size=0.2, random_state=42)

# --- Model Training ---

# 1. Initialize and Train LightGBM Regressor model (from base solution)
lgbm_model_baseline = lgb.LGBMRegressor(objective='regression_l2', metric='rmse', random_state=42, verbose=-1)
lgbm_model_baseline.fit(X_train_baseline, y_train_baseline)

# Make predictions on the validation set using LightGBM
y_pred_lgbm_baseline = lgbm_model_baseline.predict(X_val_baseline)

# 2. Initialize and Train XGBoost Regressor model (from reference solution)
xgb_model_baseline = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, verbosity=0)
xgb_model_baseline.fit(X_train_baseline, y_train_baseline)

# Make predictions on the validation set using XGBoost
y_pred_xgb_baseline = xgb_model_baseline.predict(X_val_baseline)

# --- Ensemble Predictions ---
# Simple averaging ensemble of LightGBM and XGBoost predictions
y_pred_ensemble_baseline = (y_pred_lgbm_baseline + y_pred_xgb_baseline) / 2

# --- Evaluate the Ensembled Model ---
# Evaluate the model using Root Mean Squared Error (RMSE)
rmse_ensemble_baseline = np.sqrt(mean_squared_error(y_val_baseline, y_pred_ensemble_baseline))

baseline_score = rmse_ensemble_baseline
print(f"Baseline Performance (RMSE): {baseline_score:.4f}")

# ===== ABLATION 1: Remove Ensemble (Use only LightGBM) =====
print("\nRunning Ablation 1: Remove Ensemble (Use only LightGBM)...")

# Handle missing values: Impute 'total_bedrooms' with its median (same as baseline)
if X['total_bedrooms'].isnull().any():
    median_total_bedrooms_ablation1 = X['total_bedrooms'].median()
    X_ablation1 = X.copy()
    X_ablation1['total_bedrooms'].fillna(median_total_bedrooms_ablation1, inplace=True)
else:
    X_ablation1 = X.copy()

# Split the data into training and validation sets
X_train_ablation1, X_val_ablation1, y_train_ablation1, y_val_ablation1 = train_test_split(X_ablation1, y, test_size=0.2, random_state=42)

# --- Model Training ---

# 1. Initialize and Train LightGBM Regressor model
lgbm_model_ablation1 = lgb.LGBMRegressor(objective='regression_l2', metric='rmse', random_state=42, verbose=-1)
lgbm_model_ablation1.fit(X_train_ablation1, y_train_ablation1)

# Make predictions on the validation set using LightGBM
y_pred_lgbm_ablation1 = lgbm_model_ablation1.predict(X_val_ablation1)

# --- Evaluate the LightGBM Model (no ensemble) ---
rmse_lgbm_ablation1 = np.sqrt(mean_squared_error(y_val_ablation1, y_pred_lgbm_ablation1))

ablation_1_score = rmse_lgbm_ablation1
print(f"Ablation 1 Performance (RMSE - LightGBM only): {ablation_1_score:.4f}")

# ===== ABLATION 2: Remove Missing Value Imputation for 'total_bedrooms' =====
print("\nRunning Ablation 2: Remove Missing Value Imputation for 'total_bedrooms'...")

# DO NOT impute 'total_bedrooms' - let models handle NaNs if they can
X_ablation2 = X.copy()  # Use original X with NaNs in total_bedrooms

# Split the data into training and validation sets
X_train_ablation2, X_val_ablation2, y_train_ablation2, y_val_ablation2 = train_test_split(X_ablation2, y, test_size=0.2, random_state=42)

# --- Model Training ---

# 1. Initialize and Train LightGBM Regressor model
# LightGBM can handle NaNs by default
lgbm_model_ablation2 = lgb.LGBMRegressor(objective='regression_l2', metric='rmse', random_state=42, verbose=-1)
lgbm_model_ablation2.fit(X_train_ablation2, y_train_ablation2)

# Make predictions on the validation set using LightGBM
y_pred_lgbm_ablation2 = lgbm_model_ablation2.predict(X_val_ablation2)

# 2. Initialize and Train XGBoost Regressor model
# XGBoost can handle NaNs by default
xgb_model_ablation2 = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, verbosity=0)
xgb_model_ablation2.fit(X_train_ablation2, y_train_ablation2)

# Make predictions on the validation set using XGBoost
y_pred_xgb_ablation2 = xgb_model_ablation2.predict(X_val_ablation2)

# --- Ensemble Predictions ---
y_pred_ensemble_ablation2 = (y_pred_lgbm_ablation2 + y_pred_xgb_ablation2) / 2

# --- Evaluate the Ensembled Model ---
rmse_ensemble_ablation2 = np.sqrt(mean_squared_error(y_val_ablation2, y_pred_ensemble_ablation2))

ablation_2_score = rmse_ensemble_ablation2
print(f"Ablation 2 Performance (RMSE - No total_bedrooms imputation): {ablation_2_score:.4f}")

# ===== SUMMARY =====
print("\n===== ABLATION STUDY SUMMARY =====")
ablations = [
    ("Baseline (Original Code)", baseline_score),
    ("Ablation 1 (LightGBM Only)", ablation_1_score),
    ("Ablation 2 (No total_bedrooms Imputation)", ablation_2_score),
]

print(f"Baseline Performance (RMSE): {baseline_score:.4f}")
print(f"Ablation 1 Performance (RMSE - LightGBM Only): {ablation_1_score:.4f}")
print(f"Ablation 2 Performance (RMSE - No total_bedrooms Imputation): {ablation_2_score:.4f}")

deltas = []
for name, score in ablations[1:]:  # Start from index 1 to skip baseline
    delta = abs(score - baseline_score)
    deltas.append((name, delta))

if deltas:
    most_impactful = max(deltas, key=lambda x: x[1])
    print(f"\nMost impactful component (largest change from baseline): {most_impactful[0]} (delta: {most_impactful[1]:.4f})")
else:
    print("\nNo ablations were performed to compare against the baseline.")
