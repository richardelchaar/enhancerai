
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
# Assume files exist. DO NOT add dummy data or file checks.
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')

# ===== BASELINE: Original Code =====
print("Running Baseline...")

# Separate target variable from features
X_baseline = train_df.drop("median_house_value", axis=1).copy()
y_baseline = train_df["median_house_value"].copy()
test_df_baseline = test_df.copy()  # Keep a copy for consistent imputation

# --- Preprocessing ---
# Handle missing values
# Impute missing values with the median of the column from the training data.
# This ensures consistency between training and test set imputation.
for col in X_baseline.columns:
    if X_baseline[col].isnull().any():
        median_val = X_baseline[col].median()  # Calculate median from training data
        X_baseline[col].fillna(median_val, inplace=True)
        # Apply the same imputation to the test set using the training set's median
        if col in test_df_baseline.columns:
            test_df_baseline[col].fillna(median_val, inplace=True)

# Split the training data into training and validation sets
X_train_baseline, X_val_baseline, y_train_baseline, y_val_baseline = train_test_split(X_baseline, y_baseline, test_size=0.2, random_state=42)

# --- Model Training and Prediction ---

# 1. LightGBM Model (from base solution)
# Initialize LightGBM Regressor model
lgbm_model_baseline = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1)
# Train the LightGBM model
lgbm_model_baseline.fit(X_train_baseline, y_train_baseline)
# Make predictions on the validation set with LightGBM
y_pred_lgbm_baseline = lgbm_model_baseline.predict(X_val_baseline)

# 2. XGBoost Model (from reference solution)
# Initialize XGBoost Regressor model
xgb_model_baseline = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, verbosity=0)
# Train the XGBoost model
xgb_model_baseline.fit(X_train_baseline, y_train_baseline)
# Make predictions on the validation set with XGBoost
y_pred_xgb_baseline = xgb_model_baseline.predict(X_val_baseline)

# --- Ensembling ---
# Simple average ensemble of the two models' predictions
y_pred_ensemble_baseline = (y_pred_lgbm_baseline + y_pred_xgb_baseline) / 2

# --- Evaluation ---
# Calculate RMSE on the validation set for the ensembled predictions
rmse_baseline = np.sqrt(mean_squared_error(y_val_baseline, y_pred_ensemble_baseline))

print(f"Baseline Performance (RMSE): {rmse_baseline:.4f}")

# ===== ABLATION 1: Remove Missing Value Imputation =====
print("\nRunning Ablation 1: Remove Missing Value Imputation...")

# Separate target variable from features
X_ablation1 = train_df.drop("median_house_value", axis=1).copy()
y_ablation1 = train_df["median_house_value"].copy()
test_df_ablation1 = test_df.copy()  # Keep a copy for consistency

# --- Preprocessing ---
# Handle missing values - THIS SECTION IS SKIPPED FOR ABLATION 1
# Models like LightGBM and XGBoost can often handle NaNs directly,
# but skipping imputation might still impact performance.
# for col in X_ablation1.columns:
#     if X_ablation1[col].isnull().any():
#         median_val = X_ablation1[col].median()
#         X_ablation1[col].fillna(median_val, inplace=True)
#         if col in test_df_ablation1.columns:
#             test_df_ablation1[col].fillna(median_val, inplace=True)

# Split the training data into training and validation sets
X_train_ablation1, X_val_ablation1, y_train_ablation1, y_val_ablation1 = train_test_split(X_ablation1, y_ablation1, test_size=0.2, random_state=42)

# --- Model Training and Prediction ---

# 1. LightGBM Model
lgbm_model_ablation1 = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1)
lgbm_model_ablation1.fit(X_train_ablation1, y_train_ablation1)
y_pred_lgbm_ablation1 = lgbm_model_ablation1.predict(X_val_ablation1)

# 2. XGBoost Model
xgb_model_ablation1 = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, verbosity=0)
xgb_model_ablation1.fit(X_train_ablation1, y_train_ablation1)
y_pred_xgb_ablation1 = xgb_model_ablation1.predict(X_val_ablation1)

# --- Ensembling ---
y_pred_ensemble_ablation1 = (y_pred_lgbm_ablation1 + y_pred_xgb_ablation1) / 2

# --- Evaluation ---
rmse_ablation1 = np.sqrt(mean_squared_error(y_val_ablation1, y_pred_ensemble_ablation1))
print(f"Ablation 1 Performance (RMSE - No Imputation): {rmse_ablation1:.4f}")

# ===== ABLATION 2: Use only LightGBM in Ensemble =====
print("\nRunning Ablation 2: Use only LightGBM in Ensemble...")

# Separate target variable from features
X_ablation2 = train_df.drop("median_house_value", axis=1).copy()
y_ablation2 = train_df["median_house_value"].copy()
test_df_ablation2 = test_df.copy()  # Keep a copy for consistent imputation

# --- Preprocessing ---
# Handle missing values (same as baseline)
for col in X_ablation2.columns:
    if X_ablation2[col].isnull().any():
        median_val = X_ablation2[col].median()
        X_ablation2[col].fillna(median_val, inplace=True)
        if col in test_df_ablation2.columns:
            test_df_ablation2[col].fillna(median_val, inplace=True)

# Split the training data into training and validation sets
X_train_ablation2, X_val_ablation2, y_train_ablation2, y_val_ablation2 = train_test_split(X_ablation2, y_ablation2, test_size=0.2, random_state=42)

# --- Model Training and Prediction ---

# 1. LightGBM Model
lgbm_model_ablation2 = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1)
lgbm_model_ablation2.fit(X_train_ablation2, y_train_ablation2)
y_pred_lgbm_ablation2 = lgbm_model_ablation2.predict(X_val_ablation2)

# 2. XGBoost Model (still trained, but its predictions won't be used in ensemble)
xgb_model_ablation2 = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, verbosity=0)
xgb_model_ablation2.fit(X_train_ablation2, y_train_ablation2)
y_pred_xgb_ablation2 = xgb_model_ablation2.predict(X_val_ablation2)  # Predictions still made

# --- Ensembling ---
# Only use LightGBM predictions for the ensemble
y_pred_ensemble_ablation2 = y_pred_lgbm_ablation2  # Ablation: Removed XGBoost from ensemble

# --- Evaluation ---
rmse_ablation2 = np.sqrt(mean_squared_error(y_val_ablation2, y_pred_ensemble_ablation2))
print(f"Ablation 2 Performance (RMSE - Only LGBM in Ensemble): {rmse_ablation2:.4f}")

# ===== SUMMARY =====
print("\n===== ABLATION STUDY SUMMARY =====")

print(f"Baseline Performance (RMSE): {rmse_baseline:.4f}")
print(f"Ablation 1 (No Missing Value Imputation) Performance (RMSE): {rmse_ablation1:.4f}")
print(f"Ablation 2 (Only LGBM in Ensemble) Performance (RMSE): {rmse_ablation2:.4f}")

ablations = [
    ("Ablation 1 (No Missing Value Imputation)", rmse_ablation1),
    ("Ablation 2 (Only LGBM in Ensemble)", rmse_ablation2),
]

deltas = [(name, abs(score - rmse_baseline)) for name, score in ablations]

if deltas:
    most_impactful = max(deltas, key=lambda x: x[1])
    print(f"\nMost impactful component: {most_impactful[0]} (Absolute change from baseline: {most_impactful[1]:.4f})")
else:
    print("\nNo ablations were performed or recorded.")
