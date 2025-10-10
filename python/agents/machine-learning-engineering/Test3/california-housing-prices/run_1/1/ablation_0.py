
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

# Identify features and target
TARGET = 'median_house_value'
features = [col for col in train_df.columns if col != TARGET]

# Separate features (X) and target (y)
X = train_df[features]
y = train_df[TARGET]

# --- Preprocessing (integrated from both solutions, ensuring consistency) ---

# Handle missing values: Impute 'total_bedrooms' with the median
median_total_bedrooms = None
if 'total_bedrooms' in X.columns:
    if X['total_bedrooms'].isnull().any():
        median_total_bedrooms = X['total_bedrooms'].median()
        X['total_bedrooms'] = X['total_bedrooms'].fillna(median_total_bedrooms)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---

# 1. LightGBM Model (from base solution)
print("Training LightGBM model (Baseline)...")
lgbm_params = {
    'objective': 'regression_l2',
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    'boosting_type': 'gbdt',
}
lgbm_model = lgb.LGBMRegressor(**lgbm_params)
lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
print("LightGBM model training complete (Baseline).")

# 2. XGBoost Model (from reference solution)
print("Training XGBoost model (Baseline)...")
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb_model.fit(X_train, y_train)
print("XGBoost model training complete (Baseline).")

# --- Prediction and Ensemble ---

# Make predictions on the validation set with LightGBM
y_pred_val_lgbm = lgbm_model.predict(X_val)

# Make predictions on the validation set with XGBoost
y_pred_val_xgb = xgb_model.predict(X_val)

# Ensemble the predictions by averaging (simple average ensemble)
y_pred_val_ensemble = (y_pred_val_lgbm + y_pred_val_xgb) / 2

# --- Evaluation ---

# Evaluate the ensembled model using Root Mean Squared Error (RMSE)
baseline_score = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble))

print(f"Baseline Performance: {baseline_score:.4f}")

# ===== ABLATION 1: Remove Ensembling (use only LightGBM) =====
print("\nRunning Ablation 1: Remove Ensembling (use only LightGBM)...")

# Identify features and target (same as baseline)
TARGET = 'median_house_value'
features_ablation1 = [col for col in train_df.columns if col != TARGET]

# Separate features (X) and target (y)
X_ablation1 = train_df[features_ablation1]
y_ablation1 = train_df[TARGET]

# --- Preprocessing (same as baseline) ---
median_total_bedrooms_ablation1 = None
if 'total_bedrooms' in X_ablation1.columns:
    if X_ablation1['total_bedrooms'].isnull().any():
        median_total_bedrooms_ablation1 = X_ablation1['total_bedrooms'].median()
        X_ablation1['total_bedrooms'] = X_ablation1['total_bedrooms'].fillna(median_total_bedrooms_ablation1)

# Split the data into training and validation sets (same as baseline)
X_train_ablation1, X_val_ablation1, y_train_ablation1, y_val_ablation1 = train_test_split(X_ablation1, y_ablation1, test_size=0.2, random_state=42)

# --- Model Training ---

# 1. LightGBM Model (same as baseline)
print("Training LightGBM model (Ablation 1)...")
lgbm_params_ablation1 = {
    'objective': 'regression_l2',
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    'boosting_type': 'gbdt',
}
lgbm_model_ablation1 = lgb.LGBMRegressor(**lgbm_params_ablation1)
lgbm_model_ablation1.fit(X_train_ablation1, y_train_ablation1, eval_set=[(X_val_ablation1, y_val_ablation1)], callbacks=[lgb.early_stopping(100, verbose=False)])
print("LightGBM model training complete (Ablation 1).")

# 2. XGBoost Model (still trained, but its predictions won't be used for ensemble)
print("Training XGBoost model (Ablation 1 - will not be ensembled)...")
xgb_model_ablation1 = xgb.XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb_model_ablation1.fit(X_train_ablation1, y_train_ablation1)
print("XGBoost model training complete (Ablation 1).")

# --- Prediction and Ensemble (MODIFIED for ablation) ---

# Make predictions on the validation set with LightGBM
y_pred_val_lgbm_ablation1 = lgbm_model_ablation1.predict(X_val_ablation1)

# Make predictions on the validation set with XGBoost (predictions made but not used for ensemble)
# y_pred_val_xgb_ablation1 = xgb_model_ablation1.predict(X_val_ablation1)

# Ensemble the predictions by averaging (MODIFIED: use only LightGBM)
y_pred_val_ensemble_ablation1 = y_pred_val_lgbm_ablation1

# --- Evaluation ---
ablation_1_score = np.sqrt(mean_squared_error(y_val_ablation1, y_pred_val_ensemble_ablation1))
print(f"Ablation 1 Performance: {ablation_1_score:.4f}")

# ===== ABLATION 2: Remove 'total_bedrooms' feature =====
print("\nRunning Ablation 2: Remove 'total_bedrooms' feature...")

# Identify features and target (MODIFIED for ablation)
TARGET = 'median_house_value'
features_ablation2 = [col for col in train_df.columns if col != TARGET and col != 'total_bedrooms']

# Separate features (X) and target (y)
X_ablation2 = train_df[features_ablation2]
y_ablation2 = train_df[TARGET]

# --- Preprocessing (MODIFIED for ablation: no 'total_bedrooms' to impute) ---
# The imputation block for 'total_bedrooms' is effectively removed as the feature is not present.
# median_total_bedrooms_ablation2 = None
# if 'total_bedrooms' in X_ablation2.columns: # This condition will now be false
#     if X_ablation2['total_bedrooms'].isnull().any():
#         median_total_bedrooms_ablation2 = X_ablation2['total_bedrooms'].median()
#         X_ablation2['total_bedrooms'] = X_ablation2['total_bedrooms'].fillna(median_total_bedrooms_ablation2)

# Split the data into training and validation sets (same as baseline)
X_train_ablation2, X_val_ablation2, y_train_ablation2, y_val_ablation2 = train_test_split(X_ablation2, y_ablation2, test_size=0.2, random_state=42)

# --- Model Training ---

# 1. LightGBM Model (same as baseline, but with modified features)
print("Training LightGBM model (Ablation 2)...")
lgbm_params_ablation2 = {
    'objective': 'regression_l2',
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    'boosting_type': 'gbdt',
}
lgbm_model_ablation2 = lgb.LGBMRegressor(**lgbm_params_ablation2)
lgbm_model_ablation2.fit(X_train_ablation2, y_train_ablation2, eval_set=[(X_val_ablation2, y_val_ablation2)], callbacks=[lgb.early_stopping(100, verbose=False)])
print("LightGBM model training complete (Ablation 2).")

# 2. XGBoost Model (same as baseline, but with modified features)
print("Training XGBoost model (Ablation 2)...")
xgb_model_ablation2 = xgb.XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb_model_ablation2.fit(X_train_ablation2, y_train_ablation2)
print("XGBoost model training complete (Ablation 2).")

# --- Prediction and Ensemble (same as baseline, but with modified features) ---

# Make predictions on the validation set with LightGBM
y_pred_val_lgbm_ablation2 = lgbm_model_ablation2.predict(X_val_ablation2)

# Make predictions on the validation set with XGBoost
y_pred_val_xgb_ablation2 = xgb_model_ablation2.predict(X_val_ablation2)

# Ensemble the predictions by averaging (simple average ensemble)
y_pred_val_ensemble_ablation2 = (y_pred_val_lgbm_ablation2 + y_pred_val_xgb_ablation2) / 2

# --- Evaluation ---
ablation_2_score = np.sqrt(mean_squared_error(y_val_ablation2, y_pred_val_ensemble_ablation2))
print(f"Ablation 2 Performance: {ablation_2_score:.4f}")

# ===== SUMMARY =====
print("\n===== ABLATION STUDY SUMMARY =====")
ablations = [
    ("Baseline", baseline_score),
    ("Ablation 1: Remove Ensembling (LightGBM only)", ablation_1_score),
    ("Ablation 2: Remove 'total_bedrooms' feature", ablation_2_score),
]

print(f"Baseline Performance: {baseline_score:.4f}")
print(f"Ablation 1 Performance (Remove Ensembling): {ablation_1_score:.4f}")
print(f"Ablation 2 Performance (Remove 'total_bedrooms'): {ablation_2_score:.4f}")

deltas = []
# Calculate absolute change from baseline for each ablation
for name, score in ablations[1:]:  # Skip baseline for delta calculation
    delta = abs(score - baseline_score)
    deltas.append((name, delta))

if deltas:
    most_impactful = max(deltas, key=lambda x: x[1])
    print(f"\nMost impactful component ablation: '{most_impactful[0]}' (Absolute change from baseline: {most_impactful[1]:.4f})")
else:
    print("\nNo ablations were performed to compare against the baseline.")
