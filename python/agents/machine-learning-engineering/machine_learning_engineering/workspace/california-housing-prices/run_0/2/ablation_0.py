
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
import os

# --- Dummy Data Generation for Reproducibility ---
# This section ensures the script can run even if the 'input' directory and files are not present.
# In a real scenario, you would simply ensure your 'input' folder exists with train.csv and test.csv.
if not os.path.exists("./input"):
    os.makedirs("./input")

if not os.path.exists("./input/train.csv"):
    print("Generating dummy train.csv...")
    np.random.seed(42)
    dummy_data = {
        'longitude': np.random.uniform(-125, -114, 1000),
        'latitude': np.random.uniform(32, 42, 1000),
        'housing_median_age': np.random.randint(1, 52, 1000),
        'total_rooms': np.random.randint(100, 6000, 1000),
        'total_bedrooms': np.random.randint(50, 1200, 1000),
        'population': np.random.randint(100, 3000, 1000),
        'households': np.random.randint(50, 1100, 1000),
        'median_income': np.random.uniform(0.5, 10, 1000),
        'median_house_value': np.random.rand(1000) * 500000
    }
    dummy_df = pd.DataFrame(dummy_data)
    # Introduce some missing values for total_bedrooms
    missing_indices = np.random.choice(dummy_df.index, size=50, replace=False)
    dummy_df.loc[missing_indices, 'total_bedrooms'] = np.nan
    dummy_df.to_csv("./input/train.csv", index=False)

if not os.path.exists("./input/test.csv"):
    print("Generating dummy test.csv...")
    np.random.seed(43)
    dummy_data_test = {
        'longitude': np.random.uniform(-125, -114, 200),
        'latitude': np.random.uniform(32, 42, 200),
        'housing_median_age': np.random.randint(1, 52, 200),
        'total_rooms': np.random.randint(100, 6000, 200),
        'total_bedrooms': np.random.randint(50, 1200, 200),
        'population': np.random.randint(100, 3000, 200),
        'households': np.random.randint(50, 1100, 200),
        'median_income': np.random.uniform(0.5, 10, 200),
    }
    dummy_df_test = pd.DataFrame(dummy_data_test)
    # Introduce some missing values for total_bedrooms
    missing_indices_test = np.random.choice(dummy_df_test.index, size=10, replace=False)
    dummy_df_test.loc[missing_indices_test, 'total_bedrooms'] = np.nan
    dummy_df_test.to_csv("./input/test.csv", index=False)

print("Data setup complete. Running ablation study...")
# --- End of Dummy Data Generation ---

# ===== BASELINE: Original Code =====
print("\n===== Running Baseline: Original Code =====")

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# Identify features and target
TARGET = 'median_house_value'
features = [col for col in train_df.columns if col != TARGET]

# Separate features (X) and target (y)
X = train_df[features]
y = train_df[TARGET]

# --- Preprocessing (integrated from both solutions, ensuring consistency) ---

# Handle missing values: Impute 'total_bedrooms' with the median
# The base solution specifically handles 'total_bedrooms'.
# The reference solution handles all missing numerical features with their median.
# For this dataset, 'total_bedrooms' is the primary one, so we'll use the base's specific handling.
# Calculate median from the full training features before splitting for consistency.
median_total_bedrooms_baseline = None
if 'total_bedrooms' in X.columns:
    if X['total_bedrooms'].isnull().any():
        median_total_bedrooms_baseline = X['total_bedrooms'].median()
        X['total_bedrooms'] = X['total_bedrooms'].fillna(median_total_bedrooms_baseline)

# Split the data into training and validation sets
# Using a fixed random_state for reproducibility
X_train_baseline, X_val_baseline, y_train_baseline, y_val_baseline = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---

# 1. LightGBM Model (from base solution)
# print("Training LightGBM model...") # Suppressing internal prints for cleaner ablation output
lgbm_params_baseline = {
    'objective': 'regression_l2',
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'random_state': 42,
    'n_jobs': -1,  # Use all available cores
    'verbose': -1,  # Suppress verbose output
    'boosting_type': 'gbdt',
}
lgbm_model_baseline = lgb.LGBMRegressor(**lgbm_params_baseline)
lgbm_model_baseline.fit(X_train_baseline, y_train_baseline)
# print("LightGBM model training complete.")

# 2. XGBoost Model (from reference solution)
# print("Training XGBoost model...")
xgb_model_baseline = xgb.XGBRegressor(
    objective='reg:squarederror',  # Objective for regression tasks
    eval_metric='rmse',            # Evaluation metric is Root Mean Squared Error
    n_estimators=1000,             # Number of boosting rounds
    learning_rate=0.05,            # Step size shrinkage to prevent overfitting
    max_depth=6,                   # Maximum depth of a tree
    random_state=42,               # For reproducibility
    n_jobs=-1,                     # Use all available CPU cores - FIXED: Removed quotes
    verbosity=0                    # Suppress verbose output
)
xgb_model_baseline.fit(X_train_baseline, y_train_baseline)
# print("XGBoost model training complete.")

# --- Prediction and Ensemble ---

# Make predictions on the validation set with LightGBM
y_pred_val_lgbm_baseline = lgbm_model_baseline.predict(X_val_baseline)

# Make predictions on the validation set with XGBoost
y_pred_val_xgb_baseline = xgb_model_baseline.predict(X_val_baseline)

# Ensemble the predictions by averaging (simple average ensemble)
y_pred_val_ensemble_baseline = (y_pred_val_lgbm_baseline + y_pred_val_xgb_baseline) / 2

# --- Evaluation ---

# Evaluate the ensembled model using Root Mean Squared Error (RMSE)
rmse_baseline = np.sqrt(mean_squared_error(y_val_baseline, y_pred_val_ensemble_baseline))

# Print the final validation performance
print(f'Baseline Performance (RMSE): {rmse_baseline:.4f}')

# The rest of the original code for test data prediction is not strictly needed for ablation study performance comparison
# as we are focusing on validation set performance, but including for completeness if it were critical.
# For this study, we only need the validation RMSE.

# ===== ABLATION 1: Impute 'total_bedrooms' with 0 instead of Median =====
print("\n===== Running Ablation 1: Impute 'total_bedrooms' with 0 =====")

# Load the training data
train_df_ablation1 = pd.read_csv("./input/train.csv")

# Identify features and target
TARGET = 'median_house_value'
features = [col for col in train_df_ablation1.columns if col != TARGET]

# Separate features (X) and target (y)
X_ablation1 = train_df_ablation1[features].copy()  # Use .copy() to avoid SettingWithCopyWarning
y_ablation1 = train_df_ablation1[TARGET].copy()

# --- Preprocessing (modified for ablation) ---

# Handle missing values: Impute 'total_bedrooms' with 0
median_total_bedrooms_ablation1 = None  # Not used in this ablation, but kept for structural consistency
if 'total_bedrooms' in X_ablation1.columns:
    if X_ablation1['total_bedrooms'].isnull().any():
        # ABLATION CHANGE: Fill with 0 instead of median
        X_ablation1['total_bedrooms'] = X_ablation1['total_bedrooms'].fillna(0)

# Split the data into training and validation sets
X_train_ablation1, X_val_ablation1, y_train_ablation1, y_val_ablation1 = train_test_split(X_ablation1, y_ablation1, test_size=0.2, random_state=42)

# --- Model Training ---

# 1. LightGBM Model
lgbm_params_ablation1 = {
    'objective': 'regression_l2', 'metric': 'rmse', 'n_estimators': 1000,
    'learning_rate': 0.05, 'num_leaves': 31, 'random_state': 42, 'n_jobs': -1, 'verbose': -1, 'boosting_type': 'gbdt',
}
lgbm_model_ablation1 = lgb.LGBMRegressor(**lgbm_params_ablation1)
lgbm_model_ablation1.fit(X_train_ablation1, y_train_ablation1)

# 2. XGBoost Model
xgb_model_ablation1 = xgb.XGBRegressor(
    objective='reg:squarederror', eval_metric='rmse', n_estimators=1000,
    learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, verbosity=0  # Fixed: Removed quotes
)
xgb_model_ablation1.fit(X_train_ablation1, y_train_ablation1)

# --- Prediction and Ensemble ---
y_pred_val_lgbm_ablation1 = lgbm_model_ablation1.predict(X_val_ablation1)
y_pred_val_xgb_ablation1 = xgb_model_ablation1.predict(X_val_ablation1)
y_pred_val_ensemble_ablation1 = (y_pred_val_lgbm_ablation1 + y_pred_val_xgb_ablation1) / 2

# --- Evaluation ---
rmse_ablation1 = np.sqrt(mean_squared_error(y_val_ablation1, y_pred_val_ensemble_ablation1))
print(f'Ablation 1 Performance (RMSE): {rmse_ablation1:.4f}')

# ===== ABLATION 2: Remove XGBoost from Ensemble (use only LightGBM) =====
print("\n===== Running Ablation 2: Remove XGBoost from Ensemble =====")

# Load the training data
train_df_ablation2 = pd.read_csv("./input/train.csv")

# Identify features and target
TARGET = 'median_house_value'
features = [col for col in train_df_ablation2.columns if col != TARGET]

# Separate features (X) and target (y)
X_ablation2 = train_df_ablation2[features].copy()
y_ablation2 = train_df_ablation2[TARGET].copy()

# --- Preprocessing (same as baseline) ---

# Handle missing values: Impute 'total_bedrooms' with the median
median_total_bedrooms_ablation2 = None
if 'total_bedrooms' in X_ablation2.columns:
    if X_ablation2['total_bedrooms'].isnull().any():
        median_total_bedrooms_ablation2 = X_ablation2['total_bedrooms'].median()
        X_ablation2['total_bedrooms'] = X_ablation2['total_bedrooms'].fillna(median_total_bedrooms_ablation2)

# Split the data into training and validation sets
X_train_ablation2, X_val_ablation2, y_train_ablation2, y_val_ablation2 = train_test_split(X_ablation2, y_ablation2, test_size=0.2, random_state=42)

# --- Model Training ---

# 1. LightGBM Model (same as baseline)
lgbm_params_ablation2 = {
    'objective': 'regression_l2', 'metric': 'rmse', 'n_estimators': 1000,
    'learning_rate': 0.05, 'num_leaves': 31, 'random_state': 42, 'n_jobs': -1, 'verbose': -1, 'boosting_type': 'gbdt',
}
lgbm_model_ablation2 = lgb.LGBMRegressor(**lgbm_params_ablation2)
lgbm_model_ablation2.fit(X_train_ablation2, y_train_ablation2)

# 2. XGBoost Model (still trained, but its predictions won't be used in the ensemble)
xgb_model_ablation2 = xgb.XGBRegressor(
    objective='reg:squarederror', eval_metric='rmse', n_estimators=1000,
    learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, verbosity=0  # Fixed: Removed quotes
)
xgb_model_ablation2.fit(X_train_ablation2, y_train_ablation2)

# --- Prediction and Ensemble (modified for ablation) ---

# Make predictions on the validation set with LightGBM
y_pred_val_lgbm_ablation2 = lgbm_model_ablation2.predict(X_val_ablation2)

# Make predictions on the validation set with XGBoost (will not be used for final ensemble here)
# y_pred_val_xgb_ablation2 = xgb_model_ablation2.predict(X_val_ablation2) # This line is functionally commented out by not being used

# ABLATION CHANGE: Ensemble using only LightGBM predictions
y_pred_val_ensemble_ablation2 = y_pred_val_lgbm_ablation2

# --- Evaluation ---
rmse_ablation2 = np.sqrt(mean_squared_error(y_val_ablation2, y_pred_val_ensemble_ablation2))
print(f'Ablation 2 Performance (RMSE): {rmse_ablation2:.4f}')

# ===== SUMMARY =====
print("\n===== ABLATION STUDY SUMMARY =====")
ablations = [
    ("Baseline", rmse_baseline),
    ("Ablation 1: 'total_bedrooms' imputed with 0", rmse_ablation1),
    ("Ablation 2: Only LightGBM (No XGBoost Ensemble)", rmse_ablation2),
]

print("\n--- Performance Results ---")
for name, score in ablations:
    print(f"{name}: RMSE = {score:.4f}")

deltas = []
for name, score in ablations[1:]:  # Start from the first ablation, skip baseline
    delta = abs(score - rmse_baseline)
    deltas.append((name, delta))
    print(f"Change from Baseline for '{name}': {delta:.4f}")

if deltas:
    most_impactful = max(deltas, key=lambda x: x[1])
    print(f"\nConclusion: The component whose ablation resulted in the largest change in performance is: '{most_impactful[0]}'")
    print(f"This change had a delta RMSE of: {most_impactful[1]:.4f}")
else:
    print("\nNo ablations performed to compare against baseline.")

# Final Validation Performance output as requested
final_validation_score = rmse_baseline  # The baseline performance is typically the one to report if not specified otherwise
print(f'Final Validation Performance: {final_validation_score}')
