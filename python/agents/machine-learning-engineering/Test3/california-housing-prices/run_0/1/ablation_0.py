
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
from sklearn.impute import SimpleImputer
import os

# --- Mock Data Generation (CRITICAL for runnable script) ---
# This part ensures the script can run without pre-existing train.csv or test.csv
if not os.path.exists('./input'):
    os.makedirs('./input')

# Generate dummy train.csv if it doesn't exist
if not os.path.exists('./input/train.csv'):
    print("Creating dummy './input/train.csv' for demonstration purposes...")
    # Generate synthetic data resembling the dataset description
    np.random.seed(42)
    n_samples = 1000
    data = {
        'longitude': np.random.uniform(-125, -114, n_samples),
        'latitude': np.random.uniform(32, 42, n_samples),
        'housing_median_age': np.random.randint(1, 52, n_samples),
        'total_rooms': np.random.randint(100, 10000, n_samples),
        'total_bedrooms': np.random.randint(50, 2000, n_samples),
        'population': np.random.randint(100, 5000, n_samples),
        'households': np.random.randint(50, 1500, n_samples),
        'median_income': np.random.uniform(0.5, 15, n_samples),
        'median_house_value': np.random.rand(n_samples) * 500000 + 50000  # Target variable
    }
    dummy_df = pd.DataFrame(data)

# Introduce some missing values for imputation test in 'total_bedrooms' and 'population'
    for col in ['total_bedrooms', 'population']:
        missing_indices = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
        dummy_df.loc[missing_indices, col] = np.nan

dummy_df.to_csv('./input/train.csv', index=False)
print("Dummy 'train.csv' created.")
else:  # Corrected indentation for this 'else' block
    print("Using existing './input/train.csv'.")

# Generate dummy test.csv if it doesn't exist
if not os.path.exists('./input/test.csv'):
    print("Creating dummy './input/test.csv' for demonstration purposes...")
    np.random.seed(43)  # Different seed for test data
    n_samples_test = 200  # Example number of test samples
    data_test = {
        'longitude': np.random.uniform(-125, -114, n_samples_test),
        'latitude': np.random.uniform(32, 42, n_samples_test),
        'housing_median_age': np.random.randint(1, 52, n_samples_test),
        'total_rooms': np.random.randint(100, 10000, n_samples_test),
        'total_bedrooms': np.random.randint(50, 2000, n_samples_test),
        'population': np.random.randint(100, 5000, n_samples_test),
        'households': np.random.randint(50, 1500, n_samples_test),
        'median_income': np.random.uniform(0.5, 15, n_samples_test),
    }
    dummy_test_df = pd.DataFrame(data_test)

# Introduce some missing values for imputation test in 'total_bedrooms' and 'population'
    for col in ['total_bedrooms', 'population']:
        missing_indices = np.random.choice(n_samples_test, int(n_samples_test * 0.05), replace=False)
        dummy_test_df.loc[missing_indices, col] = np.nan

dummy_test_df.to_csv('./input/test.csv', index=False)
print("Dummy 'test.csv' created.")
else:
    print("Using existing './input/test.csv'.")
# --- End Mock Data Generation ---

# ===== BASELINE: Original Code =====
print("\nRunning Baseline: Original Code (Ensembled, Median Imputation)")
# --- 1. Data Loading and Preprocessing ---
train_df_baseline = pd.read_csv("./input/train.csv")

# Separate features and target
X_baseline = train_df_baseline.drop("median_house_value", axis=1)
y_baseline = train_df_baseline["median_house_value"]

# Handle missing values using SimpleImputer (median strategy)
imputer_baseline = SimpleImputer(strategy='median')
X_imputed_baseline = imputer_baseline.fit_transform(X_baseline)
X_baseline = pd.DataFrame(X_imputed_baseline, columns=X_baseline.columns)

# Split the data into training and validation sets
X_train_baseline, X_val_baseline, y_train_baseline, y_val_baseline = train_test_split(X_baseline, y_baseline, test_size=0.2, random_state=42)

# --- 2. Model Training ---
# 2.1. LightGBM Model
# Changed silent=True to verbose=-1 to suppress verbose output, as silent=True is deprecated
lgbm_model_baseline = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1)
lgbm_model_baseline.fit(X_train_baseline, y_train_baseline)
y_pred_lgbm_baseline = lgbm_model_baseline.predict(X_val_baseline)

# 2.2. XGBoost Model
xgb_model_baseline = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse',
                                      random_state=42, n_jobs=-1, verbosity=0)
xgb_model_baseline.fit(X_train_baseline, y_train_baseline)
y_pred_xgb_baseline = xgb_model_baseline.predict(X_val_baseline)

# --- 3. Ensembling ---
y_pred_ensemble_baseline = (y_pred_lgbm_baseline + y_pred_xgb_baseline) / 2

# --- 4. Evaluation of the Ensembled Model ---
rmse_ensemble_baseline = np.sqrt(mean_squared_error(y_val_baseline, y_pred_ensemble_baseline))
baseline_score = rmse_ensemble_baseline
print(f"Baseline Performance (Ensembled RMSE): {baseline_score:.4f}")

# ===== ABLATION 1: Change Imputation Strategy (median -> mean) =====
print("\nRunning Ablation 1: Change Imputation Strategy (Median to Mean)")
# --- 1. Data Loading and Preprocessing ---
train_df_ab1 = pd.read_csv("./input/train.csv")

# Separate features and target
X_ab1 = train_df_ab1.drop("median_house_value", axis=1)
y_ab1 = train_df_ab1["median_house_value"]

# Handle missing values using SimpleImputer (mean strategy)
imputer_ab1 = SimpleImputer(strategy='mean')  # --- Ablation Change 1 ---
X_imputed_ab1 = imputer_ab1.fit_transform(X_ab1)
X_ab1 = pd.DataFrame(X_imputed_ab1, columns=X_ab1.columns)

# Split the data into training and validation sets
X_train_ab1, X_val_ab1, y_train_ab1, y_val_ab1 = train_test_split(X_ab1, y_ab1, test_size=0.2, random_state=42)

# --- 2. Model Training ---
# 2.1. LightGBM Model
lgbm_model_ab1 = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1)
lgbm_model_ab1.fit(X_train_ab1, y_train_ab1)
y_pred_lgbm_ab1 = lgbm_model_ab1.predict(X_val_ab1)

# 2.2. XGBoost Model
xgb_model_ab1 = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse',
                                 random_state=42, n_jobs=-1, verbosity=0)
xgb_model_ab1.fit(X_train_ab1, y_train_ab1)
y_pred_xgb_ab1 = xgb_model_ab1.predict(X_val_ab1)

# --- 3. Ensembling ---
y_pred_ensemble_ab1 = (y_pred_lgbm_ab1 + y_pred_xgb_ab1) / 2

# --- 4. Evaluation of the Ensembled Model ---
rmse_ensemble_ab1 = np.sqrt(mean_squared_error(y_val_ab1, y_pred_ensemble_ab1))
ablation_1_score = rmse_ensemble_ab1
print(f"Ablation 1 Performance (Ensembled RMSE with Mean Imputation): {ablation_1_score:.4f}")

# ===== ABLATION 2: Remove Ensembling (use only LightGBM) =====
print("\nRunning Ablation 2: Remove Ensembling (Use LightGBM Model Only)")
# --- 1. Data Loading and Preprocessing ---
train_df_ab2 = pd.read_csv("./input/train.csv")

# Separate features and target
X_ab2 = train_df_ab2.drop("median_house_value", axis=1)
y_ab2 = train_df_ab2["median_house_value"]

# Handle missing values using SimpleImputer (median strategy)
imputer_ab2 = SimpleImputer(strategy='median')
X_imputed_ab2 = imputer_ab2.fit_transform(X_ab2)
X_ab2 = pd.DataFrame(X_imputed_ab2, columns=X_ab2.columns)

# Split the data into training and validation sets
X_train_ab2, X_val_ab2, y_train_ab2, y_val_ab2 = train_test_split(X_ab2, y_ab2, test_size=0.2, random_state=42)

# --- 2. Model Training ---
# 2.1. LightGBM Model (used as the sole model for evaluation)
lgbm_model_ab2 = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1)
lgbm_model_ab2.fit(X_train_ab2, y_train_ab2)
y_pred_lgbm_ab2 = lgbm_model_ab2.predict(X_val_ab2)

# --- 3. Ensembling (Skipped for ablation, only LightGBM is used) ---
y_pred_ensemble_ab2 = y_pred_lgbm_ab2  # --- Ablation Change 2: Use only LightGBM predictions ---

# --- 4. Evaluation of the Single Model ---
rmse_ensemble_ab2 = np.sqrt(mean_squared_error(y_val_ab2, y_pred_ensemble_ab2))
ablation_2_score = rmse_ensemble_ab2
print(f"Ablation 2 Performance (LightGBM Only RMSE): {ablation_2_score:.4f}")

# ===== SUMMARY =====
print("\n===== ABLATION STUDY SUMMARY =====")
ablations = [
    ("Baseline (Ensembled, Median Imputation)", baseline_score),
    ("Ablation 1 (Ensembled, Mean Imputation)", ablation_1_score),
    ("Ablation 2 (LightGBM Only, Median Imputation)", ablation_2_score),
]

print("\nPerformance Results (RMSE):")
for name, score in ablations:
    print(f"- {name}: {score:.4f}")

deltas = []
# Calculate absolute difference from baseline for each ablation
for name, score in ablations[1:]:
    delta = abs(score - baseline_score)
    deltas.append((name, delta))

if deltas:
    most_impactful = max(deltas, key=lambda x: x[1])
    print(f"\nMost impactful component's ablation: '{most_impactful[0]}' (Absolute change in RMSE from Baseline: {most_impactful[1]:.4f})")
else:
    print("\nNo ablations performed to compare.")

# Print the final validation performance as per prompt's original code structure
final_validation_score = baseline_score  # Assign baseline_score to final_validation_score for clarity
print(f"Final Validation Performance: {final_validation_score:.4f}")

# ===== PREDICTION ON TEST DATA =====
print("\nMaking predictions on test data...")

# Load test data
test_df = pd.read_csv("./input/test.csv")

# Impute missing values in test data using the imputer fitted on the training data (from baseline)
# Ensure columns match before transforming
X_test_imputed = imputer_baseline.transform(test_df[X_baseline.columns])
X_test = pd.DataFrame(X_test_imputed, columns=X_baseline.columns)

# Predict with baseline LightGBM model
test_pred_lgbm = lgbm_model_baseline.predict(X_test)

# Predict with baseline XGBoost model
test_pred_xgb = xgb_model_baseline.predict(X_test)

# Ensembled predictions
final_predictions = (test_pred_lgbm + test_pred_xgb) / 2

# Ensure predictions are non-negative (house values cannot be negative)
final_predictions[final_predictions < 0] = 0

# Save predictions to submission.csv
output_df = pd.DataFrame({'median_house_value': final_predictions})
output_df.to_csv("submission.csv", index=False, header=False)

print("Predictions saved to submission.csv")
