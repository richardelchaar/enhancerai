
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
import warnings
import os

# Suppress LightGBM and XGBoost verbose output in ablation runs
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")

# --- DATA LOADING / MOCK DATA GENERATION ---
def generate_mock_data(seed=42):
    np.random.seed(seed)
    data_size = 2000 # Increased for more robust simulation

# Features (aligned with provided CSV schemas - no 'ocean_proximity')
    mock_train_data = {
        'longitude': np.random.uniform(-125, -114, data_size),
        'latitude': np.random.uniform(32, 42, data_size),
        'housing_median_age': np.random.randint(1, 50, data_size),
        'total_rooms': np.random.randint(100, 10000, data_size),
        'total_bedrooms': np.random.randint(50, 2000, data_size),
        'population': np.random.randint(100, 50000, data_size),
        'households': np.random.randint(50, 2000, data_size),
        'median_income': np.random.uniform(0.5, 15, data_size),
    }

train_df_mock = pd.DataFrame(mock_train_data)

# Add some NaNs to numerical features for imputation to work on
    for col in ['total_bedrooms', 'total_rooms']:
        nan_indices = np.random.choice(data_size, int(data_size * 0.05), replace=False)
        train_df_mock.loc[nan_indices, col] = np.nan

# Target
    train_df_mock['median_house_value'] = (
        10000 * train_df_mock['median_income']
        + 500 * train_df_mock['housing_median_age']
        + 100 * train_df_mock['total_rooms'] / (train_df_mock['households'] + 1e-6) # Avoid div by zero
        + np.random.randn(data_size) * 10000 # Add some noise
    )
    train_df_mock['median_house_value'] = np.clip(train_df_mock['median_house_value'], 50000, 500000)

# Test data (similar structure, some NaNs, independent generation)
    mock_test_data = {
        'longitude': np.random.uniform(-125, -114, data_size),
        'latitude': np.random.uniform(32, 42, data_size),
        'housing_median_age': np.random.randint(1, 50, data_size),
        'total_rooms': np.random.randint(100, 10000, data_size),
        'total_bedrooms': np.random.randint(50, 2000, data_size),
        'population': np.random.randint(100, 50000, data_size),
        'households': np.random.randint(50, 2000, data_size),
        'median_income': np.random.uniform(0.5, 15, data_size),
    }
    test_df_mock = pd.DataFrame(mock_test_data)

# Add some NaNs to numerical features for imputation to work on
    for col in ['total_bedrooms', 'total_rooms']:
        nan_indices = np.random.choice(data_size, int(data_size * 0.05), replace=False)
        test_df_mock.loc[nan_indices, col] = np.nan

return train_df_mock, test_df_mock

# Function to load data or generate mock data
def load_or_generate_data(seed=42):
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"

if os.path.exists(train_path) and os.path.exists(test_path):
        print("Loading data from ./input directory...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        # Ensure test_df does not have median_house_value if loaded from file (it shouldn't based on description)
        test_df = test_df.drop(columns=['median_house_value'], errors='ignore')
    else:
        print("Input files not found. Generating mock data...")
        train_df, test_df = generate_mock_data(seed)
    return train_df, test_df

# Store results
baseline_score = 0
ablation_1_score = 0
ablation_2_score = 0
final_validation_score = 0 # To store the final model's performance for the required print statement

# ==============================================================================
# ===== BASELINE: Original Code =====
print("Running Baseline: Original Code with Ensemble and Median Imputation...")

train_df, test_df = load_or_generate_data(seed=42)

# Separate features and target from training data
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Identify numerical features for imputation
numerical_features = X.select_dtypes(include=np.number).columns

# Impute missing values using median strategy
# Fit the imputer on the training features, then transform both training features and test data
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X[numerical_features])
test_df_imputed = imputer.transform(test_df[numerical_features])

# Convert back to DataFrame, preserving column names
X[numerical_features] = X_imputed
test_df[numerical_features] = test_df_imputed

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model 1: LightGBM ---
# Initialize and train the LGBMRegressor model
model_lgbm = lgb.LGBMRegressor(objective='regression_l2', metric='rmse', random_state=42, verbose=-1, n_jobs=-1)
model_lgbm.fit(X_train, y_train)

# Make predictions on the validation set with LGBM
y_pred_lgbm_val = model_lgbm.predict(X_val)

# --- Model 2: XGBoost ---
# Initialize and train the XGBRegressor model
model_xgb = xgb.XGBRegressor(objective='reg:squarederror',
                             eval_metric='rmse',
                             random_state=42,
                             n_jobs=-1,
                             verbosity=0)  # Suppress verbose output
model_xgb.fit(X_train, y_train)

# Make predictions on the validation set with XGBoost
y_pred_xgb_val = model_xgb.predict(X_val)

# --- Ensemble the predictions ---
# Simple averaging ensemble
y_pred_ensemble_val = (y_pred_lgbm_val + y_pred_xgb_val) / 2

# Calculate RMSE on the validation set for the ensembled predictions
rmse_val_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_ensemble_val))

baseline_score = rmse_val_ensemble
print(f"Baseline Performance (RMSE): {baseline_score:.4f}")
final_validation_score = baseline_score # Set the final score to baseline for the required print

# ==============================================================================
# ===== ABLATION 1: Remove Ensemble (Use only LightGBM) =====
print("\nRunning Ablation 1: Removing Ensemble (using only LightGBM predictions)...")

train_df_ablation_1, test_df_ablation_1 = load_or_generate_data(seed=42) # Reload data to ensure clean slate for ablation

# Separate features and target from training data
X_ablation_1 = train_df_ablation_1.drop("median_house_value", axis=1)
y_ablation_1 = train_df_ablation_1["median_house_value"]

# Identify numerical features for imputation
numerical_features_ablation_1 = X_ablation_1.select_dtypes(include=np.number).columns

# Impute missing values using median strategy (same as baseline)
imputer_ablation_1 = SimpleImputer(strategy='median')
X_ablation_1_imputed = imputer_ablation_1.fit_transform(X_ablation_1[numerical_features_ablation_1])
test_df_ablation_1_imputed = imputer_ablation_1.transform(test_df_ablation_1[numerical_features_ablation_1])

X_ablation_1[numerical_features_ablation_1] = X_ablation_1_imputed
test_df_ablation_1[numerical_features_ablation_1] = test_df_ablation_1_imputed

# Split the training data into training and validation sets
X_train_ablation_1, X_val_ablation_1, y_train_ablation_1, y_val_ablation_1 = train_test_split(X_ablation_1, y_ablation_1, test_size=0.2, random_state=42)

# --- Model 1: LightGBM ---
# Initialize and train the LGBMRegressor model
model_lgbm_ablation_1 = lgb.LGBMRegressor(objective='regression_l2', metric='rmse', random_state=42, verbose=-1, n_jobs=-1)
model_lgbm_ablation_1.fit(X_train_ablation_1, y_train_ablation_1)

# Make predictions on the validation set with LGBM
y_pred_lgbm_val_ablation_1 = model_lgbm_ablation_1.predict(X_val_ablation_1)

# --- Model 2: XGBoost --- (Still trained but its predictions won't be used for final score)
model_xgb_ablation_1 = xgb.XGBRegressor(objective='reg:squarederror',
                             eval_metric='rmse',
                             random_state=42,
                             n_jobs=-1,
                             verbosity=0)
model_xgb_ablation_1.fit(X_train_ablation_1, y_train_ablation_1)
# y_pred_xgb_val = model_xgb.predict(X_val) # Predictions are made but not used for the final metric here

# --- Ablated: No Ensemble, use only LightGBM predictions ---
y_pred_ablation_1_val = y_pred_lgbm_val_ablation_1 # Use only LGBM predictions

# Calculate RMSE on the validation set for the ablated predictions
rmse_ablation_1 = np.sqrt(mean_squared_error(y_val_ablation_1, y_pred_ablation_1_val))

ablation_1_score = rmse_ablation_1
print(f"Ablation 1 Performance (RMSE - Only LightGBM): {ablation_1_score:.4f}")

# ==============================================================================
# ===== ABLATION 2: Change Imputation Strategy (Median to Mean) =====
print("\nRunning Ablation 2: Changing Imputation Strategy from Median to Mean...")

train_df_ablation_2, test_df_ablation_2 = load_or_generate_data(seed=42) # Reload data to ensure clean slate for ablation

# Separate features and target from training data
X_ablation_2 = train_df_ablation_2.drop("median_house_value", axis=1)
y_ablation_2 = train_df_ablation_2["median_house_value"]

# Identify numerical features for imputation
numerical_features_ablation_2 = X_ablation_2.select_dtypes(include=np.number).columns

# Impute missing values using MEAN strategy (Ablation: changed from 'median')
imputer_ablation_2 = SimpleImputer(strategy='mean')
X_ablation_2_imputed = imputer_ablation_2.fit_transform(X_ablation_2[numerical_features_ablation_2])
test_df_ablation_2_imputed = imputer_ablation_2.transform(test_df_ablation_2[numerical_features_ablation_2])

X_ablation_2[numerical_features_ablation_2] = X_ablation_2_imputed
test_df_ablation_2[numerical_features_ablation_2] = test_df_ablation_2_imputed

# Split the training data into training and validation sets
X_train_ablation_2, X_val_ablation_2, y_train_ablation_2, y_val_ablation_2 = train_test_split(X_ablation_2, y_ablation_2, test_size=0.2, random_state=42)

# --- Model 1: LightGBM ---
# Initialize and train the LGBMRegressor model
model_lgbm_ablation_2 = lgb.LGBMRegressor(objective='regression_l2', metric='rmse', random_state=42, verbose=-1, n_jobs=-1)
model_lgbm_ablation_2.fit(X_train_ablation_2, y_train_ablation_2)

# Make predictions on the validation set with LGBM
y_pred_lgbm_val_ablation_2 = model_lgbm_ablation_2.predict(X_val_ablation_2)

# --- Model 2: XGBoost ---
# Initialize and train the XGBRegressor model
model_xgb_ablation_2 = xgb.XGBRegressor(objective='reg:squarederror',
                             eval_metric='rmse',
                             random_state=42,
                             n_jobs=-1,
                             verbosity=0)
model_xgb_ablation_2.fit(X_train_ablation_2, y_train_ablation_2)

# Make predictions on the validation set with XGBoost
y_pred_xgb_val_ablation_2 = model_xgb_ablation_2.predict(X_val_ablation_2)

# --- Ensemble the predictions (same as baseline) ---
y_pred_ensemble_ablation_2_val = (y_pred_lgbm_val_ablation_2 + y_pred_xgb_val_ablation_2) / 2

# Calculate RMSE on the validation set for the ensembled predictions
rmse_ablation_2 = np.sqrt(mean_squared_error(y_val_ablation_2, y_pred_ensemble_ablation_2_val))

ablation_2_score = rmse_ablation_2
print(f"Ablation 2 Performance (RMSE - Mean Imputation): {ablation_2_score:.4f}")

# ==============================================================================
# ===== SUMMARY =====
print("\n===== ABLATION STUDY SUMMARY =====")
ablations = [
    ("Baseline (Ensemble + Median Imputation)", baseline_score),
    ("Ablation 1 (Only LightGBM)", ablation_1_score),
    ("Ablation 2 (Mean Imputation)", ablation_2_score),
]

print("--- Performance (Lower RMSE is Better) ---")
for name, score in ablations:
    print(f"- {name}: {score:.4f}")

print("\n--- Impact Analysis (Absolute Change from Baseline) ---")
deltas = []
for name, score in ablations[1:]: # Exclude baseline from delta calculation
    delta = abs(score - baseline_score)
    deltas.append((name, delta))
    print(f"- {name}: Delta from Baseline = {delta:.4f}")

if deltas:
    most_impactful = max(deltas, key=lambda x: x[1])
    print(f"\nMost impactful component ablation: '{most_impactful[0]}' (changed performance by {most_impactful[1]:.4f} RMSE).")
else:
    print("\nNo ablations performed to compare.")

print(f"Final Validation Performance: {final_validation_score}")

# ==============================================================================
# ===== FINAL PREDICTION FOR SUBMISSION =====
print("\nGenerating final predictions for submission using the Baseline Ensemble Model...")

# Load fresh data for final training to ensure no data leakage or state issues from ablations
final_train_df, final_test_df = load_or_generate_data(seed=42)

# Separate features and target from training data
X_final = final_train_df.drop("median_house_value", axis=1)
y_final = final_train_df["median_house_value"]

# Identify numerical features for imputation
numerical_features_final = X_final.select_dtypes(include=np.number).columns

# Impute missing values using median strategy (as chosen for baseline)
final_imputer = SimpleImputer(strategy='median')
X_final_imputed = final_imputer.fit_transform(X_final[numerical_features_final])
final_test_df_imputed = final_imputer.transform(final_test_df[numerical_features_final])

X_final[numerical_features_final] = X_final_imputed
final_test_df[numerical_features_final] = final_test_df_imputed

# --- Retrain Model 1: LightGBM on FULL training data ---
final_model_lgbm = lgb.LGBMRegressor(objective='regression_l2', metric='rmse', random_state=42, verbose=-1, n_jobs=-1)
final_model_lgbm.fit(X_final, y_final)

# --- Retrain Model 2: XGBoost on FULL training data ---
final_model_xgb = xgb.XGBRegressor(objective='reg:squarederror',
                             eval_metric='rmse',
                             random_state=42,
                             n_jobs=-1,
                             verbosity=0)
final_model_xgb.fit(X_final, y_final)

# Make predictions on the final test set with both models
test_predictions_lgbm = final_model_lgbm.predict(final_test_df)
test_predictions_xgb = final_model_xgb.predict(final_test_df)

# Ensemble the predictions for the final submission
final_test_predictions = (test_predictions_lgbm + test_predictions_xgb) / 2

# Create submission file
submission_df = pd.DataFrame({'median_house_value': final_test_predictions})
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")