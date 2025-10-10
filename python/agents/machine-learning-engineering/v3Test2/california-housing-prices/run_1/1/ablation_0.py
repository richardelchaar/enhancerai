
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
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# --- Dummy Data Generation (to make the script runnable) ---
# Create 'input' directory if it doesn't exist
if not os.path.exists("./input"):
    os.makedirs("./input")

# Check if dummy data files already exist; if not, generate them
if not (os.path.exists("./input/train.csv") and os.path.exists("./input/test.csv")):
    print("Generating dummy data files...")
    np.random.seed(42)
    data_size = 1000 # Smaller size for quicker execution
    
    # Generate dummy data for train.csv
    train_data = {
        'total_rooms': np.random.randint(100, 5000, data_size),
        'total_bedrooms': np.random.randint(50, 1000, data_size),
        'population': np.random.randint(100, 3000, data_size),
        'households': np.random.randint(50, 900, data_size),
        'median_income': np.random.rand(data_size) * 10,
        'housing_median_age': np.random.randint(1, 50, data_size),
        'latitude': np.random.uniform(32, 42, data_size),
        'longitude': np.random.uniform(-125, -114, data_size),
        'ocean_proximity': np.random.choice(['<1H OCEAN', 'INLAND', 'NEAR BAY', 'NEAR OCEAN', 'ISLAND'], data_size),
        'median_house_value': np.random.randint(50000, 500000, data_size)
    }
    train_df_gen = pd.DataFrame(train_data)

# Introduce some NaN values to test imputation
    num_nans = int(0.05 * data_size)
    train_df_gen.loc[np.random.choice(train_df_gen.index, num_nans, replace=False), 'total_bedrooms'] = np.nan
    train_df_gen.loc[np.random.choice(train_df_gen.index, num_nans, replace=False), 'population'] = np.nan
    train_df_gen.loc[np.random.choice(train_df_gen.index, num_nans, replace=False), 'housing_median_age'] = np.nan

# Introduce some inf values
    num_infs = int(0.01 * data_size)
    train_df_gen.loc[np.random.choice(train_df_gen.index, num_infs, replace=False), 'total_rooms'] = np.inf

# Generate dummy data for test.csv
    test_data = {
        'total_rooms': np.random.randint(100, 5000, data_size // 4),
        'total_bedrooms': np.random.randint(50, 1000, data_size // 4),
        'population': np.random.randint(100, 3000, data_size // 4),
        'households': np.random.randint(50, 900, data_size // 4),
        'median_income': np.random.rand(data_size // 4) * 10,
        'housing_median_age': np.random.randint(1, 50, data_size // 4),
        'latitude': np.random.uniform(32, 42, data_size // 4),
        'longitude': np.random.uniform(-125, -114, data_size // 4),
        'ocean_proximity': np.random.choice(['<1H OCEAN', 'INLAND', 'NEAR BAY'], data_size // 4),
    }
    test_df_gen = pd.DataFrame(test_data)

# Introduce some NaN values to test imputation
    num_nans_test = int(0.05 * (data_size // 4))
    test_df_gen.loc[np.random.choice(test_df_gen.index, num_nans_test, replace=False), 'total_bedrooms'] = np.nan

# Introduce some inf values
    num_infs_test = int(0.01 * (data_size // 4))
    test_df_gen.loc[np.random.choice(test_df_gen.index, num_infs_test, replace=False), 'total_rooms'] = -np.inf

# Fix IndentationError: These lines should be at the same indentation level
    train_df_gen.to_csv("./input/train.csv", index=False)
    test_df_gen.to_csv("./input/test.csv", index=False)
    print("Dummy data generation complete.")
else:
    print("Dummy data files already exist. Skipping generation.")

# ===== BASELINE: Original Code =====
print("Running Baseline...")
# Load datasets
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Prepare the data
# Separate target variable from features
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Identify features for the test set
X_test_submission = test_df.copy()

# Handle missing values (e.g., in 'total_bedrooms') and infinite values
numerical_cols_to_impute = ['total_rooms', 'total_bedrooms', 'population', 'households']

for col in numerical_cols_to_impute:
    if col in X.columns:
        # Replace infinities with NaN first
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        # Calculate median only from the training features to prevent data leakage
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        # Apply the same median to the test set
        if col in X_test_submission.columns:
            X_test_submission[col] = X_test_submission[col].replace([np.inf, -np.inf], np.nan)
            X_test_submission[col].fillna(median_val, inplace=True)

# The 'ocean_proximity' column is not present in the provided dataset schema
# and attempting to use pd.get_dummies on it causes a KeyError.
# Therefore, the lines related to 'ocean_proximity' are commented out.
# This implies 'ocean_proximity' is effectively NOT USED as a feature.
# Explicitly drop it if it exists from dummy data, to match the original code's intent.
if 'ocean_proximity' in X.columns:
    X = X.drop('ocean_proximity', axis=1)
if 'ocean_proximity' in X_test_submission.columns:
    X_test_submission = X_test_submission.drop('ocean_proximity', axis=1)

# Align columns - crucial for consistent feature sets between training and test after one-hot encoding
# This ensures that if a category is missing in one set but present in another, columns are aligned.
train_cols = X.columns
test_cols = X_test_submission.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test_submission[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X[c] = 0

# Ensure the order of columns is the same for both datasets
X_test_submission = X_test_submission[train_cols]
X = X[train_cols]

# Split the training data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training and Prediction ---

# 1. Initialize and train the LightGBM Regressor model
lgbm = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1)
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_val)

# 2. Initialize and train the CatBoost Regressor model
catboost = CatBoostRegressor(iterations=100,  # Number of boosting iterations (trees)
                             learning_rate=0.1, # Step size shrinkage
                             depth=6,         # Depth of trees
                             loss_function='RMSE', # Loss function to optimize
                             random_seed=42,    # For reproducibility
                             verbose=False,     # Suppress training output
                             allow_writing_files=False) # Prevent CatBoost from writing diagnostic files
catboost.fit(X_train, y_train)
y_pred_catboost = catboost.predict(X_val)

# --- Ensembling ---
# Simple average ensembling of predictions from LightGBM and CatBoost
y_pred_ensemble = (y_pred_lgbm + y_pred_catboost) / 2

# Evaluate the ensembled model using Root Mean Squared Error (RMSE)
rmse_val_baseline = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))

# Print the final validation performance
print(f'Baseline Performance: {rmse_val_baseline:.4f}')
baseline_score = rmse_val_baseline

# Make predictions on the actual test set for submission (optional for ablation study but good practice)
y_pred_lgbm_test = lgbm.predict(X_test_submission)
y_pred_catboost_test = catboost.predict(X_test_submission)
final_predictions = (y_pred_lgbm_test + y_pred_catboost_test) / 2
submission_df = pd.DataFrame(final_predictions, columns=['median_house_value'])
submission_df.to_csv('submission_baseline.csv', index=False)

# ===== ABLATION 1: Remove Ensembling (Use only LightGBM) =====
print("\nRunning Ablation 1: Remove Ensembling (Use only LightGBM)...")
# Load datasets
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Prepare the data
# Separate target variable from features
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Identify features for the test set
X_test_submission = test_df.copy()

# Handle missing values (e.g., in 'total_bedrooms') and infinite values
numerical_cols_to_impute = ['total_rooms', 'total_bedrooms', 'population', 'households']

for col in numerical_cols_to_impute:
    if col in X.columns:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        if col in X_test_submission.columns:
            X_test_submission[col] = X_test_submission[col].replace([np.inf, -np.inf], np.nan)
            X_test_submission[col].fillna(median_val, inplace=True)

# Drop 'ocean_proximity' if it exists, matching baseline behavior
if 'ocean_proximity' in X.columns:
    X = X.drop('ocean_proximity', axis=1)
if 'ocean_proximity' in X_test_submission.columns:
    X_test_submission = X_test_submission.drop('ocean_proximity', axis=1)

# Align columns
train_cols = X.columns
test_cols = X_test_submission.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test_submission[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X[c] = 0

X_test_submission = X_test_submission[train_cols]
X = X[train_cols]

# Split the training data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training and Prediction ---

# 1. Initialize and train the LightGBM Regressor model
lgbm = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1)
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_val)

# 2. Initialize and train the CatBoost Regressor model (still trained, but its predictions won't be used in ensemble)
catboost = CatBoostRegressor(iterations=100,
                             learning_rate=0.1,
                             depth=6,
                             loss_function='RMSE',
                             random_seed=42,
                             verbose=False,
                             allow_writing_files=False)
catboost.fit(X_train, y_train)
# y_pred_catboost = catboost.predict(X_val) # Not used in ensemble for ablation 1

# --- Ensembling (Ablated: Use only LightGBM) ---
y_pred_ensemble_ablation1 = y_pred_lgbm # Use only LightGBM predictions

# Evaluate performance
rmse_val_ablation1 = np.sqrt(mean_squared_error(y_val, y_pred_ensemble_ablation1))
print(f"Ablation 1 Performance: {rmse_val_ablation1:.4f}")
ablation_1_score = rmse_val_ablation1

# Make predictions for submission (optional)
y_pred_lgbm_test = lgbm.predict(X_test_submission)
# y_pred_catboost_test = catboost.predict(X_test_submission) # Not used
final_predictions_ablation1 = y_pred_lgbm_test
submission_df = pd.DataFrame(final_predictions_ablation1, columns=['median_house_value'])
submission_df.to_csv('submission_ablation1.csv', index=False)

# ===== ABLATION 2: Default CatBoost Hyperparameters =====
print("\nRunning Ablation 2: Default CatBoost Hyperparameters...")
# Load datasets
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Prepare the data
# Separate target variable from features
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Identify features for the test set
X_test_submission = test_df.copy()

# Handle missing values (e.g., in 'total_bedrooms') and infinite values
numerical_cols_to_impute = ['total_rooms', 'total_bedrooms', 'population', 'households']

for col in numerical_cols_to_impute:
    if col in X.columns:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        if col in X_test_submission.columns:
            X_test_submission[col] = X_test_submission[col].replace([np.inf, -np.inf], np.nan)
            X_test_submission[col].fillna(median_val, inplace=True)

# Drop 'ocean_proximity' if it exists, matching baseline behavior
if 'ocean_proximity' in X.columns:
    X = X.drop('ocean_proximity', axis=1)
if 'ocean_proximity' in X_test_submission.columns:
    X_test_submission = X_test_submission.drop('ocean_proximity', axis=1)

# Align columns
train_cols = X.columns
test_cols = X_test_submission.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test_submission[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X[c] = 0

X_test_submission = X_test_submission[train_cols]
X = X[train_cols]

# Split the training data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training and Prediction ---

# 1. Initialize and train the LightGBM Regressor model
lgbm = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1)
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_val)

# 2. Initialize and train the CatBoost Regressor model (Ablated: default hyperparameters)
catboost = CatBoostRegressor(loss_function='RMSE', # Still use RMSE as loss for fair comparison
                             random_seed=42,
                             verbose=False,
                             allow_writing_files=False) # Removed iterations, learning_rate, depth
catboost.fit(X_train, y_train)
y_pred_catboost = catboost.predict(X_val)

# --- Ensembling ---
# Simple average ensembling of predictions from LightGBM and CatBoost
y_pred_ensemble_ablation2 = (y_pred_lgbm + y_pred_catboost) / 2

# Evaluate performance
rmse_val_ablation2 = np.sqrt(mean_squared_error(y_val, y_pred_ensemble_ablation2))
print(f"Ablation 2 Performance: {rmse_val_ablation2:.4f}")
ablation_2_score = rmse_val_ablation2

# Make predictions for submission (optional)
y_pred_lgbm_test = lgbm.predict(X_test_submission)
y_pred_catboost_test = catboost.predict(X_test_submission)
final_predictions_ablation2 = (y_pred_lgbm_test + y_pred_catboost_test) / 2
submission_df = pd.DataFrame(final_predictions_ablation2, columns=['median_house_value'])
submission_df.to_csv('submission_ablation2.csv', index=False)

# ===== ABLATION STUDY SUMMARY =====
print("\n===== ABLATION STUDY SUMMARY =====")
ablations = [
    ("Baseline", baseline_score),
    ("Ablation 1: Remove Ensembling (LightGBM only)", ablation_1_score),
    ("Ablation 2: Default CatBoost Hyperparameters", ablation_2_score),
]

print("--- Performances ---")
for name, score in ablations:
    print(f"{name}: {score:.4f}")

print("\n--- Impact Analysis ---")
deltas = []
for name, score in ablations[1:]: # Start from index 1 to skip baseline
    delta = abs(score - baseline_score)
    deltas.append((name, delta))
    print(f"Change from Baseline for '{name}': {delta:.4f}")

if deltas:
    most_impactful = max(deltas, key=lambda x: x[1])
    print(f"\nMost impactful component's ablation: '{most_impactful[0]}' (resulted in a change of {most_impactful[1]:.4f} RMSE from baseline)")
else:
    print("No ablations performed to compare.")

print(f'Final Validation Performance: {baseline_score}')