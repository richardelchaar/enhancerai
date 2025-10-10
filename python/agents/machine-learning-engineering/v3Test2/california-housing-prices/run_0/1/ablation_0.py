
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

# Create dummy input directory and train.csv if they don't exist
# This is for local execution convenience if the files aren't physically present.
import os
if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/train.csv"):
    # Generate a dummy dataset similar to California Housing for demonstration
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    train_df_dummy = pd.DataFrame(housing.data, columns=housing.feature_names)
    train_df_dummy['median_house_value'] = housing.target * 100000 # Scale to typical values
    train_df_dummy['total_bedrooms'] = train_df_dummy['total_bedrooms'].astype(float) # Ensure float type
    # Introduce some missing values for testing imputation
    np.random.seed(42)
    missing_indices = np.random.choice(train_df_dummy.index, size=int(len(train_df_dummy) * 0.01), replace=False)
    train_df_dummy.loc[missing_indices, 'total_bedrooms'] = np.nan
    train_df_dummy.to_csv("./input/train.csv", index=False)
    print("Generated dummy 'train.csv' for demonstration.")

# ===== BASELINE: Original Code =====
print("Running Baseline...")

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# Separate target variable and features
TARGET_COL = 'median_house_value'
X = train_df.drop(columns=[TARGET_COL])
y = train_df[TARGET_COL]

# Handle missing values in features - impute with median for numerical columns
# 'total_bedrooms' is a common column with missing values in this dataset type.
for col in X.columns:
    if X[col].isnull().any():
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())

# Split the data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- LightGBM Model Integration ---
# Initialize LightGBM Regressor model
model_lgbm = lgb.LGBMRegressor(objective='regression',
                               metric='rmse',
                               n_estimators=100,
                               random_state=42,
                               verbose=-1) # Suppress verbose output

# Train the LightGBM model
model_lgbm.fit(X_train, y_train)

# Make predictions on the validation set using LightGBM
y_pred_val_lgbm = model_lgbm.predict(X_val)

# --- CatBoost Model Integration ---
# Identify categorical features for CatBoost.
# Based on the dataset description, all input features are numerical.
categorical_features_indices = []

# Initialize CatBoost Regressor
model_catboost = CatBoostRegressor(
    loss_function='RMSE',
    iterations=100,
    random_state=42,
    verbose=0, # Suppress model output during training
    allow_writing_files=False # Suppress writing model files to disk
)

# Train the CatBoost model
model_catboost.fit(X_train, y_train, cat_features=categorical_features_indices)

# Make predictions on the validation set using CatBoost
y_pred_val_catboost = model_catboost.predict(X_val)

# --- Ensembling Predictions ---
# A simple average ensemble of the two models
y_pred_val_ensemble = (y_pred_val_lgbm + y_pred_val_catboost) / 2

# Calculate Root Mean Squared Error on the validation set for the ensembled predictions
rmse_val_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble))

baseline_score = rmse_val_ensemble
print(f"Baseline Performance (LGBM + CatBoost Ensemble): {baseline_score:.4f}")

# ===== ABLATION 1: Remove LightGBM from Ensemble =====
print("\nRunning Ablation 1: Remove LightGBM from Ensemble (CatBoost only)...")

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# Separate target variable and features
TARGET_COL = 'median_house_value'
X = train_df.drop(columns=[TARGET_COL])
y = train_df[TARGET_COL]

# Handle missing values in features - impute with median for numerical columns
for col in X.columns:
    if X[col].isnull().any():
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- LightGBM Model Integration (Still trained, but not used in ensemble) ---
model_lgbm = lgb.LGBMRegressor(objective='regression',
                               metric='rmse',
                               n_estimators=100,
                               random_state=42,
                               verbose=-1)
model_lgbm.fit(X_train, y_train)
y_pred_val_lgbm = model_lgbm.predict(X_val)

# --- CatBoost Model Integration ---
categorical_features_indices = []
model_catboost = CatBoostRegressor(
    loss_function='RMSE',
    iterations=100,
    random_state=42,
    verbose=0,
    allow_writing_files=False
)
model_catboost.fit(X_train, y_train, cat_features=categorical_features_indices)
y_pred_val_catboost = model_catboost.predict(X_val)

# --- Ensembling Predictions (Ablation: CatBoost only) ---
y_pred_val_ensemble_ablation1 = y_pred_val_catboost # Only use CatBoost predictions

rmse_val_ablation1 = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble_ablation1))
ablation_1_score = rmse_val_ablation1
print(f"Ablation 1 Performance (CatBoost only): {ablation_1_score:.4f}")

# ===== ABLATION 2: Remove CatBoost from Ensemble =====
print("\nRunning Ablation 2: Remove CatBoost from Ensemble (LightGBM only)...")

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# Separate target variable and features
TARGET_COL = 'median_house_value'
X = train_df.drop(columns=[TARGET_COL])
y = train_df[TARGET_COL]

# Handle missing values in features - impute with median for numerical columns
for col in X.columns:
    if X[col].isnull().any():
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- LightGBM Model Integration ---
model_lgbm = lgb.LGBMRegressor(objective='regression',
                               metric='rmse',
                               n_estimators=100,
                               random_state=42,
                               verbose=-1)
model_lgbm.fit(X_train, y_train)
y_pred_val_lgbm = model_lgbm.predict(X_val)

# --- CatBoost Model Integration (Still trained, but not used in ensemble) ---
categorical_features_indices = []
model_catboost = CatBoostRegressor(
    loss_function='RMSE',
    iterations=100,
    random_state=42,
    verbose=0,
    allow_writing_files=False
)
model_catboost.fit(X_train, y_train, cat_features=categorical_features_indices)
y_pred_val_catboost = model_catboost.predict(X_val)

# --- Ensembling Predictions (Ablation: LightGBM only) ---
y_pred_val_ensemble_ablation2 = y_pred_val_lgbm # Only use LightGBM predictions

rmse_val_ablation2 = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble_ablation2))
ablation_2_score = rmse_val_ablation2
print(f"Ablation 2 Performance (LightGBM only): {ablation_2_score:.4f}")

# ===== SUMMARY =====
print("\n===== ABLATION STUDY SUMMARY =====")
ablations = [
    ("Baseline (LGBM + CatBoost Ensemble)", baseline_score),
    ("Ablation 1 (CatBoost only)", ablation_1_score),
    ("Ablation 2 (LightGBM only)", ablation_2_score),
]

print("\nPerformance Scores:")
for name, score in ablations:
    print(f"- {name}: {score:.4f}")

deltas = [(name, abs(score - baseline_score)) for name, score in ablations[1:]]
most_impactful_change_item = max(deltas, key=lambda x: x[1])

print(f"\nMost impactful component (largest change from baseline when ablated): {most_impactful_change_item[0]} (delta: {most_impactful_change_item[1]:.4f})")