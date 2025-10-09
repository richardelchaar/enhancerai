
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

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

# Print the final validation performance of the ensemble
print(f"Final Validation Performance: {rmse_val_ensemble}")