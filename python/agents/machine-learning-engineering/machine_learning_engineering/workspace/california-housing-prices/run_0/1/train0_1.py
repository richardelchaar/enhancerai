
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

# Load the training data
# As per instructions, assuming the file exists at the specified path and not using try/except.
train_df = pd.read_csv("./input/train.csv")

# Separate features and target variable
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Handle missing values: Impute 'total_bedrooms' with its median
# This is a common preprocessing step for this dataset
if X['total_bedrooms'].isnull().any():
    median_total_bedrooms = X['total_bedrooms'].median()
    X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Split the data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---

# 1. Initialize and Train LightGBM Regressor model (from base solution)
# objective='regression_l2' is for Mean Squared Error, suitable for RMSE.
# metric='rmse' explicitly sets the evaluation metric.
# random_state for reproducibility.
# verbose=-1 suppresses verbose output during training.
lgbm_model = lgb.LGBMRegressor(objective='regression_l2', metric='rmse', random_state=42, verbose=-1)
lgbm_model.fit(X_train, y_train)

# Make predictions on the validation set using LightGBM
y_pred_lgbm = lgbm_model.predict(X_val)

# 2. Initialize and Train XGBoost Regressor model (from reference solution)
# 'objective' 'reg:squarederror' is for regression with squared loss, suitable for RMSE.
# 'eval_metric' 'rmse' explicitly sets the evaluation metric.
# 'random_state' for reproducibility.
# 'verbosity=0' suppresses verbose output during training.
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, verbosity=0)
xgb_model.fit(X_train, y_train)

# Make predictions on the validation set using XGBoost
y_pred_xgb = xgb_model.predict(X_val)

# --- Ensemble Predictions ---
# Simple averaging ensemble of LightGBM and XGBoost predictions
y_pred_ensemble = (y_pred_lgbm + y_pred_xgb) / 2

# --- Evaluate the Ensembled Model ---
# Evaluate the model using Root Mean Squared Error (RMSE)
rmse_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))

# Print the final validation performance
print(f'Final Validation Performance: {rmse_ensemble}')
