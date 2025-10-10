
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

# --- 1. Data Loading and Preprocessing ---
# Load the training data. Assume file exists as per instructions.
train_df = pd.read_csv("./input/train.csv")

# Separate features and target
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Handle missing values using SimpleImputer (median strategy)
# This step is common to both base and reference solutions
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Model Training ---

# 2.1. LightGBM Model (from Base Solution)
print("Training LightGBM model...")
lgbm_model = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, silent=True)
lgbm_model.fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_val)
rmse_lgbm = np.sqrt(mean_squared_error(y_val, y_pred_lgbm))
print(f"LightGBM Validation RMSE: {rmse_lgbm}")

# 2.2. XGBoost Model (from Reference Solution)
print("Training XGBoost model...")
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse',
                             random_state=42, n_jobs=-1, verbosity=0)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_val)
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
print(f"XGBoost Validation RMSE: {rmse_xgb}")

# --- 3. Ensembling ---
# Simple ensembling: average the predictions from both models
print("Ensembling models...")
y_pred_ensemble = (y_pred_lgbm + y_pred_xgb) / 2

# --- 4. Evaluation of the Ensembled Model ---
rmse_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))

# Print the final validation performance
print(f"Final Validation Performance: {rmse_ensemble}")
