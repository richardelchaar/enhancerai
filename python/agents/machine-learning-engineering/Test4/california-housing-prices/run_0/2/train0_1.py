
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

# Load the datasets
# As per instructions, assume files are in './input' and do not use try/except.
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate target variable from features
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# --- Preprocessing ---
# Handle missing values
# Impute missing values with the median of the column from the training data.
# This ensures consistency between training and test set imputation.
for col in X.columns:
    if X[col].isnull().any():
        median_val = X[col].median()  # Calculate median from training data
        X[col].fillna(median_val, inplace=True)
        # Apply the same imputation to the test set using the training set's median
        if col in test_df.columns:
            test_df[col].fillna(median_val, inplace=True)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training and Prediction ---

# 1. LightGBM Model (from base solution)
# Initialize LightGBM Regressor model
lgbm_model = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1)
# Train the LightGBM model
lgbm_model.fit(X_train, y_train)
# Make predictions on the validation set with LightGBM
y_pred_lgbm = lgbm_model.predict(X_val)

# 2. XGBoost Model (from reference solution)
# Initialize XGBoost Regressor model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, verbosity=0)
# Train the XGBoost model
xgb_model.fit(X_train, y_train)
# Make predictions on the validation set with XGBoost
y_pred_xgb = xgb_model.predict(X_val)

# --- Ensembling ---
# Simple average ensemble of the two models' predictions
y_pred_ensemble = (y_pred_lgbm + y_pred_xgb) / 2

# --- Evaluation ---
# Calculate RMSE on the validation set for the ensembled predictions
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))

# Print the final validation performance
print(f"Final Validation Performance: {rmse_val}")

# Optional: Prepare for submission
# Make predictions on the test set using both models
# y_pred_lgbm_test = lgbm_model.predict(test_df)
# y_pred_xgb_test = xgb_model.predict(test_df)
# y_pred_ensemble_test = (y_pred_lgbm_test + y_pred_xgb_test) / 2

# Create submission file (example structure)
# submission_df = pd.DataFrame({'median_house_value': y_pred_ensemble_test})
# submission_df.to_csv('submission.csv', index=False)
