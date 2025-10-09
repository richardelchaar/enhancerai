
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

# Based on the problem description: "All the provided input data is stored in "./input" directory."
# This implies that the 'input' directory is relative to the current working directory from which the script is run.

# Load the training and test data
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')

# Separate features and target from training data
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Identify numerical features for imputation
numerical_features = X.select_dtypes(include=np.number).columns

# Impute missing values using median strategy
# Fit the imputer on the training features, then transform both training features and test data
imputer = SimpleImputer(strategy='median')
X[numerical_features] = imputer.fit_transform(X[numerical_features])
test_df[numerical_features] = imputer.transform(test_df[numerical_features])

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model 1: LightGBM ---
# Initialize and train the LGBMRegressor model
# Suppress verbose output with verbose=-1
model_lgbm = lgb.LGBMRegressor(objective='regression_l2', metric='rmse', random_state=42, verbose=-1, n_jobs=-1)
model_lgbm.fit(X_train, y_train)

# Make predictions on the validation set with LGBM
y_pred_lgbm_val = model_lgbm.predict(X_val)

# --- Model 2: XGBoost ---
# Initialize and train the XGBRegressor model
# Suppress verbose output with verbosity=0
model_xgb = xgb.XGBRegressor(objective='reg:squareerror',
                             eval_metric='rmse',
                             random_state=42,
                             n_jobs=-1,
                             verbosity=0) # verbosity=0 suppresses output
model_xgb.fit(X_train, y_train)

# Make predictions on the validation set with XGBoost
y_pred_xgb_val = model_xgb.predict(X_val)

# --- Ensemble the predictions ---
# Simple averaging ensemble
y_pred_ensemble_val = (y_pred_lgbm_val + y_pred_xgb_val) / 2

# Calculate RMSE on the validation set for the ensembled predictions
rmse_val_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_ensemble_val))

# Print the final validation performance as required
print(f"Final Validation Performance: {rmse_val_ensemble}")

# --- Final predictions on the test set ---
# Make predictions on the actual test data using the trained models
y_pred_lgbm_test = model_lgbm.predict(test_df)
y_pred_xgb_test = model_xgb.predict(test_df)

# Ensemble the test predictions
y_pred_ensemble_test = (y_pred_lgbm_test + y_pred_xgb_test) / 2

# Create the submission file
submission_df = pd.DataFrame({'median_house_value': y_pred_ensemble_test})

# Ensure the output directory exists, relative to the current working directory
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

submission_path = os.path.join(output_dir, 'submission.csv')
submission_df.to_csv(submission_path, index=False, header=True)