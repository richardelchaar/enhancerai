
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
from sklearn.model_selection import RandomizedSearchCV
import os
import re

# Load the training data
train_df = pd.read_csv("./input/train.csv")
# Load test_df early for consistent preprocessing of categorical features
test_df = pd.read_csv("./input/test.csv")

# Store the median of 'total_bedrooms' from the training data before imputation for consistency
train_total_bedrooms_median = train_df['total_bedrooms'].median()

# Handle missing values: Fill 'total_bedrooms' with the median from the training data
if 'total_bedrooms' in train_df.columns:
    train_df['total_bedrooms'].fillna(train_total_bedrooms_median, inplace=True)
if 'total_bedrooms' in test_df.columns:
    test_df['total_bedrooms'].fillna(train_total_bedrooms_median, inplace=True)

# Apply one-hot encoding to handle categorical features for both train and test data.
# Concatenate train and test data (excluding the target variable from train)
# to ensure all possible categories are encoded consistently across both datasets,
# then split them back. This avoids issues with differing columns if a category
# is present in one dataset but not the other.
all_data = pd.concat([train_df.drop('median_house_value', axis=1), test_df], ignore_index=True)

# Apply one-hot encoding to all object/category columns. This is a robust approach
# as it doesn't rely on explicitly listing column names and catches any column
# identified as 'object' by pandas.
all_data_encoded = pd.get_dummies(all_data, drop_first=False)

# Sanitize column names for XGBoost compatibility
# XGBoost does not allow feature names to contain [, ] or <.
# We replace these characters with underscores.
all_data_encoded.columns = [re.sub(r'[\[\]<]', '_', col) for col in all_data_encoded.columns]
# Optionally, further clean up multiple underscores or trailing/leading underscores
all_data_encoded.columns = [re.sub(r'_+', '_', col).strip('_') for col in all_data_encoded.columns]

# --- Robust Data Cleaning before model training ---
# This is a critical step to prevent C++ library crashes in models like XGBoost,
# which are sensitive to non-numeric types or non-finite values (NaN, Inf).

# Convert all columns to numeric, coercing non-numeric values to NaN.
# This handles cases where a column might have mixed types or unexpected string values.
for col in all_data_encoded.columns:
    all_data_encoded[col] = pd.to_numeric(all_data_encoded[col], errors='coerce')

# Fill any NaNs that might exist after coercion or from other columns not explicitly handled.
# Using 0 is a simple and generally effective strategy for tree-based models,
# especially for dummy variables or if NaNs represent 'absence'.
all_data_encoded.fillna(0, inplace=True)

# Replace infinite values (positive or negative) with NaN, then fill these NaNs.
# Infinite values can also cause C++ library crashes.
for col in all_data_encoded.columns:
    all_data_encoded[col].replace([np.inf, -np.inf], np.nan, inplace=True)
# Fill NaNs created from infs, using 0 for consistency.
all_data_encoded.fillna(0, inplace=True)

# Convert all feature columns to float32 for memory efficiency and consistent data types
# expected by C++ backend libraries.
all_data_encoded = all_data_encoded.astype(np.float32)

# Split the encoded and cleaned data back into training and testing sets
X = all_data_encoded.iloc[:len(train_df)]
X_test = all_data_encoded.iloc[len(train_df):]

# Define the target variable
y = train_df['median_house_value']

# Ensure target variable 'y' is also clean and numeric
y = pd.to_numeric(y, errors='coerce')
y.replace([np.inf, -np.inf], np.nan, inplace=True)
# Fill any NaNs in the target variable with its median (a robust choice)
y.fillna(y.median(), inplace=True)
y = y.astype(np.float32) # Convert target to float32

# --- Model Training: LightGBM ---
# Initialize LightGBM Regressor model with suppressed verbose output
lgbm = lgb.LGBMRegressor(objective='regression',
                         metric='rmse',
                         random_state=42,
                         n_jobs=-1,
                         verbose=-1) # Suppresses all verbose output

# Train the LightGBM model on the preprocessed training data
lgbm.fit(X, y)

# --- Model Training: XGBoost ---
# Initialize XGBoost Regressor model with suppressed verbose output
xgb_model_base = xgb.XGBRegressor(objective='reg:squarederror',
                                  eval_metric='rmse',
                                  random_state=42,
                                  verbosity=0) # Suppresses all verbose output

# Define the hyperparameter space for RandomizedSearchCV
param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 5, 10]
}

# Initialize RandomizedSearchCV with suppressed verbose output
xgb_random_search = RandomizedSearchCV(estimator=xgb_model_base,
                                       param_distributions=param_distributions,
                                       n_iter=50, # Number of parameter settings that are sampled
                                       scoring='neg_mean_squared_error', # Optimize for RMSE, using negative MSE
                                       cv=3, # Using 3-fold cross-validation
                                       verbose=0, # Suppress search progress output
                                       random_state=42,
                                       n_jobs=-1) # Use all available CPU cores

# Train RandomizedSearchCV to find the best XGBoost model
xgb_random_search.fit(X, y)

# Get the best estimator found by RandomizedSearchCV
xgb_model = xgb_random_search.best_estimator_

# Calculate final validation performance (RMSE) from the best XGBoost model's cross-validation score
# (neg_mean_squared_error is returned, so we take the negative and then sqrt)
final_validation_score = np.sqrt(-xgb_random_search.best_score_)

# --- Make Predictions on Test Data ---
y_pred_lgbm_test = lgbm.predict(X_test)
y_pred_xgb_test = xgb_model.predict(X_test)

# --- Model Ensembling for Test Predictions ---
# Simple averaging ensemble of LightGBM and XGBoost predictions
y_pred_ensemble_test = (y_pred_lgbm_test + y_pred_xgb_test) / 2

# --- Create Submission File ---
submission_df = pd.DataFrame({'median_house_value': y_pred_ensemble_test})

# Ensure the ./final directory exists
os.makedirs('./final', exist_ok=True)

# Save the submission file
submission_df.to_csv('./final/submission.csv', index=False)

print("Submission file 'submission.csv' created successfully in the './final' directory.")
print(f"Final Validation Performance: {final_validation_score}")