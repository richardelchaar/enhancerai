

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb # Import XGBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Define paths to the datasets
train_path = './input/train.csv'
test_path = './input/test.csv'

# Load the training and testing datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Identify features and the target variable
FEATURES = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income'
]
TARGET = 'median_house_value'

# Prepare feature matrices and target vector
X = train_df[FEATURES].copy()
y = train_df[TARGET]
X_test = test_df[FEATURES].copy()

# Handle missing values in 'total_bedrooms' for both train and test sets
# Impute with the median value calculated from the training data to prevent data leakage
median_bedrooms = X['total_bedrooms'].median()
X['total_bedrooms'].fillna(median_bedrooms, inplace=True)
X_test['total_bedrooms'].fillna(median_bedrooms, inplace=True)

# Split the training data into 80% for training and 20% for validation
# A random_state is used for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Initialization and Training ---

# Initialize a LightGBM Regressor model
lgbm_model = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42)

# Initialize an XGBoost Regressor model
# 'objective': 'reg:squarederror' is recommended for regression tasks
# 'eval_metric': 'rmse' aligns with the evaluation metric
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42)

print("Training LightGBM model...")
# Train the LightGBM model on the training set
lgbm_model.fit(X_train, y_train)
print("LightGBM model trained.")

print("Training XGBoost model...")
# Train the XGBoost model on the training set
xgb_model.fit(X_train, y_train)
print("XGBoost model trained.")


# --- Validation Predictions and Ensemble ---

# Make predictions on the validation set using LightGBM
y_pred_val_lgbm = lgbm_model.predict(X_val)

# Make predictions on the validation set using XGBoost
y_pred_val_xgb = xgb_model.predict(X_val)

# Define weights for the ensemble (can be empirically tuned)
# Example weights: you might start with equal weights or tune based on individual model performance
# For instance, if LGBM performs slightly better on validation, its weight could be higher.
lgbm_weight = 0.5
xgb_weight = 0.5
# Ensure weights sum to 1
if abs(lgbm_weight + xgb_weight - 1.0) > 1e-6:
    print("Warning: Ensemble weights do not sum to 1. Adjusting to sum to 1 for calculation.")
    total_weight = lgbm_weight + xgb_weight
    if total_weight > 0:
        lgbm_weight /= total_weight
        xgb_weight /= total_weight
    else: # Fallback to equal weights if total is zero
        lgbm_weight = 0.5
        xgb_weight = 0.5

# Ensemble the predictions using a weighted average
y_pred_val_ensemble = (y_pred_val_lgbm * lgbm_weight) + (y_pred_val_xgb * xgb_weight)

# Calculate the Root Mean Squared Error (RMSE) for the ensembled predictions
rmse_val_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble))

# Print the final validation performance for the ensembled model
print(f'Final Validation Performance (Weighted Ensemble): {rmse_val_ensemble}')

# --- Test Predictions and Ensemble ---

# Generate predictions for the test dataset using LightGBM
test_predictions_lgbm = lgbm_model.predict(X_test)

# Generate predictions for the test dataset using XGBoost
test_predictions_xgb = xgb_model.predict(X_test)

# Ensemble the test predictions using the same weighted average
test_predictions_ensemble = (test_predictions_lgbm * lgbm_weight) + (test_predictions_xgb * xgb_weight)


# Print the ensembled predictions in the specified submission format
print('median_house_value')
for pred_val in test_predictions_ensemble:
    print(f'{pred_val}')

