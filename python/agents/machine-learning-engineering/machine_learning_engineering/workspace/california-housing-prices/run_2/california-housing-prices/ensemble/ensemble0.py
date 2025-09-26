
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
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
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42)

print("Training LightGBM model...")
# Train the LightGBM model on the training set
lgbm_model.fit(X_train, y_train)
print("LightGBM model trained.")

print("Training XGBoost model...")
# Train the XGBoost model on the training set
xgb_model.fit(X_train, y_train)
print("XGBoost model trained.")

# --- Validation Predictions ---

# Make predictions on the validation set using LightGBM
y_pred_val_lgbm = lgbm_model.predict(X_val)

# Make predictions on the validation set using XGBoost
y_pred_val_xgb = xgb_model.predict(X_val)

# --- Systematic Search for Optimal Ensemble Weights ---
best_rmse = float('inf')
best_weight_lgbm = 0.0
best_weight_xgb = 0.0

# Iterate through possible weights for LightGBM
# Step size of 0.05 from 0.0 to 1.0 (inclusive)
weights_to_try = np.arange(0.0, 1.01, 0.05) 

print("\nSearching for optimal ensemble weights...")
for w_lgbm in weights_to_try:
    w_xgb = 1.0 - w_lgbm # XGBoost weight is complementary
    
    # Calculate ensembled predictions for the current weights
    current_y_pred_val_ensemble = (w_lgbm * y_pred_val_lgbm) + (w_xgb * y_pred_val_xgb)
    
    # Calculate RMSE for the current ensembled predictions
    current_rmse = np.sqrt(mean_squared_error(y_val, current_y_pred_val_ensemble))
    
    # Update best weights if current RMSE is lower
    if current_rmse < best_rmse:
        best_rmse = current_rmse
        best_weight_lgbm = w_lgbm
        best_weight_xgb = w_xgb

print(f"Optimal LightGBM Weight: {best_weight_lgbm:.2f}")
print(f"Optimal XGBoost Weight: {best_weight_xgb:.2f}")

# Print the final validation performance using the optimal weights
print(f'Final Validation Performance: {best_rmse}')

# --- Test Predictions and Ensemble with Optimal Weights ---

# Generate predictions for the test dataset using LightGBM
test_predictions_lgbm = lgbm_model.predict(X_test)

# Generate predictions for the test dataset using XGBoost
test_predictions_xgb = xgb_model.predict(X_test)

# Ensemble the test predictions using the optimal weights found
test_predictions_ensemble = (best_weight_lgbm * test_predictions_lgbm) + \
                            (best_weight_xgb * test_predictions_xgb)

# --- Output Test Predictions in Submission Format ---
print('median_house_value')
for pred_val in test_predictions_ensemble:
    print(f'{pred_val}')

