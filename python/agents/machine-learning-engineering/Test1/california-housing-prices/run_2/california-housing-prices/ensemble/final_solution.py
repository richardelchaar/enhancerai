

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression # Import for the meta-learner
import os

# Define paths to the datasets
train_path = './input/train.csv'
test_path = './input/test.csv'

# Load the training and testing datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Identify features and the target variable (from Solution 2)
FEATURES = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income'
]
TARGET = 'median_house_value'

# Prepare feature matrices and target vector (from Solution 2)
X = train_df[FEATURES].copy()
y = train_df[TARGET]
X_test = test_df[FEATURES].copy()

# Handle missing values in 'total_bedrooms' for both train and test sets
# Impute with the median value calculated from the training data to prevent data leakage (from Solution 2)
median_bedrooms = X['total_bedrooms'].median()
X['total_bedrooms'].fillna(median_bedrooms, inplace=True)
X_test['total_bedrooms'].fillna(median_bedrooms, inplace=True)

# Split the training data into 80% for training and 20% for validation (from Solution 2)
# A random_state is used for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Base Model Initialization and Training (from Solution 2) ---

# Initialize a LightGBM Regressor model
lgbm_model = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42)

# Initialize an XGBoost Regressor model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42)

# Train the LightGBM model on the training set
lgbm_model.fit(X_train, y_train)

# Train the XGBoost model on the training set
xgb_model.fit(X_train, y_train)

# --- Base Model Predictions ---

# Make predictions on the validation set using LightGBM
y_pred_val_lgbm = lgbm_model.predict(X_val)

# Make predictions on the validation set using XGBoost
y_pred_val_xgb = xgb_model.predict(X_val)

# Generate predictions for the test dataset using LightGBM
test_predictions_lgbm = lgbm_model.predict(X_test)

# Generate predictions for the test dataset using XGBoost
test_predictions_xgb = xgb_model.predict(X_test)

# --- Meta-learner Training (Stacking) ---

# Create a new dataset for the meta-learner using validation predictions of base models
# The features for the meta-learner are the predictions from the base models
X_meta_val = np.column_stack((y_pred_val_lgbm, y_pred_val_xgb))

# Initialize the meta-learner (Linear Regression as per the plan)
meta_learner = LinearRegression()

# Train the meta-learner on the validation set
# The meta-learner learns how to best combine the base model predictions to predict the true target
meta_learner.fit(X_meta_val, y_val)

# --- Ensemble Predictions ---

# Use the trained meta-learner to combine base model predictions on the validation set
y_pred_val_ensemble = meta_learner.predict(X_meta_val)

# Calculate the Root Mean Squared Error (RMSE) for the ensembled predictions on the validation set
rmse_val_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble))

# Print the final validation performance for the ensembled model
print(f'Final Validation Performance: {rmse_val_ensemble}')

# --- Test Predictions and Submission ---

# Create a new dataset for the meta-learner using test predictions of base models
X_meta_test = np.column_stack((test_predictions_lgbm, test_predictions_xgb))

# Use the trained meta-learner to combine base model predictions on the test set
test_predictions_ensemble = meta_learner.predict(X_meta_test)

# Create submission DataFrame
submission_df = pd.DataFrame({'median_house_value': test_predictions_ensemble})

# Ensure the ./final directory exists
os.makedirs('./final', exist_ok=True)

# Save the submission file
submission_df.to_csv('./final/submission.csv', index=False)

print("Submission file 'submission.csv' created successfully in the './final' directory.")

