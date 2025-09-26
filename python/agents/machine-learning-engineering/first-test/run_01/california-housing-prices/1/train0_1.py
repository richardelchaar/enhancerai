

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load datasets
# All data is assumed to be in the './input' directory as per task description.
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Preprocessing: Handle missing values
# The 'total_bedrooms' column often contains a few missing values in this dataset.
# Filling with the median is a robust strategy for tree-based models and a simple approach.
train_df['total_bedrooms'].fillna(train_df['total_bedrooms'].median(), inplace=True)
test_df['total_bedrooms'].fillna(test_df['total_bedrooms'].median(), inplace=True)

# Define features (X) and target (y)
# Features are all columns except 'median_house_value' in the training set.
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income']
target = 'median_house_value'

X = train_df[features]
y = train_df[target]

# Split training data into a training set and a hold-out validation set
# A 80/20 split is commonly used. random_state ensures reproducibility.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Initialization ---

# Initialize LightGBM Regressor model (from base solution)
# objective='regression' specifies the task type.
# metric='rmse' sets the evaluation metric for training monitoring (though we calculate explicitly).
# random_state for reproducibility of the model's internal randomness.
lgbm_model = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42)

# Initialize XGBoost Regressor model (from reference solution)
# objective='reg:squarederror' is standard for regression tasks.
# eval_metric='rmse' explicitly sets RMSE as the evaluation metric for early stopping (though not used here).
# random_state for reproducibility of the model's internal randomness.
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_jobs=-1)

# --- Model Training ---

# Train the LightGBM model on the training data
print("Training LightGBM model...")
lgbm_model.fit(X_train, y_train)
print("LightGBM model training complete.")

# Train the XGBoost model on the training data
print("Training XGBoost model...")
xgb_model.fit(X_train, y_train)
print("XGBoost model training complete.")

# --- Validation Predictions and Ensemble ---

# Make predictions from LightGBM on the hold-out validation set
y_pred_val_lgbm = lgbm_model.predict(X_val)

# Make predictions from XGBoost on the hold-out validation set
y_pred_val_xgb = xgb_model.predict(X_val)

# Ensemble the validation predictions (simple average)
y_pred_val_ensemble = (y_pred_val_lgbm + y_pred_val_xgb) / 2

# Evaluate the ensembled model on the validation set using Root Mean Squared Error (RMSE)
rmse_val_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble))

# Print the final validation performance as required
print(f"Final Validation Performance: {rmse_val_ensemble}")

# --- Test Predictions and Submission ---

# Prepare test data for final predictions
X_test = test_df[features]

# Generate predictions from LightGBM on the actual test set
test_predictions_lgbm = lgbm_model.predict(X_test)

# Generate predictions from XGBoost on the actual test set
test_predictions_xgb = xgb_model.predict(X_test)

# Ensemble the test predictions (simple average)
test_predictions_ensemble = (test_predictions_lgbm + test_predictions_xgb) / 2

# Create the submission file in the specified format
submission_df = pd.DataFrame({'median_house_value': test_predictions_ensemble})
submission_df.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully.")

