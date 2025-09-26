

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import os

# Load datasets
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate target variable from features in the training data
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Handle missing values for 'total_bedrooms'
# Impute with the median from the training data to prevent data leakage.
# This median will be used for both the training features and the test set.
median_total_bedrooms = X['total_bedrooms'].median()

X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)
test_df['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Split the training data into training and validation sets
# A fixed random_state is used for reproducibility.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---

# 1. Initialize and Train LightGBM Regressor model (from base solution)
lgbm_model = lgb.LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
lgbm_model.fit(X_train, y_train)

# 2. Initialize and Train XGBoost Regressor model (from reference solution)
# Using a common objective for regression and a simple random_state for reproducibility.
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)

# --- Prediction and Ensemble ---

# Make predictions on the validation set using LightGBM
y_pred_val_lgbm = lgbm_model.predict(X_val)

# Make predictions on the validation set using XGBoost
y_pred_val_xgb = xgb_model.predict(X_val)

# Ensemble the predictions
# A simple average ensemble is used as specified for simplicity.
y_pred_val_ensemble = (y_pred_val_lgbm + y_pred_val_xgb) / 2

# --- Evaluation ---

# Evaluate the ensembled model using Root Mean Squared Error (RMSE)
rmse_val_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble))

# Print the final validation performance in the specified format
print(f"Final Validation Performance: {rmse_val_ensemble}")

# Optionally, for generating a submission file, one would predict on the test_df:
# test_predictions_lgbm = lgbm_model.predict(test_df)
# test_predictions_xgb = xgb_model.predict(test_df)
# test_predictions_ensemble = (test_predictions_lgbm + test_predictions_xgb) / 2
# submission_df = pd.DataFrame({'median_house_value': test_predictions_ensemble})
# submission_df.to_csv('submission.csv', index=False)

