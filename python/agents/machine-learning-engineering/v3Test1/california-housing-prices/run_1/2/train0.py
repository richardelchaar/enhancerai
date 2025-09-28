

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb # Import XGBoost for the reference solution model

# Load the training data
train_df = pd.read_csv('./input/train.csv')

# Separate target variable from features
# The target variable is 'median_house_value'
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Handle missing values
# For simplicity and robustness, we impute missing 'total_bedrooms' values with the median of that column.
# This is a common and effective strategy for handling sporadic missing numerical data.
if 'total_bedrooms' in X.columns and X['total_bedrooms'].isnull().any():
    median_total_bedrooms = X['total_bedrooms'].median()
    X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Split the training data into training and validation sets
# A 80/20 split is used, and a fixed random_state ensures reproducibility of the split.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---

# Initialize and train LightGBM Regressor (from the base solution)
# We use the 'regression' objective which minimizes L2 loss (MSE).
# The 'metric' parameter is set to 'rmse' for consistency with the evaluation metric.
# random_state is set for reproducibility of the model training.
# verbose=-1 is used to suppress all verbose output during training.
lgbm_model = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1)
lgbm_model.fit(X_train, y_train)

# Initialize and train XGBoost Regressor (from the reference solution)
# objective='reg:squarederror' is specified for regression problems.
# eval_metric='rmse' explicitly sets the evaluation metric to RMSE.
# random_state is set for reproducibility of the model training process.
# verbosity=0 is used to suppress all verbose output during training.
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, verbosity=0)
xgb_model.fit(X_train, y_train)

# --- Predictions ---

# Make predictions on the validation set using LightGBM
y_pred_lgbm = lgbm_model.predict(X_val)

# Make predictions on the validation set using XGBoost
y_pred_xgb = xgb_model.predict(X_val)

# --- Ensembling ---
# For ensembling, we will use a simple average of the predictions from both models.
# This is a straightforward and often effective way to combine models.
y_pred_ensemble = (y_pred_lgbm + y_pred_xgb) / 2

# --- Evaluation ---

# Evaluate the ensembled Model using Root Mean Squared Error (RMSE) on the validation set
# First, calculate Mean Squared Error, then take its square root.
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))

# Print the final validation performance in the specified format
print(f"Final Validation Performance: {rmse_val}")

