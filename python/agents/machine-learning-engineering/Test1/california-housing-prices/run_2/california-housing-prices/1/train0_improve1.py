

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training data
try:
    df_train = pd.read_csv("./input/train.csv")
except FileNotFoundError:
    df_train = pd.read_csv("train.csv")

# Separate features (X) and target (y)
X = df_train.drop("median_house_value", axis=1)
y = df_train["median_house_value"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values in 'total_bedrooms' with the mean from the training set
# This prevents data leakage from the validation set
mean_total_bedrooms = X_train['total_bedrooms'].mean()
X_train['total_bedrooms'].fillna(mean_total_bedrooms, inplace=True)
X_val['total_bedrooms'].fillna(mean_total_bedrooms, inplace=True) # Apply the same mean to the validation set


import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
import numpy as np

# --- LightGBM Model Training and Prediction ---
# Initialize the LightGBM Regressor model
lgbm = lgb.LGBMRegressor(random_state=42)

# Train the LightGBM model
lgbm.fit(X_train, y_train)

# Make predictions on the validation set using LightGBM
y_pred_lgbm = lgbm.predict(X_val)

# --- XGBoost Model Training and Prediction ---
# Initialize the XGBoost Regressor model
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)

# Train the XGBoost model
model_xgb.fit(X_train, y_train)

# Make predictions on the validation set using XGBoost
y_pred_xgb = model_xgb.predict(X_val)

# --- Stacking Ensemble with Ridge Meta-Learner ---
# Prepare the training data for the meta-learner using predictions from base models on the validation set
X_meta_train = np.column_stack((y_pred_lgbm, y_pred_xgb))

# Initialize the meta-learner (Ridge Regressor)
meta_learner = Ridge(random_state=42)

# Train the meta-learner on the base model predictions and the actual validation targets
meta_learner.fit(X_meta_train, y_val)

# Make ensemble predictions using the trained meta-learner
# For the final ensemble prediction, if `X_test` were available, we would predict on its base model outputs.
# Here, we assume `y_pred_lgbm` and `y_pred_xgb` are the predictions we want to combine for the final output.
# The meta-learner directly gives us the combined prediction for the validation set.
y_pred_ensemble = meta_learner.predict(X_meta_train)


# Calculate Root Mean Squared Error (RMSE) for the ensembled predictions
rmse_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))

# Print the final validation performance of the ensemble
print(f"Final Validation Performance: {rmse_ensemble}")

