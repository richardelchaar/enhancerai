

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training data
try:
    train_df = pd.read_csv('./input/train.csv')
except FileNotFoundError:
    # Fallback for Kaggle environment where data might be in ../input/
    train_df = pd.read_csv('../input/train.csv')


# Define features (X) and target (y)
TARGET = 'median_house_value'
features = [col for col in train_df.columns if col != TARGET]
X = train_df[features]
y = train_df[TARGET]

# Split the data into training and validation sets BEFORE imputation to ensure no data leakage
# A reasonable split ratio like 80/20 is often used for validation.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values: Impute 'total_bedrooms' with the median
# Calculate median from the TRAINING features ONLY to prevent data leakage from the validation set.
median_total_bedrooms = X_train['total_bedrooms'].median()

# Impute 'total_bedrooms' in training data
X_train['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)
# Impute 'total_bedrooms' in validation data using the median from training data
X_val['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)


# --- Model 1: LightGBM Regressor ---
# Initialize the LightGBM Regressor model
# Using objective='regression' and metric='rmse' as suggested by the model description.
# A random_state is set for reproducibility.
lgbm = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42)

# Train the model
lgbm.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_lgbm = lgbm.predict(X_val)


# --- Model 2: XGBoost Regressor ---
# Initialize the XGBoost Regressor model
# Using objective='reg:squarederror' for regression and 'rmse' as evaluation metric.
# random_state is set for reproducibility. n_jobs=-1 utilizes all available CPU cores.
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_jobs=-1)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_xgb = xgb_model.predict(X_val)


# --- Ensemble the models ---
# Simple averaging of the predictions from LightGBM and XGBoost
y_pred_ensemble = (y_pred_lgbm + y_pred_xgb) / 2

# Calculate the Root Mean Squared Error (RMSE) on the validation set for the ensemble
rmse = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))

# Print the final validation performance
print(f"Final Validation Performance: {rmse}")

