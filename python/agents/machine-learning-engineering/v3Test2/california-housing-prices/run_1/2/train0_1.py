
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

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# Handle missing values: Fill 'total_bedrooms' with the median
# This is a common and simple imputation strategy for this dataset,
# as 'total_bedrooms' is the only column typically found with missing values.
if 'total_bedrooms' in train_df.columns:
    train_df['total_bedrooms'].fillna(train_df['total_bedrooms'].median(), inplace=True)

# Define features (X) and target (y)
# The target variable is 'median_house_value'
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Split the training data into a training set and a hold-out validation set
# A 80/20 split is commonly used for this purpose.
# random_state ensures reproducibility of the split.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training: LightGBM ---
# Initialize LightGBM Regressor model
# objective='regression' is specified for continuous target prediction.
# metric='rmse' aligns with the competition's evaluation metric.
# random_state for reproducibility.
# n_jobs=-1 utilizes all available CPU cores for faster training.
# verbose=-1 suppresses all verbose output during training, as required.
lgbm = lgb.LGBMRegressor(objective='regression',
                         metric='rmse',
                         random_state=42,
                         n_jobs=-1,
                         verbose=-1)

# Train the LightGBM model on the training data
lgbm.fit(X_train, y_train)

# Make predictions on the hold-out validation set using LightGBM
y_pred_lgbm = lgbm.predict(X_val)

# --- Model Training: XGBoost ---
# Initialize XGBoost Regressor model
# objective='reg:squarederror' is specified for regression tasks using squared error.
# eval_metric='rmse' aligns with the competition's evaluation metric.
# random_state ensures reproducibility of the model's internal randomness.
# verbosity=0 suppresses all verbose output during training, as required.
xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                             eval_metric='rmse',
                             random_state=42,
                             verbosity=0) # Suppress verbose output

# Train the XGBoost model on the training data
xgb_model.fit(X_train, y_train)

# Make predictions on the hold-out validation set using XGBoost
y_pred_xgb = xgb_model.predict(X_val)

# --- Model Ensembling ---
# Simple averaging ensemble of LightGBM and XGBoost predictions
y_pred_ensemble = (y_pred_lgbm + y_pred_xgb) / 2

# Calculate the Root Mean Squared Error (RMSE) on the validation set for the ensemble
# This is the proposed evaluation metric, directly matching the competition metric.
rmse_val_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))

# Print the final validation performance in the specified format
print(f'Final Validation Performance: {rmse_val_ensemble}')