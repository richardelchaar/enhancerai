
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training data
try:
    train_df = pd.read_csv("./input/train.csv")
except FileNotFoundError:
    train_df = pd.read_csv("train.csv")  # Fallback for local execution if input/ is not used

# Separate features and target variable
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Handle missing values: Impute 'total_bedrooms' with its median
# This is a common preprocessing step for this dataset
if X['total_bedrooms'].isnull().any():
    median_total_bedrooms = X['total_bedrooms'].median()
    X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Split the data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LightGBM Regressor model
# objective='regression_l2' is for Mean Squared Error, which is suitable for RMSE.
# metric='rmse' explicitly sets the evaluation metric.
# random_state for reproducibility.
# verbose=-1 suppresses verbose output during training.
lgbm_model = lgb.LGBMRegressor(objective='regression_l2', metric='rmse', random_state=42, verbose=-1)

# Train the model
lgbm_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = lgbm_model.predict(X_val)

# Evaluate the model using Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance
print(f'Final Validation Performance: {rmse}')
