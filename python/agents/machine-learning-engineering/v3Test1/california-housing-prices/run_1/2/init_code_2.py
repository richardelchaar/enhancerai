
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Load the training data
train_df = pd.read_csv('./input/train.csv')

# Separate target variable from features
# The target variable is 'median_house_value'
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Handle missing values for 'total_bedrooms'
# For robustness, we impute missing 'total_bedrooms' values with the median of that column.
# This is a common and effective strategy for handling sporadic missing numerical data in this dataset.
if 'total_bedrooms' in X.columns and X['total_bedrooms'].isnull().any():
    median_total_bedrooms = X['total_bedrooms'].median()
    X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Split the training data into training and validation sets
# A 80/20 split is used, and a fixed random_state ensures reproducibility of the split.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Regressor
# objective='reg:squarederror' is specified for regression problems, aiming to minimize squared error.
# eval_metric='rmse' explicitly sets the evaluation metric to RMSE, aligning with the task metric.
# random_state is set for reproducibility of the model training process.
# verbosity=0 is used to suppress all verbose output during training, as required.
model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, verbosity=0)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Evaluate the Model using Root Mean Squared Error (RMSE) on the validation set
# First, calculate Mean Squared Error, then take its square root to get RMSE.
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance in the specified format
print(f"Final Validation Performance: {rmse_val}")
