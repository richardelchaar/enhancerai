
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
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# Handle missing values in 'total_bedrooms' by filling with the median
# This is a common practice for this dataset and helps prevent errors during training.
if 'total_bedrooms' in train_df.columns:
    train_df['total_bedrooms'].fillna(train_df['total_bedrooms'].median(), inplace=True)

# Define features (X) and target (y)
# The target variable for this task is 'median_house_value'.
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Split the data into a training set and a hold-out validation set.
# A 80/20 split (test_size=0.2) is a standard approach.
# random_state ensures reproducibility of the split for consistent results.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Regressor model
# objective='reg:squarederror' is specified for regression tasks using squared error.
# eval_metric='rmse' aligns with the competition's evaluation metric.
# random_state ensures reproducibility of the model's internal randomness.
# verbosity=0 suppresses all verbose output during training, as required.
xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                             eval_metric='rmse',
                             random_state=42,
                             verbosity=0) # Suppress verbose output

# Train the model on the training data
xgb_model.fit(X_train, y_train)

# Make predictions on the hold-out validation set
y_pred_val = xgb_model.predict(X_val)

# Calculate the Root Mean Squared Error (RMSE) on the validation set.
# This metric is used to evaluate the model's performance and is specified in the task description.
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance in the exact specified format.
print(f'Final Validation Performance: {rmse_val}')