
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

# Load the datasets
# Assume files exist in the current directory or './input'
# Check for './input' first, then current directory if not found (though problem states './input')
train_file_path = "./input/train.csv"
test_file_path = "./input/test.csv"

# Load train data
train_df = pd.read_csv(train_file_path)
# Load test data
test_df = pd.read_csv(test_file_path)

# Separate target variable from features
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Handle missing values
# The 'total_bedrooms' column often has missing values in this dataset.
# Impute missing values with the median of the column.
# Apply the same imputation to the test set using the training set's median.
if X['total_bedrooms'].isnull().any():
    median_total_bedrooms = X['total_bedrooms'].median()
    X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

if test_df['total_bedrooms'].isnull().any():
    # Use the median calculated from the training data for the test set
    test_df['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Regressor model
# 'objective': 'reg:squarederror' is recommended for regression tasks.
# 'eval_metric': 'rmse' is specified in the problem description.
# 'random_state' for reproducibility.
# 'verbosity': 0 suppresses verbose output.
model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, verbosity=0)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Calculate RMSE on the validation set
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance
print(f"Final Validation Performance: {rmse_val}")
