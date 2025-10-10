
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

# Separate features (X) and target (y)
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Handle missing values: Impute with the median
# This is a common and robust strategy for numerical features like 'total_bedrooms'
# Check if there are any missing values in the features DataFrame
if X.isnull().any().any():
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)

# Split the training data into training and validation sets
# A test_size of 0.2 means 20% of the data will be used for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost Regressor model
# Parameters are chosen as suggested in the model description
# verbosity=0 is used to suppress verbose output during training
model_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',  # Objective for regression tasks
    eval_metric='rmse',            # Evaluation metric is Root Mean Squared Error
    n_estimators=1000,             # Number of boosting rounds
    learning_rate=0.05,            # Step size shrinkage to prevent overfitting
    max_depth=6,                   # Maximum depth of a tree
    random_state=42,               # For reproducibility
    n_jobs=-1,                     # Use all available CPU cores
    verbosity=0                    # Suppress verbose output
)

# Train the model on the training data
model_xgb.fit(X_train, y_train)

# Make predictions on the validation set
# The bug was here: y_val was passed instead of X_val
y_pred_val = model_xgb.predict(X_val)

# Calculate the Root Mean Squared Error (RMSE) on the validation set
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance
print(f'Final Validation Performance: {rmse_val}')
