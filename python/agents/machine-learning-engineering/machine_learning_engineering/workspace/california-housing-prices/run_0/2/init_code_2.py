
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
from sklearn.impute import SimpleImputer

# Load the training data
try:
    train_df = pd.read_csv("./input/train.csv")
except FileNotFoundError:
    print("Error: train.csv not found. Please ensure it is in the './input/' directory.")
    # In a real Kaggle environment, you'd handle this more gracefully,
    # but for this specific problem, we'll assume the file exists.
    # For now, let's create an empty dataframe to prevent further errors for self-contained execution example
    train_df = pd.DataFrame()

# Separate features and target
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Identify numerical features for imputation
numerical_features = X.select_dtypes(include=np.number).columns

# Impute missing values
# Use median imputation, as it's robust to outliers and total_bedrooms often has few NaNs.
imputer = SimpleImputer(strategy='median')
X[numerical_features] = imputer.fit_transform(X[numerical_features])

# Split the training data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBRegressor model
# objective='reg:squarederror' corresponds to mean squared error, which is used for RMSE.
# eval_metric='rmse' monitors RMSE during training (though we're suppressing verbose output).
# random_state for reproducibility.
# n_jobs=-1 utilizes all available CPU cores.
# verbosity=0 suppresses all verbose output during training.
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', 
                             eval_metric='rmse', 
                             random_state=42, 
                             n_jobs=-1, 
                             verbosity=0) # Suppress verbose output

# Train the model
model_xgb.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model_xgb.predict(X_val)

# Calculate RMSE on the validation set
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance
print(f"Final Validation Performance: {rmse_val}")