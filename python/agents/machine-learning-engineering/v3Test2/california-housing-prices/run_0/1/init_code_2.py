
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
from catboost import CatBoostRegressor

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# Separate target variable and features
TARGET_COL = 'median_house_value'
X = train_df.drop(columns=[TARGET_COL])
y = train_df[TARGET_COL]

# Handle missing values in features - impute with median for numerical columns
# The 'total_bedrooms' column is known to have missing values in this dataset.
# We iterate through all columns to ensure robustness for any potential NaNs.
for col in X.columns:
    if X[col].isnull().any():
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())

# Split the data into training and validation sets
# Using a fixed random_state for reproducibility as required.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical features for CatBoost.
# Based on the dataset description, all input features are numerical.
# If there were categorical features, their column names or indices would be listed here.
categorical_features_indices = []

# Initialize CatBoost Regressor
# 'loss_function': 'RMSE' is specified as the metric for the competition.
# 'iterations' (equivalent to n_estimators) is set to 100 as a baseline, as per the model description example.
# 'random_state' is set for reproducibility.
# 'verbose=0' is used to suppress model output during training, as required.
model_catboost = CatBoostRegressor(
    loss_function='RMSE',
    iterations=100,
    random_state=42,
    verbose=0,
    allow_writing_files=False # Suppress writing model files to disk
)

# Train the model
# The `cat_features` parameter is provided, even if empty, to explicitly state
# that no features are to be treated as categorical by CatBoost for this dataset.
model_catboost.fit(X_train, y_train, cat_features=categorical_features_indices)

# Make predictions on the validation set
y_pred_val = model_catboost.predict(X_val)

# Calculate Root Mean Squared Error on the validation set
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance in the specified format.
print(f"Final Validation Performance: {rmse_val}")