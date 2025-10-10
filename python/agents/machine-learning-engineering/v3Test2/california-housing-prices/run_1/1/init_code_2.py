
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
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load datasets
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Prepare the data
# Separate target variable from features
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Identify features for the test set
X_test_submission = test_df.copy()

# Handle missing values, specifically for 'total_bedrooms'
# Impute with the median for simplicity
# Identify numerical columns that might contain NaNs based on common patterns in such datasets
numerical_cols_to_impute = ['total_bedrooms'] # Often 'total_bedrooms' has NaNs

for col in numerical_cols_to_impute:
    if col in X.columns:
        # Calculate median only from the training features to prevent data leakage
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        # Apply the same median to the test set
        if col in X_test_submission.columns:
            X_test_submission[col].fillna(median_val, inplace=True)

# Split the training data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoost Regressor model
# Using parameters as described and suppressing verbose output
model = CatBoostRegressor(iterations=100,  # Number of boosting iterations (trees)
                          learning_rate=0.1, # Step size shrinkage
                          depth=6,         # Depth of trees
                          loss_function='RMSE', # Loss function to optimize
                          random_seed=42,    # For reproducibility
                          verbose=False,     # Suppress training output
                          allow_writing_files=False) # Prevent CatBoost from writing diagnostic files

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Evaluate the model using Root Mean Squared Error (RMSE)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance
print(f'Final Validation Performance: {rmse_val}')