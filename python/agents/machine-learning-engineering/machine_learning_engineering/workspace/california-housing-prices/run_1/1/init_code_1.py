
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

# Load datasets
try:
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the './input' directory.")
    exit() # This should not be reached based on instructions, but as a safeguard during development.

# Prepare the data
# Separate target variable from features
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Identify features for the test set
X_test_submission = test_df.copy()

# Handle missing values (e.g., in 'total_bedrooms')
# Impute with the median for simplicity as per instructions
for col in ['total_rooms', 'total_bedrooms', 'population', 'households']:
    if col in X.columns:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        if col in X_test_submission.columns:
            X_test_submission[col] = X_test_submission[col].replace([np.inf, -np.inf], np.nan)
            X_test_submission[col].fillna(median_val, inplace=True)

# Split the training data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LightGBM Regressor model
# Suppress verbose output using silent=True or verbose=-1
lgbm = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1, silent=True)

# Train the model
lgbm.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = lgbm.predict(X_val)

# Evaluate the model using Root Mean Squared Error (RMSE)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance
print(f'Final Validation Performance: {rmse_val}')

# Make predictions on the test set for submission (optional for problem, but good practice)
# y_pred_submission = lgbm.predict(X_test_submission)

# Create submission file (optional, based on task format description)
# submission_df = pd.DataFrame({'median_house_value': y_pred_submission})
# submission_df.to_csv('submission.csv', index=False)