
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
train_df = pd.read_csv("./input/train.csv")

# Identify features and target
TARGET = 'median_house_value'
features = [col for col in train_df.columns if col != TARGET]

# Separate features (X) and target (y)
X = train_df[features]
y = train_df[TARGET]

# Handle missing values: Impute 'total_bedrooms' with the median
# Check if 'total_bedrooms' is a feature and has missing values
if 'total_bedrooms' in X.columns and X['total_bedrooms'].isnull().any():
    median_total_bedrooms = X['total_bedrooms'].median()
    X['total_bedrooms'] = X['total_bedrooms'].fillna(median_total_bedrooms)

# Split the data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LightGBM Regressor
# Parameters are chosen to align with the example and task description (regression, RMSE metric)
# verbose=-1 suppresses all verbose output during training
lgbm_params = {
    'objective': 'regression_l2',
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'random_state': 42,
    'n_jobs': -1,  # Use all available cores
    'verbose': -1,  # Suppress verbose output
    'boosting_type': 'gbdt',
}
model = lgb.LGBMRegressor(**lgbm_params)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Evaluate the model using Root Mean Squared Error (RMSE)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance
print(f'Final Validation Performance: {rmse_val}')

# Load the test data for submission
test_df = pd.read_csv("./input/test.csv")

# Ensure test data has the same preprocessing as training data
if 'total_bedrooms' in test_df.columns and test_df['total_bedrooms'].isnull().any():
    # Use the median from the training data for consistency
    test_df['total_bedrooms'] = test_df['total_bedrooms'].fillna(median_total_bedrooms)

# Align columns - this step is crucial if test_df might have different columns or order
# For this specific dataset, columns are expected to be consistent, but it's good practice.
X_test_processed = test_df[features]

# Predict on the test data
test_predictions = model.predict(X_test_processed)

# Create a submission DataFrame (optional, but good practice for Kaggle)
submission_df = pd.DataFrame({'median_house_value': test_predictions})

# Save the submission file (optional, but good practice for Kaggle)
# submission_df.to_csv('submission.csv', index=False)
