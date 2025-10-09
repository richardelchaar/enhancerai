
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
from sklearn.impute import SimpleImputer

# Load the training data
try:
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the './input/' directory.")
    exit()

# Separate features and target
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Identify numerical features for imputation
numerical_features = X.select_dtypes(include=np.number).columns

# Impute missing values
# Use median imputation, as it's robust to outliers.
# Fit the imputer on the training data, then transform both train and test.
imputer = SimpleImputer(strategy='median')
X[numerical_features] = imputer.fit_transform(X[numerical_features])
test_df[numerical_features] = imputer.transform(test_df[numerical_features])

# Split the training data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LGBMRegressor model
# objective='regression_l2' is for Mean Squared Error (MSE), which is appropriate for RMSE.
# metric='rmse' will monitor RMSE during training.
# random_state for reproducibility.
# verbose=-1 suppresses all verbose output during training.
# n_jobs=-1 utilizes all available CPU cores.
model_lgbm = lgb.LGBMRegressor(objective='regression_l2', metric='rmse', random_state=42, verbose=-1, n_jobs=-1)

# Train the model
model_lgbm.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model_lgbm.predict(X_val)

# Calculate RMSE on the validation set
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance
print(f"Final Validation Performance: {rmse_val}")

# To generate a submission file for the test_df (optional, not strictly required by prompt but good practice)
# Train on the full training data
# full_model_lgbm = lgb.LGBMRegressor(objective='regression_l2', metric='rmse', random_state=42, verbose=-1, n_jobs=-1)
# full_model_lgbm.fit(X, y)
# test_predictions = full_model_lgbm.predict(test_df)
# submission_df = pd.DataFrame({'median_house_value': test_predictions})
# submission_df.to_csv('submission.csv', index=False)