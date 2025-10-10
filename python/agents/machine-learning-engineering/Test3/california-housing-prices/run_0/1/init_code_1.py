
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
except FileNotFoundError:
    print("Error: train.csv not found. Make sure it's in the ./input directory.")
    exit()

# Separate features and target
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Handle missing values
# Use SimpleImputer to fill missing values, particularly in 'total_bedrooms'
# A robust strategy is to fill with the median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LightGBM Regressor model
# objective='regression' for L2 loss, metric='rmse' for evaluation
# random_state for reproducibility, silent=True to suppress verbose output
lgbm_model = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, silent=True)

# Train the model
lgbm_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = lgbm_model.predict(X_val)

# Evaluate the model using Root Mean Squared Error (RMSE)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance
print(f"Final Validation Performance: {rmse_val}")

# --- For submission (optional, but good practice for a competition) ---
# Load test data and apply the same preprocessing steps
# try:
#     test_df = pd.read_csv("./input/test.csv")
# except FileNotFoundError:
#     print("Error: test.csv not found. Make sure it's in the ./input directory.")
#     exit()

# # Apply the same imputer fit on training data to the test data
# test_imputed = imputer.transform(test_df)
# X_test = pd.DataFrame(test_imputed, columns=test_df.columns)

# # Make predictions on the test set
# test_predictions = lgbm_model.predict(X_test)

# # Create submission file
# submission_df = pd.DataFrame({'median_house_value': test_predictions})
# submission_df.to_csv('submission.csv', index=False)
# print("Submission file created: submission.csv")
