
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
train_df = pd.read_csv("./input/train.csv")

# Separate features and target
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Handle missing values, particularly in 'total_bedrooms'
# Use SimpleImputer to fill missing values with the median strategy
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Regressor model
# objective='reg:squarederror' is standard for regression with L2 loss
# eval_metric='rmse' for evaluation, random_state for reproducibility
# n_jobs=-1 uses all available CPU cores, verbosity=0 suppresses output
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse',
                             random_state=42, n_jobs=-1, verbosity=0)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = xgb_model.predict(X_val)

# Evaluate the model using Root Mean Squared Error (RMSE)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance
print(f"Final Validation Performance: {rmse_val}")
