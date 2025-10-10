
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
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# --- 1. Data Loading and Preprocessing ---
# Load the training data. Assume file exists as per instructions.
train_df = pd.read_csv("./input/train.csv")

# Separate features and target
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Handle missing values using IterativeImputer
imputer = IterativeImputer(BayesianRidge(), max_iter=10, random_state=42)
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Model Training ---

# 2.1. LightGBM Model (from Base Solution)
lgbm_model = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, silent=True)
lgbm_model.fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_val)

# 2.2. XGBoost Model (from Reference Solution)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse',
                             random_state=42, n_jobs=-1, verbosity=0)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_val)

# --- 3. Ensembling ---
# Simple ensembling: average the predictions from both models
y_pred_ensemble = (y_pred_lgbm + y_pred_xgb) / 2

# --- 4. Evaluation of the Ensembled Model ---
rmse_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))

# Print the final validation performance
print(f"Final Validation Performance: {rmse_ensemble}")

# --- 5. Prepare for Submission ---
# Load the test data
test_df = pd.read_csv("./input/test.csv")

# Impute missing values in test data using the *fitted* imputer
test_imputed = imputer.transform(test_df)
test_df_imputed = pd.DataFrame(test_imputed, columns=test_df.columns)

# Generate predictions on the test set
test_pred_lgbm = lgbm_model.predict(test_df_imputed)
test_pred_xgb = xgb_model.predict(test_df_imputed)
test_pred_ensemble = (test_pred_lgbm + test_pred_xgb) / 2

# Create submission file
submission_df = pd.DataFrame({'median_house_value': test_pred_ensemble})
submission_df.to_csv("submission.csv", index=False)
