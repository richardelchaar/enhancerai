
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
from sklearn.linear_model import LinearRegression # For the meta-learner

# Load the training data from the specified path
df_train = pd.read_csv('./input/train.csv')

# Handle missing values: Impute 'total_bedrooms' with its median.
df_train['total_bedrooms'].fillna(df_train['total_bedrooms'].median(), inplace=True)

# Define features (X) and target (y)
X = df_train.drop('median_house_value', axis=1)
y = df_train['median_house_value']

# Split the data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- LightGBM Model ---
lgbm_model = lgb.LGBMRegressor(objective='regression',
                               metric='rmse',
                               random_state=42,
                               verbose=-1)
lgbm_model.fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_val)

# --- XGBoost Model ---
xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                             eval_metric='rmse',
                             random_state=42,
                             verbosity=0)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_val)
