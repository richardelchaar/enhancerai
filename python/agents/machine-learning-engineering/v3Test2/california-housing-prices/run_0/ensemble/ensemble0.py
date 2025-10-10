
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import xgboost as xgb

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# Separate target variable and features
TARGET_COL = 'median_house_value'
X = train_df.drop(columns=[TARGET_COL])
y = train_df[TARGET_COL]

# Identify numerical features for imputation (all features are numerical in this context)
numerical_features = X.select_dtypes(include=np.number).columns

# Impute missing values using median strategy (from Solution 2)
imputer = SimpleImputer(strategy='median')
X[numerical_features] = imputer.fit_transform(X[numerical_features])

# Split the data into training and validation sets
# Using a fixed random_state for reproducibility, consistent with both solutions
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- LightGBM Model from Solution 1 ---
# Define the parameter distribution for RandomizedSearchCV
param_dist_lgbm_s1 = {
    'n_estimators': randint(low=50, high=300),
    'learning_rate': uniform(loc=0.01, scale=0.2),
    'num_leaves': randint(low=20, high=60),
    'max_depth': randint(low=5, high=15),
    'min_child_samples': randint(low=20, high=100),
    'subsample': uniform(loc=0.6, scale=0.4),
    'colsample_bytree': uniform(loc=0.6, scale=0.4),
    'reg_alpha': uniform(loc=0, scale=0.1),
    'reg_lambda': uniform(loc=0, scale=0.1),
}

# Initialize a base LightGBM Regressor model
base_lgbm_s1 = lgb.LGBMRegressor(objective='regression',
                                 metric='rmse',
                                 random_state=42,
                                 verbose=-1)

# Initialize RandomizedSearchCV to optimize LightGBM hyperparameters
model_lgbm_s1 = RandomizedSearchCV(estimator=base_lgbm_s1,
                                   param_distributions=param_dist_lgbm_s1,
                                   n_iter=50,
                                   scoring='neg_root_mean_squared_error',
                                   cv=5,
                                   verbose=0, # Suppress verbose output during execution
                                   random_state=42,
                                   n_jobs=-1)

# Train the LightGBM model
model_lgbm_s1.fit(X_train, y_train)

# Make predictions on the validation set using LightGBM
y_pred_val_lgbm_s1 = model_lgbm_s1.predict(X_val)

# --- CatBoost Model from Solution 1 ---
# Identify categorical features for CatBoost (assuming none for this dataset as per S1)
categorical_features_indices = []

# Initialize CatBoost Regressor
model_catboost_s1 = CatBoostRegressor(
    loss_function='RMSE',
    iterations=100,
    random_state=42,
    verbose=0,
    allow_writing_files=False
)

# Train the CatBoost model
model_catboost_s1.fit(X_train, y_train, cat_features=categorical_features_indices)

# Make predictions on the validation set using CatBoost
y_pred_val_catboost_s1 = model_catboost_s1.predict(X_val)

# --- Ensemble Solution 1's predictions ---
y_pred_val_s1_ensemble = (y_pred_val_lgbm_s1 + y_pred_val_catboost_s1) / 2

# --- LightGBM Model from Solution 2 ---
# Initialize and train the LGBMRegressor model
model_lgbm_s2 = lgb.LGBMRegressor(objective='regression_l2', metric='rmse', random_state=42, verbose=-1, n_jobs=-1)
model_lgbm_s2.fit(X_train, y_train)

# Make predictions on the validation set with LGBM
y_pred_val_lgbm_s2 = model_lgbm_s2.predict(X_val)

# --- XGBoost Model from Solution 2 ---
# Initialize and train the XGBRegressor model
model_xgb_s2 = xgb.XGBRegressor(objective='reg:squarederror',
                             eval_metric='rmse',
                             random_state=42,
                             n_jobs=-1,
                             verbosity=0)
model_xgb_s2.fit(X_train, y_train)

# Make predictions on the validation set with XGBoost
y_pred_val_xgb_s2 = model_xgb_s2.predict(X_val)

# --- Ensemble Solution 2's predictions ---
y_pred_val_s2_ensemble = (y_pred_val_lgbm_s2 + y_pred_val_xgb_s2) / 2

# --- Meta-Ensemble the predictions from Solution 1 and Solution 2 ---
# Simple averaging ensemble of the two solution's final predictions
y_pred_final_ensemble = (y_pred_val_s1_ensemble + y_pred_val_s2_ensemble) / 2

# Calculate Root Mean Squared Error on the validation set for the final ensembled predictions
rmse_final_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_final_ensemble))

# Print the final validation performance of the meta-ensemble
print(f"Final Validation Performance: {rmse_final_ensemble}")