
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
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import os # Import os for creating directories

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# Separate target variable and features
TARGET_COL = 'median_house_value'
X = train_df.drop(columns=[TARGET_COL])
y = train_df[TARGET_COL]

# Store medians from the full training set for imputation.
# This ensures that missing values in the test set are imputed with values derived
# solely from the training data, preventing data leakage.
imputation_medians = {}
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        imputation_medians[col] = X[col].median()

# Handle missing values in training features - impute with stored medians.
X = X.fillna(imputation_medians)

# Split the data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- LightGBM Model Integration ---

# Define the parameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(low=50, high=300),        # Number of boosting rounds
    'learning_rate': uniform(loc=0.01, scale=0.2),    # Step size shrinkage
    'num_leaves': randint(low=20, high=60),           # Max number of leaves in one tree
    'max_depth': randint(low=5, high=15),             # Max tree depth
    'min_child_samples': randint(low=20, high=100),   # Minimum number of data in a child
    'subsample': uniform(loc=0.6, scale=0.4),         # Subsample ratio of the training instance
    'colsample_bytree': uniform(loc=0.6, scale=0.4),  # Subsample ratio of columns when constructing each tree
    'reg_alpha': uniform(loc=0, scale=0.1),           # L1 regularization term
    'reg_lambda': uniform(loc=0, scale=0.1),          # L2 regularization term
}

# Initialize a base LightGBM Regressor model
base_lgbm = lgb.LGBMRegressor(objective='regression',
                              metric='rmse',
                              random_state=42,
                              verbose=-1) # Suppress verbose output

# Initialize RandomizedSearchCV to optimize LightGBM hyperparameters
model_lgbm_search = RandomizedSearchCV(estimator=base_lgbm,
                                param_distributions=param_dist,
                                n_iter=50,             # Number of parameter settings that are sampled
                                scoring='neg_root_mean_squared_error', # Metric to optimize
                                cv=5,                  # 5-fold cross-validation
                                verbose=1,             # Verbosity level
                                random_state=42,
                                n_jobs=-1)             # Use all available cores

# Train the LightGBM model using RandomizedSearchCV on the training split
model_lgbm_search.fit(X_train, y_train)

# Extract the best estimator found by RandomizedSearchCV
model_lgbm = model_lgbm_search.best_estimator_

# Make predictions on the validation set using LightGBM
y_pred_val_lgbm = model_lgbm.predict(X_val)

# --- CatBoost Model Integration ---
# Identify categorical features for CatBoost.
# Based on the dataset description, all input features are numerical.
categorical_features_indices = []

# Initialize CatBoost Regressor
model_catboost = CatBoostRegressor(
    loss_function='RMSE',
    iterations=100, # Using a fixed number of iterations for simplicity
    random_state=42,
    verbose=0, # Suppress model output during training
    allow_writing_files=False # Suppress writing model files to disk
)

# Train the CatBoost model on the training split
model_catboost.fit(X_train, y_train, cat_features=categorical_features_indices)

# Make predictions on the validation set using CatBoost
y_pred_val_catboost = model_catboost.predict(X_val)

# --- Ensembling Predictions (Validation) ---
# A simple average ensemble of the two models
y_pred_val_ensemble = (y_pred_val_lgbm + y_pred_val_catboost) / 2

# Calculate Root Mean Squared Error on the validation set for the ensembled predictions
rmse_val_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble))

# Print the final validation performance of the ensemble
print(f"Final Validation Performance: {rmse_val_ensemble}")

# --- Final Model Training and Prediction for Submission ---

# Retrain the LightGBM model (with best parameters) on the full training data (X, y)
# The `model_lgbm` already holds the best parameters found by RandomizedSearchCV.
# Calling fit again will retrain it on the full dataset.
model_lgbm.fit(X, y)

# Retrain the CatBoost model on the full training data (X, y)
# Calling fit again on `model_catboost` will retrain it on the full dataset.
model_catboost.fit(X, y, cat_features=categorical_features_indices)

# Load the test data
test_df = pd.read_csv("./input/test.csv")

# Preprocess test data: impute missing values using medians derived from the training set
test_df = test_df.fillna(imputation_medians)

# Make predictions on the test set using the retrained models
test_predictions_lgbm = model_lgbm.predict(test_df)
test_predictions_catboost = model_catboost.predict(test_df)

# Ensemble the test predictions
final_predictions = (test_predictions_lgbm + test_predictions_catboost) / 2

# Create the submission DataFrame
submission_df = pd.DataFrame({'median_house_value': final_predictions})

# Ensure the ./final directory exists
os.makedirs('./final', exist_ok=True)

# Save the submission file to ./final/submission.csv
submission_df.to_csv('./final/submission.csv', index=False)