
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
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Load the datasets
# Assuming train.csv and test.csv are located in the ./input directory
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate target variable from features
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Handle missing values
# The 'total_bedrooms' column is known to have a small number of missing values
# in this dataset. We impute them with the median value calculated from the training data.
median_total_bedrooms_train = X['total_bedrooms'].median()
X['total_bedrooms'] = X['total_bedrooms'].fillna(median_total_bedrooms_train)
test_df['total_bedrooms'] = test_df['total_bedrooms'].fillna(median_total_bedrooms_train)

# Split the training data into a training set and a hold-out validation set
# This ensures we evaluate the model on unseen data before making final predictions.
# A fixed random_state is used for reproducibility.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training for Validation ---

# 1. Initialize and Train LightGBM Regressor (from Base Solution)
lgbm_model_val = lgb.LGBMRegressor(objective='regression',
                               metric='rmse',
                               random_state=42,
                               verbose=-1,
                               n_jobs=-1)

print("Training LightGBM model for validation...")
lgbm_model_val.fit(X_train, y_train)
print("LightGBM model training for validation complete.")

# 2. Initialize and Train CatBoost Regressor (from Reference Solution)
catboost_model_val = CatBoostRegressor(loss_function='RMSE',
                                   random_seed=42,
                                   verbose=0,  # Suppress training output
                                   iterations=100,  # Number of boosting iterations
                                   learning_rate=0.1) # Step size shrinkage

print("Training CatBoost model for validation...")
catboost_model_val.fit(X_train, y_train)
print("CatBoost model training for validation complete.")

# --- Make Predictions on Validation Set ---

# Make predictions on the hold-out validation set using both models
y_pred_lgbm_val = lgbm_model_val.predict(X_val)
y_pred_catboost_val = catboost_model_val.predict(X_val)

# --- Ensemble the Models for Validation ---

# Simple averaging ensemble
y_pred_ensemble_val = (y_pred_lgbm_val + y_pred_catboost_val) / 2

# Calculate the Root Mean Squared Error (RMSE) on the ensembled validation predictions
rmse_ensemble_val = np.sqrt(mean_squared_error(y_val, y_pred_ensemble_val))

# Print the final validation performance in the specified format
print(f"Final Validation Performance: {rmse_ensemble_val}")


# --- Model Training for Final Submission (using full training data) ---

# 1. Initialize and Train LightGBM Regressor on full dataset
lgbm_model_final = lgb.LGBMRegressor(objective='regression',
                                     metric='rmse',
                                     random_state=42,
                                     verbose=-1,
                                     n_jobs=-1)

print("\nTraining LightGBM model on full dataset for final submission...")
lgbm_model_final.fit(X, y) # Train on the entire training dataset
print("LightGBM model training for final submission complete.")

# 2. Initialize and Train CatBoost Regressor on full dataset
catboost_model_final = CatBoostRegressor(loss_function='RMSE',
                                         random_seed=42,
                                         verbose=0,
                                         iterations=100,
                                         learning_rate=0.1)

print("Training CatBoost model on full dataset for final submission...")
catboost_model_final.fit(X, y) # Train on the entire training dataset
print("CatBoost model training for final submission complete.")

# --- Make Predictions on Test Set ---

# Make predictions on the test set using both final models
y_pred_lgbm_test = lgbm_model_final.predict(test_df)
y_pred_catboost_test = catboost_model_final.predict(test_df)

# --- Ensemble the Models for Test Set ---

# Simple averaging ensemble for test predictions
y_pred_ensemble_test = (y_pred_lgbm_test + y_pred_catboost_test) / 2

# Ensure predictions are non-negative, as median_house_value cannot be negative
y_pred_ensemble_test[y_pred_ensemble_test < 0] = 0

# --- Create Submission File ---

# Create the submission DataFrame
submission_df = pd.DataFrame({'median_house_value': y_pred_ensemble_test})

# Create the ./final directory if it doesn't exist
os.makedirs("./final", exist_ok=True)

# Save the submission file
submission_file_path = "./final/submission.csv"
submission_df.to_csv(submission_file_path, index=False)

print(f"\nSubmission file created successfully at: {submission_file_path}")

