
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

# --- Feature Engineering Function to Add ---


def add_extra_features(df):
    """
    Adds new ratio features and removes 'total_bedrooms' as per ablation study findings.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with new features and 'total_bedrooms' removed.
    """
    # Create ratio features, adding a small epsilon to denominators to avoid division by zero
    epsilon = 1e-6

# The following 'if' statements need to be correctly indented within the function.
    if 'total_rooms' in df.columns and 'population' in df.columns:
        df['rooms_per_person'] = df['total_rooms'] / (df['population'] + epsilon)

    if 'population' in df.columns and 'households' in df.columns:
        df['population_per_household'] = df['population'] / (df['households'] + epsilon)

    # Ablation study insight: Remove 'total_bedrooms' feature
    if 'total_bedrooms' in df.columns:
        df = df.drop('total_bedrooms', axis=1)

    return df


# Load the training data
train_df = pd.read_csv("./input/train.csv")

# Identify features and target
TARGET = 'median_house_value'

# Separate features (X) and target (y)
X = train_df.drop(TARGET, axis=1)
y = train_df[TARGET]

# --- Apply Feature Engineering to Training Data ---
# Apply the new feature engineering function to the training features.
# Using .copy() to ensure the original X DataFrame remains untouched if needed,
# although X_processed will be used for all subsequent steps.
X_processed = add_extra_features(X.copy())

# Update the 'features' list to reflect the columns in X_processed
# This list will now include the new ratio features and exclude 'total_bedrooms'.
features = X_processed.columns.tolist()

# --- Preprocessing (integrated from both solutions, ensuring consistency) ---
# The original script included imputation for 'total_bedrooms'.
# However, the 'add_extra_features' function explicitly removes 'total_bedrooms'.
# Therefore, the imputation block for 'total_bedrooms' is no longer necessary and has been removed.

# Split the data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# --- Model Training ---

# 1. LightGBM Model (from base solution)
print("Training LightGBM model...")
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
lgbm_model = lgb.LGBMRegressor(**lgbm_params)
lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
print("LightGBM model training complete.")

# 2. XGBoost Model (from reference solution)
print("Training XGBoost model...")
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',  # Objective for regression tasks
    eval_metric='rmse',            # Evaluation metric is Root Mean Squared Error
    n_estimators=1000,             # Number of boosting rounds
    learning_rate=0.05,            # Step size shrinkage to prevent overfitting
    max_depth=6,                   # Maximum depth of a tree
    random_state=42,               # For reproducibility
    n_jobs=-1,                     # Use all available CPU cores
    verbosity=0                    # Suppress verbose output
)
xgb_model.fit(X_train, y_train)
print("XGBoost model training complete.")

# --- Prediction and Ensemble ---

# Make predictions on the validation set with LightGBM
y_pred_val_lgbm = lgbm_model.predict(X_val)

# Make predictions on the validation set with XGBoost
y_pred_val_xgb = xgb_model.predict(X_val)

# Ensemble the predictions by averaging (simple average ensemble)
y_pred_val_ensemble = (y_pred_val_lgbm + y_pred_val_xgb) / 2

# --- Evaluation ---

# Evaluate the ensembled model using Root Mean Squared Error (RMSE)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble))

# Print the final validation performance
print(f'Final Validation Performance: {rmse_val}')

# --- Test Data Prediction (for submission, if needed) ---

# Load the test data
test_df = pd.read_csv("./input/test.csv")

# --- Apply Feature Engineering to Test Data ---
# Apply the same feature engineering function to the test data to ensure consistency.
test_df = add_extra_features(test_df.copy())

# Align columns - ensure test data has the same features and order as training data
# Use the 'features' list derived from the processed training data (X_processed)
# It's crucial that X_test_processed only contains columns that were in 'features'
# after processing the training data.
# In this specific case, add_extra_features ensures that the test_df will have the same
# feature set (including new ratios and exclusion of 'total_bedrooms') as X_processed.
X_test_processed = test_df[features]

# Predict on the test data with LightGBM
test_predictions_lgbm = lgbm_model.predict(X_test_processed)

# Predict on the test data with XGBoost
test_predictions_xgb = xgb_model.predict(X_test_processed)

# Ensemble test predictions
test_predictions_ensemble = (test_predictions_lgbm + test_predictions_xgb) / 2

# Create a submission DataFrame (optional, but good practice for Kaggle)
submission_df = pd.DataFrame({'median_house_value': test_predictions_ensemble})

# Save the submission file (optional, but good practice for Kaggle)
submission_df.to_csv('submission.csv', index=False)
