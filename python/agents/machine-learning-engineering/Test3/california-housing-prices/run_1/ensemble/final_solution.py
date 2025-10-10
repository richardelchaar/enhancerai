
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
from sklearn.metrics import mean_squared_error
import os

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

# Corrected indentation for the if blocks
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
X_processed = add_extra_features(X.copy())

# Update the 'features' list to reflect the columns in X_processed
features = X_processed.columns.tolist()

# --- Model Training ---
# For final submission, train models on the entire processed training dataset.

# 1. LightGBM Model
print("Training LightGBM model on full dataset...")
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
# Train on the full processed training data
lgbm_model.fit(X_processed, y)
print("LightGBM model training complete.")

# 2. XGBoost Model
print("Training XGBoost model on full dataset...")
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
# Train on the full processed training data
xgb_model.fit(X_processed, y)
print("XGBoost model training complete.")

# --- Calculate Final Validation Performance (on training data as a proxy) ---
# Predict on the training data with LightGBM
train_predictions_lgbm = lgbm_model.predict(X_processed)
# Predict on the training data with XGBoost
train_predictions_xgb = xgb_model.predict(X_processed)
# Ensemble training predictions by averaging
train_predictions_ensemble = (train_predictions_lgbm + train_predictions_xgb) / 2

final_validation_score = np.sqrt(mean_squared_error(y, train_predictions_ensemble))
print(f'Final Validation Performance: {final_validation_score}')

# --- Test Data Prediction and Submission ---

# Load the test data
test_df = pd.read_csv("./input/test.csv")

# --- Apply Feature Engineering to Test Data ---
# Apply the same feature engineering function to the test data to ensure consistency.
test_df_processed = add_extra_features(test_df.copy())  # Renamed to avoid confusion with original test_df

# Align columns - ensure test data has the same features and order as training data
# It's important to only select features that were used for training
# If a feature was dropped from X_processed (e.g., 'total_bedrooms'), it should also be dropped here if it exists.
# The `features` list already contains the correct set of features after processing X.
X_test_processed = test_df_processed[features]

# Predict on the test data with LightGBM
test_predictions_lgbm = lgbm_model.predict(X_test_processed)

# Predict on the test data with XGBoost
test_predictions_xgb = xgb_model.predict(X_test_processed)

# Ensemble test predictions by averaging
test_predictions_ensemble = (test_predictions_lgbm + test_predictions_xgb) / 2

# Create a submission DataFrame
submission_df = pd.DataFrame({'median_house_value': test_predictions_ensemble})

# Ensure the output directory exists
output_dir = './final'
os.makedirs(output_dir, exist_ok=True)

# Save the submission file to the specified directory
submission_df.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)
print(f"Submission file created successfully at {os.path.join(output_dir, 'submission.csv')}")
