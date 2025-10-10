import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# Identify features and target
TARGET = 'median_house_value'
features = [col for col in train_df.columns if col != TARGET]

# Separate features (X) and target (y)
X = train_df[features]
y = train_df[TARGET]

# --- Preprocessing (integrated from both solutions, ensuring consistency) ---

# Handle missing values: Impute 'total_bedrooms' with the median
# The base solution specifically handles 'total_bedrooms'.
# The reference solution handles all missing numerical features with their median.
# For this dataset, 'total_bedrooms' is the primary one, so we'll use the base's specific handling.
# Calculate median from the full training features before splitting for consistency.
median_total_bedrooms = None
if 'total_bedrooms' in X.columns:
    if X['total_bedrooms'].isnull().any():
        median_total_bedrooms = X['total_bedrooms'].median()
        X['total_bedrooms'] = X['total_bedrooms'].fillna(median_total_bedrooms)
    # If no missing values, median_total_bedrooms remains None, but it's not used later.

# Split the data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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
    'n_jobs': -1, # Use all available cores
    'verbose': -1, # Suppress verbose output
    'boosting_type': 'gbdt',
}
lgbm_model = lgb.LGBMRegressor(**lgbm_params)
lgbm_model.fit(X_train, y_train)
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

# Ensure test data has the same preprocessing as training data
if 'total_bedrooms' in test_df.columns:
    if test_df['total_bedrooms'].isnull().any():
        # Use the median from the training data for consistency
        # Ensure median_total_bedrooms was actually calculated (i.e., there were NaNs in train)
        if median_total_bedrooms is not None:
            test_df['total_bedrooms'] = test_df['total_bedrooms'].fillna(median_total_bedrooms)
        else:
            # Fallback if no NaNs were in training data but are in test data (unlikely but robust)
            test_df['total_bedrooms'] = test_df['total_bedrooms'].fillna(test_df['total_bedrooms'].median())

# Align columns - ensure test data has the same features and order as training data
X_test_processed = test_df[features]

# Predict on the test data with LightGBM
test_predictions_lgbm = lgbm_model.predict(X_test_processed)

# Predict on the test data with XGBoost
test_predictions_xgb = xgb_model.predict(X_test_processed)

# Ensemble test predictions
test_predictions_ensemble = (test_predictions_lgbm + test_predictions_xgb) / 2

# Create a submission DataFrame (optional, but good practice for Kaggle)
# submission_df = pd.DataFrame({'median_house_value': test_predictions_ensemble})

# Save the submission file (optional, but good practice for Kaggle)
# submission_df.to_csv('submission.csv', index=False)