

import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the datasets
try:
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
except FileNotFoundError:
    # If running locally without input/ directory, assume files are in current directory
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

# Identify features and target
TARGET_COL = 'median_house_value'
FEATURES = [col for col in train_df.columns if col != TARGET_COL]

# Simple imputation for missing values in 'total_bedrooms'
# Calculate median from the training data to avoid data leakage
median_total_bedrooms_train = train_df['total_bedrooms'].median()
train_df['total_bedrooms'].fillna(median_total_bedrooms_train, inplace=True)
test_df['total_bedrooms'].fillna(median_total_bedrooms_train, inplace=True) # Use train median for test set

# Prepare data for models
X = train_df[FEATURES]
y = train_df[TARGET_COL]

# Split the training data into training and validation sets
# This helps evaluate the model's performance on unseen data before making final predictions
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- LightGBM Model Training ---
# Initialize the LightGBM Regressor model
lgbm_model = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, n_jobs=-1)

# Train the LightGBM model
lgbm_model.fit(X_train, y_train)

# Make predictions on the validation set with LightGBM
y_val_pred_lgbm = lgbm_model.predict(X_val)

# Make predictions on the actual test set with LightGBM
test_predictions_lgbm = lgbm_model.predict(test_df[FEATURES])


# --- XGBoost Model Training ---
# Initialize the XGBoost Regressor model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_jobs=-1)

# Train the XGBoost model
xgb_model.fit(X_train, y_train)

# Make predictions on the validation set with XGBoost
y_val_pred_xgb = xgb_model.predict(X_val)

# Make predictions on the actual test set with XGBoost
test_predictions_xgb = xgb_model.predict(test_df[FEATURES])



# --- Ensembling Predictions ---
# Implement weighted average ensemble for validation predictions

best_weight_lgbm = 0.5  # Initialize with simple average weight
min_rmse_val_ensemble = float('inf')
y_val_pred_ensemble = None # Initialize to store the best ensemble predictions

# Grid search for optimal weight for LightGBM, with XGBoost weight being (1 - weight_lgbm)
# Searching in steps of 0.01 from 0.0 to 1.0 (inclusive)
# This small grid search determines the optimal blend
weights_to_try = np.arange(0.0, 1.01, 0.01)

for w_lgbm in weights_to_try:
    w_xgb = 1.0 - w_lgbm
    current_y_val_pred_ensemble = (w_lgbm * y_val_pred_lgbm) + (w_xgb * y_val_pred_xgb)
    rmse_current = np.sqrt(mean_squared_error(y_val, current_y_val_pred_ensemble))

    if rmse_current < min_rmse_val_ensemble:
        min_rmse_val_ensemble = rmse_current
        best_weight_lgbm = w_lgbm
        y_val_pred_ensemble = current_y_val_pred_ensemble # Store the predictions for the best weight

best_weight_xgb = 1.0 - best_weight_lgbm

# Evaluate the ensembled model using Root Mean Squared Error (RMSE) on the validation set
# The rmse_val_ensemble is already the minimum found during the grid search
rmse_val_ensemble = min_rmse_val_ensemble

# Print the validation performance of the ensemble with optimal weights
print(f"Final Validation Performance (Weighted Ensemble): {rmse_val_ensemble:.6f}")
print(f"Optimal Weights - LightGBM: {best_weight_lgbm:.2f}, XGBoost: {best_weight_xgb:.2f}")

# Apply the optimal weights to the test predictions
final_test_predictions = (best_weight_lgbm * test_predictions_lgbm) + \
                         (best_weight_xgb * test_predictions_xgb)


# Create the submission file
submission_df = pd.DataFrame({'median_house_value': final_test_predictions})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

