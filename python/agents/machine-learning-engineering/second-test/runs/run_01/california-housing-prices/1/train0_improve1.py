

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



import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# --- Stacking Ensemble Implementation ---

# Prepare the meta-features for training the meta-model
# X_meta_train will contain the predictions of base models on the validation set
X_meta_train = np.column_stack((y_val_pred_lgbm, y_val_pred_xgb))

# Initialize and train the meta-model (e.g., Ridge Regression)
# A small alpha (regularization strength) is often a good starting point for Ridge
meta_model = Ridge(alpha=1.0)
meta_model.fit(X_meta_train, y_val)

# Generate predictions from the meta-model on the validation set
y_val_pred_ensemble = meta_model.predict(X_meta_train)

# Evaluate the ensembled model using Root Mean Squared Error (RMSE) on the validation set
rmse_val_ensemble = np.sqrt(mean_squared_error(y_val, y_val_pred_ensemble))

# Print the validation performance of the ensemble
print(f"Final Validation Performance (Stacking): {rmse_val_ensemble}")

# Prepare the meta-features for making final test predictions
# X_meta_test will contain the predictions of base models on the test set
X_meta_test = np.column_stack((test_predictions_lgbm, test_predictions_xgb))

# Generate final test predictions using the trained meta-model
final_test_predictions = meta_model.predict(X_meta_test)


# Create the submission file
submission_df = pd.DataFrame({'median_house_value': final_test_predictions})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

