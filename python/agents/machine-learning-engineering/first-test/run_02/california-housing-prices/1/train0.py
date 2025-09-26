
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb # Import XGBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the datasets
train_df = pd.read_csv('input/train.csv')

# Separate target variable from features
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Handle missing values in 'total_bedrooms'
# Calculate median from the training features to prevent data leakage
median_total_bedrooms = X['total_bedrooms'].median()
X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Split the training data into training and validation sets
# A test_size of 0.2 means 20% of the data will be used for validation.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- LightGBM Model Training (from Base Solution) ---
# Initialize the LightGBM Regressor
model_lgbm = lgb.LGBMRegressor(random_state=42)

# Train the LightGBM model on the training set
model_lgbm.fit(X_train, y_train)

# Make predictions with LightGBM on the validation set
y_pred_val_lgbm = model_lgbm.predict(X_val)

# --- XGBoost Model Training (from Reference Solution) ---
# Initialize the XGBoost Regressor
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Train the XGBoost model on the training set
model_xgb.fit(X_train, y_train)

# Make predictions with XGBoost on the validation set
y_pred_val_xgb = model_xgb.predict(X_val)

# --- Ensemble Predictions ---
# Simple averaging ensemble of LightGBM and XGBoost predictions
y_pred_val_ensemble = (y_pred_val_lgbm + y_pred_val_xgb) / 2

# Evaluate the ensembled model using Root Mean Squared Error (RMSE)
rmse_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble))

# Print the final validation performance in the required format
print(f"Final Validation Performance: {rmse_ensemble}")

# For generating a submission file, you would typically train the models on the full
# training data (X, y) and then predict on the separate test.csv data.
# However, the task specifically asks only for the evaluation metric on a hold-out validation set.
# Therefore, the following code for submission is commented out as it's not required by the prompt.

# # Load the test data for final predictions
# test_df = pd.read_csv('input/test.csv')
#
# # Apply the same missing value imputation strategy to the test set
# test_df['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)
#
# # Retrain models on the entire training dataset for final submission predictions
# full_model_lgbm = lgb.LGBMRegressor(random_state=42)
# full_model_lgbm.fit(X, y)
#
# full_model_xgb = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
# full_model_xgb.fit(X, y)
#
# # Make predictions on the test set
# test_predictions_lgbm = full_model_lgbm.predict(test_df)
# test_predictions_xgb = full_model_xgb.predict(test_df)
#
# # Ensemble test predictions
# test_predictions_ensemble = (test_predictions_lgbm + test_predictions_xgb) / 2
#
# # Create submission DataFrame
# submission_df = pd.DataFrame({'median_house_value': test_predictions_ensemble})
#
# # Save submission file
# submission_df.to_csv('submission.csv', index=False)
