
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import os

# Load the training data
train_df = pd.read_csv('./input/train.csv')

# Separate target variable from features
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Handle missing values in total_bedrooms for both training and test data
# Using median imputation as it's robust to outliers for numerical features
imputer = SimpleImputer(strategy='median')

# Fit imputer on training data and transform total_bedrooms
X['total_bedrooms'] = imputer.fit_transform(X[['total_bedrooms']])

# Split the data into training and validation sets
# A common split ratio is 80% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the LightGBM Regressor
# Using default parameters for simplicity, objective='regression_l2' for MSE
lgbm_model = lgb.LGBMRegressor(objective='regression_l2', random_state=42)
lgbm_model.fit(X_train, y_train)

# Initialize and train the XGBoost Regressor
# objective='reg:squarederror' is standard for regression with squared loss
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions on the validation set for both models
y_pred_lgbm = lgbm_model.predict(X_val)
y_pred_xgb = xgb_model.predict(X_val)

# Ensemble the predictions by averaging (simple ensembling technique)
y_pred_ensemble = (y_pred_lgbm + y_pred_xgb) / 2

# Evaluate the ensembled model using RMSE (Root Mean Squared Error)
rmse_val_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))

# Print the final validation performance
print(f'Final Validation Performance: {rmse_val_ensemble}')

# Prepare test data and make predictions for submission (optional but good practice)
# Load the test data
test_df = pd.read_csv('./input/test.csv')

# Apply the same imputation strategy to the test data
# It's crucial to use the imputer fitted on the training data
test_df['total_bedrooms'] = imputer.transform(test_df[['total_bedrooms']])

# Make predictions on the test set with both models
test_predictions_lgbm = lgbm_model.predict(test_df)
test_predictions_xgb = xgb_model.predict(test_df)

# Ensemble test predictions
test_predictions_ensemble = (test_predictions_lgbm + test_predictions_xgb) / 2

# Create submission file (commented out as per original instructions, focusing on validation performance)
# submission_df = pd.DataFrame({'median_house_value': test_predictions_ensemble})
# submission_df.to_csv('submission.csv', index=False)
# print("Submission file created: submission.csv")
