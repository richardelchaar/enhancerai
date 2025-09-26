

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Initialize and train the LightGBM Regressor (on X_train, y_train for tuning/validation)
# Using default parameters for simplicity, objective='regression_l2' for MSE
lgbm_model = lgb.LGBMRegressor(objective='regression_l2', random_state=42)
lgbm_model.fit(X_train, y_train)

# --- Hyperparameter Tuning for XGBoost ---
# Define the parameter grid for light hyperparameter tuning
param_grid_xgb = {
    'n_estimators': [50, 100, 200],  # Explore different numbers of boosting rounds
    'learning_rate': [0.05, 0.1, 0.2], # Explore different learning rates
    'max_depth': [3, 5, 7]           # Explore different tree depths
}

# Initialize the XGBoost Regressor for GridSearch
# objective='reg:squarederror' is standard for regression with squared loss
# use_label_encoder=False and eval_metric='rmse' are added to suppress warnings in newer XGBoost versions
xgb_base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, use_label_encoder=False, eval_metric='rmse')

# Perform GridSearchCV for hyperparameter tuning
# Scoring is set to 'neg_root_mean_squared_error' for RMSE optimization (GridSearchCV maximizes scores)
# cv=3 for a relatively quick, "light" tuning process
grid_search_xgb = GridSearchCV(
    estimator=xgb_base_model,
    param_grid=param_grid_xgb,
    scoring='neg_root_mean_squared_error',
    cv=3,
    n_jobs=-1,  # Use all available CPU cores
    verbose=0   # Suppress verbose output during search
)

grid_search_xgb.fit(X_train, y_train)

# Get the best XGBoost model found by GridSearchCV
xgb_model = grid_search_xgb.best_estimator_

# Make predictions on the validation set for both models
y_pred_lgbm = lgbm_model.predict(X_val)
y_pred_xgb = xgb_model.predict(X_val)

# --- Weighted Average Ensemble ---
# Calculate RMSE for each model on the validation set to determine appropriate weights
rmse_lgbm = np.sqrt(mean_squared_error(y_val, y_pred_lgbm))
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))

# Determine weights based on the inverse of RMSE (lower RMSE -> higher weight)
# An epsilon is added to avoid division by zero or extremely large weights for very small RMSEs
epsilon = 1e-6
weight_lgbm = 1 / (rmse_lgbm + epsilon)
weight_xgb = 1 / (rmse_xgb + epsilon)

# Normalize the weights so they sum to 1
total_weight = weight_lgbm + weight_xgb
normalized_weight_lgbm = weight_lgbm / total_weight
normalized_weight_xgb = weight_xgb / total_weight

# Ensemble the predictions using a weighted average
y_pred_ensemble = (normalized_weight_lgbm * y_pred_lgbm + normalized_weight_xgb * y_pred_xgb)

# Evaluate the ensembled model using RMSE (Root Mean Squared Error)
rmse_val_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))

# Print the final validation performance
print(f'Final Validation Performance: {rmse_val_ensemble}')

# --- Retrain models on the full training data (X, y) for final predictions ---
# It's good practice to train the final models on the entire available training data
# LightGBM
# The lgbm_model is re-fitted on the full dataset X and y
lgbm_model.fit(X, y) 

# XGBoost
# The xgb_model (which holds the best_estimator_ from GridSearchCV) is re-fitted on the full dataset X and y
xgb_model.fit(X, y) 

# Prepare test data and make predictions for submission
# Load the test data
test_df = pd.read_csv('./input/test.csv')

# Apply the same imputation strategy to the test data
# It's crucial to use the imputer fitted on the training data
test_df['total_bedrooms'] = imputer.transform(test_df[['total_bedrooms']])

# Make predictions on the test set with both models (now trained on full dataset)
test_predictions_lgbm = lgbm_model.predict(test_df)
test_predictions_xgb = xgb_model.predict(test_df)

# Ensemble test predictions using the validation-derived weights
test_predictions_ensemble = (normalized_weight_lgbm * test_predictions_lgbm + 
                             normalized_weight_xgb * test_predictions_xgb)

# Create submission file
os.makedirs('./final', exist_ok=True) # Ensure the ./final directory exists
submission_df = pd.DataFrame({'median_house_value': test_predictions_ensemble})
submission_df.to_csv('./final/submission.csv', index=False)
print("Submission file created: ./final/submission.csv")

