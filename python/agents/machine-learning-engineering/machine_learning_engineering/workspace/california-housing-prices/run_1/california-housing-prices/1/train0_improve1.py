

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import os

# Load datasets
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate target variable from features in the training data
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Handle missing values for 'total_bedrooms'
# Impute with the median from the training data to prevent data leakage.
# This median will be used for both the training features and the test set.
median_total_bedrooms = X['total_bedrooms'].median()

X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)
test_df['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Split the training data into training and validation sets
# A fixed random_state is used for reproducibility.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---

# 1. Initialize and Train LightGBM Regressor model (from base solution)
lgbm_model = lgb.LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
lgbm_model.fit(X_train, y_train)

# 2. Initialize and Train XGBoost Regressor model (from reference solution)
# Using a common objective for regression and a simple random_state for reproducibility.
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)

# --- Prediction and Ensemble ---


import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

# Make predictions on the validation set using LightGBM
y_pred_val_lgbm = lgbm_model.predict(X_val)

# Make predictions on the validation set using XGBoost
y_pred_val_xgb = xgb_model.predict(X_val)

# Ensure predictions and true values are numpy arrays for consistent operations
y_pred_val_lgbm = np.asarray(y_pred_val_lgbm)
y_pred_val_xgb = np.asarray(y_pred_val_xgb)
y_val = np.asarray(y_val) # Assuming y_val is already defined and available

# Define the objective function to minimize (e.g., Mean Squared Error for regression)
# This function calculates the error of the weighted ensemble predictions against the true values
def objective_function_mse(weights, predictions_1, predictions_2, true_values):
    # Calculate the ensemble predictions based on the given weights
    ensemble_predictions = weights[0] * predictions_1 + weights[1] * predictions_2
    # Return the Mean Squared Error as the error metric
    return mean_squared_error(true_values, ensemble_predictions)

# Initial guess for the weights (e.g., equal weights)
initial_weights = [0.5, 0.5]

# Define bounds for the weights: non-negative (0 to 1 as they sum to 1)
# Each weight must be between 0 and 1 inclusive.
bounds = ((0.0, 1.0), (0.0, 1.0))

# Define constraints: the sum of weights must be equal to 1.
# This ensures it's a true weighted average.
# {'type': 'eq'} means an equality constraint, where the function must return 0.
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# Perform the optimization to find the optimal weights
# 'SLSQP' method is suitable for problems with bounds and equality/inequality constraints.
optimization_result = minimize(
    objective_function_mse,
    initial_weights,
    args=(y_pred_val_lgbm, y_pred_val_xgb, y_val),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# Extract the optimal weights from the optimization result
optimal_weights = optimization_result.x
w_lgbm_opt, w_xgb_opt = optimal_weights

# Ensemble the predictions using the optimal non-negative weights
y_pred_val_ensemble = w_lgbm_opt * y_pred_val_lgbm + w_xgb_opt * y_pred_val_xgb


# --- Evaluation ---

# Evaluate the ensembled model using Root Mean Squared Error (RMSE)
rmse_val_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble))

# Print the final validation performance in the specified format
print(f"Final Validation Performance: {rmse_val_ensemble}")

# Optionally, for generating a submission file, one would predict on the test_df:
# test_predictions_lgbm = lgbm_model.predict(test_df)
# test_predictions_xgb = xgb_model.predict(test_df)
# test_predictions_ensemble = (test_predictions_lgbm + test_predictions_xgb) / 2
# submission_df = pd.DataFrame({'median_house_value': test_predictions_ensemble})
# submission_df.to_csv('submission.csv', index=False)

