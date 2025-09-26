
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor # Import CatBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# --- Preprocessing ---
# Identify features and target
TARGET = 'median_house_value'
# All columns except the target are features
FEATURES = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income'
]

# Handle missing values in 'total_bedrooms'
# Impute with the median value from the training data
median_total_bedrooms = train_df['total_bedrooms'].median()
train_df['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Prepare data for models
X = train_df[FEATURES]
y = train_df[TARGET]

# Split training data into training and validation sets
# Using a fixed random_state for reproducibility and consistent validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Initialize and Train LightGBM Model (from Base Solution) ---
# Initialize LightGBM Regressor with parameters from the example description.
model_lgbm = lgb.LGBMRegressor(objective='regression',
                               n_estimators=100,            # Number of boosting rounds
                               learning_rate=0.1,           # Step size shrinkage
                               num_leaves=31,               # Max number of leaves in one tree
                               random_state=42,             # For reproducibility
                               n_jobs=-1)                   # Use all available CPU cores

# Train the LightGBM model on the training data
model_lgbm.fit(X_train, y_train)

# --- Initialize and Train XGBoost Model (from Base Solution) ---
# Initialize XGBoost Regressor with parameters suitable for a simple first solution.
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', # For regression tasks to minimize squared error
                             n_estimators=100,
                             learning_rate=0.1,
                             max_depth=5,
                             random_state=42,
                             n_jobs=-1)                   # Use all available CPU cores

# Train the XGBoost model on the training data
model_xgb.fit(X_train, y_train)

# --- Initialize and Train CatBoost Model (from Reference Solution) ---
# CatBoost can automatically handle categorical features if specified.
# In this specific dataset, all features selected are numerical,
# so the `cat_features` parameter will be an empty list or not provided.
categorical_features_indices = [] # No categorical features in the selected list

# Initialize CatBoost Regressor with parameters from the example description.
model_catboost = CatBoostRegressor(iterations=100,               # Number of boosting rounds
                                   learning_rate=0.1,             # Step size shrinkage
                                   depth=5,                       # Depth of the tree
                                   loss_function='RMSE',          # Objective for regression
                                   random_seed=42,                # For reproducibility
                                   verbose=0,                     # Suppress verbose output during training
                                   cat_features=categorical_features_indices) # Specify categorical features if any

# Train the CatBoost model on the training data
model_catboost.fit(X_train, y_train)


# --- Make Predictions and Ensemble ---
# Make predictions with the LightGBM model
y_pred_lgbm = model_lgbm.predict(X_val)

# Make predictions with the XGBoost model
y_pred_xgb = model_xgb.predict(X_val)

# Make predictions with the CatBoost model
y_pred_catboost = model_catboost.predict(X_val)

# Ensemble the predictions by taking a simple average of all three models
y_pred_ensemble = (y_pred_lgbm + y_pred_xgb + y_pred_catboost) / 3

# --- Evaluate Ensembled Model on Validation Set ---
# Calculate Root Mean Squared Error (RMSE) for the ensembled predictions
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))

# Print the final validation performance in the specified format
print(f'Final Validation Performance: {rmse_val}')
