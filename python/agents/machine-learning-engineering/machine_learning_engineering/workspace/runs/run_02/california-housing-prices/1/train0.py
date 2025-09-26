
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load the datasets
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Define features (X) and target (y)
TARGET = 'median_house_value'
FEATURES = [col for col in train_df.columns if col != TARGET]

X = train_df[FEATURES]
y = train_df[TARGET]
X_test_final = test_df[FEATURES] # Features for the final test set prediction

# Handle missing values
# Use SimpleImputer to fill missing values, typically for 'total_bedrooms'
imputer = SimpleImputer(strategy='median')

# Fit imputer on training data and transform both training and test data
X[FEATURES] = imputer.fit_transform(X[FEATURES])
X_test_final[FEATURES] = imputer.transform(X_test_final[FEATURES]) # Apply to final test set as well

# Split the training data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- LightGBM Model Training (from Base Solution) ---
# Initialize and train the LightGBM Regressor
lgbm_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
    'early_stopping_round': 50
}

lgbm_model = lgb.LGBMRegressor(**lgbm_params)

# Train the LightGBM model
lgbm_model.fit(X_train, y_train,
               eval_set=[(X_val, y_val)],
               eval_metric='rmse',
               callbacks=[lgb.early_stopping(50, verbose=False)])

# Make predictions on the validation set using LightGBM
y_pred_val_lgbm = lgbm_model.predict(X_val)

# --- CatBoost Model Training (from Reference Solution) ---
# Initialize and train the CatBoost Regressor
cat_model = CatBoostRegressor(
    iterations=200,          # A reasonable number of boosting iterations
    learning_rate=0.1,       # Default learning rate
    depth=6,                 # Default tree depth
    loss_function='RMSE',    # Objective function for regression
    random_seed=42,          # For reproducibility
    verbose=0                # Suppress training output
)

# Train the CatBoost model
cat_model.fit(X_train, y_train)

# Make predictions on the validation set using CatBoost
y_pred_val_cat = cat_model.predict(X_val)

# --- Ensemble Predictions ---
# Simple averaging ensemble
y_pred_val_ensemble = (y_pred_val_lgbm + y_pred_val_cat) / 2

# Evaluate the ensembled model using Root Mean Squared Error (RMSE)
rmse_val_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble))

# Print the final validation performance of the ensembled model
print(f"Final Validation Performance: {rmse_val_ensemble}")

# To generate predictions for the submission file (optional)
# y_pred_test_lgbm = lgbm_model.predict(X_test_final)
# y_pred_test_cat = cat_model.predict(X_test_final)
# y_pred_test_ensemble = (y_pred_test_lgbm + y_pred_test_cat) / 2
# submission_df = pd.DataFrame({'median_house_value': y_pred_test_ensemble})
# submission_df.to_csv('submission.csv', index=False)
