
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load the datasets
try:
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
except FileNotFoundError:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

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

# Initialize and train the LightGBM Regressor
# The objective is 'regression' for continuous targets
# The metric is 'rmse' which aligns with the task's evaluation metric
lgbm_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 1000,  # A reasonable number of estimators
    'learning_rate': 0.05,
    'feature_fraction': 0.8, # Fraction of features to consider per iteration
    'bagging_fraction': 0.8, # Fraction of data to sample per iteration
    'bagging_freq': 1,
    'lambda_l1': 0.1, # L1 regularization
    'lambda_l2': 0.1, # L2 regularization
    'num_leaves': 31, # Max number of leaves in one tree
    'verbose': -1, # Suppress verbose output
    'n_jobs': -1, # Use all available cores
    'seed': 42,
    'boosting_type': 'gbdt',
    'early_stopping_round': 50 # Early stopping to prevent overfitting
}

model = lgb.LGBMRegressor(**lgbm_params)

# Train the model
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='rmse',
          callbacks=[lgb.early_stopping(50, verbose=False)]) # Use lgb.early_stopping callback

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Evaluate the model using Root Mean Squared Error (RMSE)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance
print(f"Final Validation Performance: {rmse_val}")

# To generate predictions for the submission file (optional, but good practice for Kaggle)
# y_pred_test = model.predict(X_test_final)
# submission_df = pd.DataFrame({'median_house_value': y_pred_test})
# submission_df.to_csv('submission.csv', index=False)
