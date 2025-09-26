
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import os

# Load datasets
# The data is located in the './input' directory as per the instructions.
try:
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the './input' directory.")
    # Exit or handle error appropriately for a robust script.
    # For this specific task, we will assume files exist for now.
    raise

# Separate target variable
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Handle missing values: Impute 'total_bedrooms' with the median
# Calculate median from the training data only to prevent data leakage
median_total_bedrooms = X['total_bedrooms'].median()
X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)
test_df['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Split the training data into training and validation sets
# Use a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Regressor model
# For simplicity, no extensive hyper-parameter tuning or ensembling as per requirements.
# Using a common objective for regression and a simple random_state for reproducibility.
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = xgb_model.predict(X_val)

# Evaluate the model using Root Mean Squared Error (RMSE)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance
print(f"Final Validation Performance: {rmse_val}")

# Generate predictions for the test.csv dataset
test_predictions = xgb_model.predict(test_df)

# Create submission file
submission_df = pd.DataFrame({'median_house_value': test_predictions})
# Ensure the submission file format matches the requirement
# submission_df.to_csv('submission.csv', index=False)
