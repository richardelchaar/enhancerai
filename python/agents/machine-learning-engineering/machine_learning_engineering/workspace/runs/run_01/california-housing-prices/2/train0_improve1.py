
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

# Define file paths
TRAIN_FILE = "./input/train.csv"
TEST_FILE = "./input/test.csv"

# Load the datasets
# Assuming the files are present in the specified directory.
# If files are not found, pd.read_csv will raise a FileNotFoundError.
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# --- Data Preprocessing ---
# Identify features and target
TARGET = 'median_house_value'
# All columns in the training data except the target are features
FEATURES = [col for col in train_df.columns if col != TARGET]

# Handle missing values: 'total_bedrooms' is a common column with missing values
# Impute missing values with the median for relevant numerical columns
# This approach handles potential NaNs in both train and test sets
for col in ['total_rooms', 'total_bedrooms', 'population', 'households']:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(train_df[col].median())
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(test_df[col].median())

# Prepare feature matrices and target vector
train_labels = train_df[TARGET]
train_features = train_df[FEATURES]
test_features = test_df[FEATURES]

# Ensure column consistency between training and test features
# This step is crucial if there are differences in columns (e.g., categorical features handled differently)
# For this specific dataset, assuming only numerical features and direct match.
# If any feature was present in train but not test (or vice versa), this would add it with a default value.
# However, given the dataset structure, features should align.
missing_in_test_cols = set(train_features.columns) - set(test_features.columns)
for c in missing_in_test_cols:
    test_features[c] = 0  # Impute with 0 or a sensible default if feature is missing in test

missing_in_train_cols = set(test_features.columns) - set(train_features.columns)
for c in missing_in_train_cols:
    train_features[c] = 0 # Impute with 0 if feature is missing in train

# Align the order of columns in the test set to match the training set
test_features = test_features[train_features.columns]


# --- Model Training ---
# Split the training data to create a validation set for evaluation
# This helps estimate the model's performance on unseen data
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

# Initialize and train a RandomForestRegressor model
# RandomForest is a robust ensemble method suitable for regression tasks
# n_estimators: number of trees in the forest
# random_state: for reproducibility
# n_jobs=-1: uses all available processor cores
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# --- Validation ---
# Make predictions on the validation set
val_predictions = model.predict(X_val)

# Calculate Root Mean Squared Error (RMSE) on the validation set
rmse_val = np.sqrt(mean_squared_error(y_val, val_predictions))

# Print the final validation performance as required
print(f"Final Validation Performance: {rmse_val}")

# --- Prediction on Test Data ---
# Generate predictions for the actual test dataset
test_predictions = model.predict(test_features)

# --- Submission Format ---
# Output the predictions in the specified format:
# 'median_house_value' header followed by each prediction on a new line
print("median_house_value")
for pred_value in test_predictions:
    print(pred_value)

