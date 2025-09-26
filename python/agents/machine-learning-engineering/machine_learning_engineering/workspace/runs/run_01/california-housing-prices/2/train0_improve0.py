
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import os

# Define input directory
input_dir = "./input"

# Load the datasets
train_file_path = os.path.join(input_dir, "train.csv")
test_file_path = os.path.join(input_dir, "test.csv")

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# Separate target variable from features
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Identify numerical features for imputation
numerical_cols = X.select_dtypes(include=np.number).columns

# Impute missing values with the median for numerical columns
# Store medians from the training set to apply to the test set
imputation_medians = {}
for col in numerical_cols:
    if X[col].isnull().any():
        median_val = X[col].median()
        imputation_medians[col] = median_val
        X[col].fillna(median_val, inplace=True)

# Apply imputation to the test set using medians from the training set
for col, median_val in imputation_medians.items():
    if col in test_df.columns:
        test_df[col].fillna(median_val, inplace=True)

# Ensure that the test set has the same columns and order as the training set features
# This is crucial if there are differences in columns (e.g., due to one-hot encoding or missing columns)
# In this specific dataset, the columns are expected to be the same, but this is good practice.
missing_in_test_cols = set(X.columns) - set(test_df.columns)
for c in missing_in_test_cols:
    test_df[c] = 0 # Fill missing columns in test with 0 (or another appropriate default)

# Reorder test_df columns to match X.columns
test_df = test_df[X.columns]

# Train a RandomForestRegressor model
# Using a random state for reproducibility and n_jobs for parallel processing
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Split the training data for validation to evaluate model performance
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model on the training split
model.fit(X_train, y_train)

# Make predictions on the validation set
val_predictions = model.predict(X_val)

# Calculate RMSE for validation performance
rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f"Final Validation Performance: {rmse}")

# Train the model on the full training data for final predictions
model.fit(X, y)

# Make predictions on the actual test set
final_predictions = model.predict(test_df)

# Prepare submission file
submission_df = pd.DataFrame({'median_house_value': final_predictions})
submission_df.to_csv("submission.csv", index=False)
