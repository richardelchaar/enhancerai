import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training data
try:
    train_df = pd.read_csv("./input/train.csv")
except FileNotFoundError:
    print("Ensure train.csv is in the ./input directory.")
    exit()

# Separate target variable and features
TARGET_COL = 'median_house_value'
X = train_df.drop(columns=[TARGET_COL])
y = train_df[TARGET_COL]

# Handle missing values in features - impute with median for numerical columns
# 'total_bedrooms' is a common column with missing values in this dataset type.
for col in X.columns:
    if X[col].isnull().any():
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())

# Split the data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LightGBM Regressor model
# objective='regression' for continuous target variable
# metric='rmse' for Root Mean Squared Error, aligning with the competition metric
# n_estimators=100 as a simple baseline, as per the model description's example.
# verbose=-1 to suppress console output during training.
model_lgbm = lgb.LGBMRegressor(objective='regression',
                               metric='rmse',
                               n_estimators=100,
                               random_state=42,
                               verbose=-1) # Suppress verbose output

# Train the model
model_lgbm.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model_lgbm.predict(X_val)

# Calculate Root Mean Squared Error on the validation set
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance
print(f"Final Validation Performance: {rmse_val}")

# (Optional: If submission file were required, it would be generated here)
# try:
#     test_df = pd.read_csv("./input/test.csv")
# except FileNotFoundError:
#     print("Ensure test.csv is in the ./input directory.")
#     exit()

# # Handle missing values in test data similarly to training data
# for col in test_df.columns:
#     if test_df[col].isnull().any():
#         if pd.api.types.is_numeric_dtype(test_df[col]):
#             test_df[col] = test_df[col].fillna(test_df[col].median())

# # Make predictions on the test set
# test_predictions = model_lgbm.predict(test_df)

# # Create submission DataFrame
# submission_df = pd.DataFrame({'median_house_value': test_predictions})

# # Save submission file
# submission_df.to_csv('submission.csv', index=False)
# print("Submission file created successfully!")