
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Define paths to the datasets
train_path = './input/train.csv'
test_path = './input/test.csv'

# Load the training and testing datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Identify features and the target variable
# As per the model description, we will use all numerical features
FEATURES = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income'
]
TARGET = 'median_house_value'

# Prepare feature matrices and target vector
X = train_df[FEATURES].copy()
y = train_df[TARGET]
X_test = test_df[FEATURES].copy()

# Handle missing values in 'total_bedrooms' for both train and test sets
# Impute with the median value from the training data to prevent data leakage
median_bedrooms_train = X['total_bedrooms'].median()
X['total_bedrooms'].fillna(median_bedrooms_train, inplace=True)
X_test['total_bedrooms'].fillna(median_bedrooms_train, inplace=True)

# Split the training data into 80% for training and 20% for validation
# A random_state is used for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost Regressor model
# 'objective': 'reg:squarederror' is the recommended objective for regression tasks
# 'eval_metric': 'rmse' directly aligns with the competition's evaluation metric
# 'random_state' ensures reproducibility
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42)

# Train the XGBoost model on the training set
xgb_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = xgb_model.predict(X_val)

# Calculate the Root Mean Squared Error (RMSE) on the validation set
# This is the chosen evaluation metric as it aligns with the competition metric
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance
print(f'Final Validation Performance: {rmse_val}')

# Make predictions on the actual test dataset for submission
test_predictions = xgb_model.predict(X_test)

# Print predictions in the specified submission format
# The submission format requires a header followed by each prediction on a new line
print('median_house_value')
for pred_val in test_predictions:
    print(f'{pred_val}')

