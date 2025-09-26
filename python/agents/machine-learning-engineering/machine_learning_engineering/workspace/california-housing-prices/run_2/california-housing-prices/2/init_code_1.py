
import pandas as pd
import numpy as np
import lightgbm as lgb
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
FEATURES = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income'
]
TARGET = 'median_house_value'

X = train_df[FEATURES].copy()
y = train_df[TARGET]
X_test = test_df[FEATURES].copy()

# Handle missing values in 'total_bedrooms'
# Impute with the median value calculated from the training data
median_bedrooms = X['total_bedrooms'].median()
X['total_bedrooms'].fillna(median_bedrooms, inplace=True)
X_test['total_bedrooms'].fillna(median_bedrooms, inplace=True)

# Split the training data into 80% for training and 20% for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a lightgbm.LGBMRegressor model
# with objective='regression' and metric='rmse', and a fixed random_state for reproducibility.
lgbm = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42)

# Train the LightGBM model on the training set
lgbm.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = lgbm.predict(X_val)

# Calculate the Root Mean Squared Error (RMSE) as the evaluation metric.
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the RMSE on the validation set
print(f'Final Validation Performance: {rmse_val}')

# Generate predictions for the test dataset
test_predictions = lgbm.predict(X_test)

# Print the predictions in the specified submission format
print('median_house_value')
for pred_val in test_predictions:
    print(f'{pred_val}')

