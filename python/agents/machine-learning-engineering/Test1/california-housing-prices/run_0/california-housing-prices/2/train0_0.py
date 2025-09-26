
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import os

# Load the training data
# The path is './input/train.csv' as per instructions
train_df = pd.read_csv('./input/train.csv')

# Separate target variable from features
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Handle missing values in total_bedrooms
# Using median imputation as it's robust to outliers
imputer = SimpleImputer(strategy='median')
X['total_bedrooms'] = imputer.fit_transform(X[['total_bedrooms']])

# Split the data into training and validation sets
# A common split ratio is 80% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the LightGBM Regressor
# Using default parameters for simplicity as requested, objective='regression_l2' for MSE
model = lgb.LGBMRegressor(objective='regression_l2', random_state=42)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Evaluate the model using RMSE
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance
print(f'Final Validation Performance: {rmse_val}')

# Prepare test data and make predictions for submission (optional but good practice)
# Load the test data
# The path is './input/test.csv' as per instructions
test_df = pd.read_csv('./input/test.csv')

# Apply the same imputation strategy to the test data
test_df['total_bedrooms'] = imputer.transform(test_df[['total_bedrooms']])

# Make predictions on the test set
test_predictions = model.predict(test_df)

# Create submission file (optional, but requested in Kaggle tasks)
# submission_df = pd.DataFrame({'median_house_value': test_predictions})
# submission_df.to_csv('submission.csv', index=False)
# print("Submission file created: submission.csv")
