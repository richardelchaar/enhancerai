
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import os

# Load the training data
# The path is './input/train.csv'
train_df = pd.read_csv('./input/train.csv')

# Separate target variable from features
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Handle missing values in total_bedrooms
# Using median imputation as it's robust to outliers for numerical features
imputer = SimpleImputer(strategy='median')
X['total_bedrooms'] = imputer.fit_transform(X[['total_bedrooms']])

# Split the data into training and validation sets
# A common split ratio is 80% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost Regressor
# objective='reg:squarederror' is standard for regression with squared loss
# Using default parameters for simplicity as requested, no hyper-parameter optimization
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Evaluate the model using RMSE (Root Mean Squared Error)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance as required
print(f'Final Validation Performance: {rmse_val}')

# Prepare test data and make predictions for submission
# Load the test data
test_df = pd.read_csv('./input/test.csv')

# Apply the same imputation strategy to the test data
# It's crucial to use the imputer fitted on the training data
test_df['total_bedrooms'] = imputer.transform(test_df[['total_bedrooms']])

# Make predictions on the test set
test_predictions = model.predict(test_df)

# The following lines are commented out as the task only explicitly asks for validation performance
# and not for generating a submission file, but are included to show the complete workflow.
# Create submission file as per Kaggle format
# submission_df = pd.DataFrame({'median_house_value': test_predictions})
# submission_df.to_csv('submission.csv', index=False)
# print("Submission file created: submission.csv")
