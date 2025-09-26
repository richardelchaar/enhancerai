
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the datasets
train_df = pd.read_csv('input/train.csv')

# Separate target variable from features
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Handle missing values in 'total_bedrooms'
# Calculate median from the training features to prevent data leakage
median_total_bedrooms = X['total_bedrooms'].median()
X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Split the training data into training and validation sets
# A test_size of 0.2 means 20% of the data will be used for validation.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LightGBM Regressor
# Using default 'regression' objective which optimizes for L2 loss (MSE), suitable for RMSE metric.
# random_state is set for reproducibility.
model = lgb.LGBMRegressor(random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Evaluate the model using Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance in the required format
print(f"Final Validation Performance: {rmse}")

# For generating a submission file, you would typically train the model on the full
# training data (X, y) and then predict on the separate test.csv data.
# However, the task specifically asks only for the evaluation metric on a hold-out validation set.
# Therefore, the following code for submission is commented out as it's not required by the prompt.

# # Load the test data for final predictions
# test_df = pd.read_csv('input/test.csv')
#
# # Apply the same missing value imputation strategy to the test set
# test_df['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)
#
# # Retrain the model on the entire training dataset for final submission predictions
# full_model = lgb.LGBMRegressor(random_state=42)
# full_model.fit(X, y)
#
# # Make predictions on the test set
# test_predictions = full_model.predict(test_df)
#
# # Create submission DataFrame
# submission_df = pd.DataFrame({'median_house_value': test_predictions})
#
# # Save submission file
# submission_df.to_csv('submission.csv', index=False)
