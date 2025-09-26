
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training and test data
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')

# Define features (X) and target (y)
TARGET = 'median_house_value'
features = [col for col in train_df.columns if col != TARGET]

X = train_df[features]
y = train_df[TARGET]

# Handle missing values for 'total_bedrooms'
# Calculate median from the training features to prevent data leakage.
median_total_bedrooms = X['total_bedrooms'].median()

# Impute 'total_bedrooms' in training data
X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)
# Impute 'total_bedrooms' in test data using the median from training data
test_df['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Split the training data into training and validation sets
# A common split ratio like 80/20 is used for robust validation.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost Regressor model
# Using objective='reg:squarederror' for regression and 'rmse' as evaluation metric.
# random_state is set for reproducibility.
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_jobs=-1)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = xgb_model.predict(X_val)

# Calculate the Root Mean Squared Error (RMSE) on the validation set
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance
print(f"Final Validation Performance: {rmse}")

# Make predictions on the actual test set for submission
X_test_submission = test_df[features]
test_predictions = xgb_model.predict(X_test_submission)

# Create the submission DataFrame
submission_df = pd.DataFrame({'median_house_value': test_predictions})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)
