
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the datasets
train_df = pd.read_csv('input/train.csv')

# Separate target variable from features
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Handle missing values in 'total_bedrooms'
# Calculate median from the training features to prevent data leakage from the validation set
median_total_bedrooms = X['total_bedrooms'].median()
X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Split the training data into training and validation sets
# A test_size of 0.2 means 20% of the data will be used for validation.
# random_state is set for reproducibility.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost Regressor
# objective='reg:squarederror' is the standard objective for regression with squared loss,
# which is appropriate for the RMSE metric.
# random_state is set for reproducibility.
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Evaluate the model using Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance in the required format
print(f"Final Validation Performance: {rmse}")
