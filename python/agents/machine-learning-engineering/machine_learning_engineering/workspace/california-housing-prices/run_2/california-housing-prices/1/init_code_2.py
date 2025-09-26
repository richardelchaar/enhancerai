
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training data
# Using a relative path that works both in a typical Kaggle environment and locally
try:
    df_train = pd.read_csv("./input/train.csv")
except FileNotFoundError:
    df_train = pd.read_csv("train.csv") # Fallback for local testing if input/ is not used

# Separate features (X) and target (y)
X = df_train.drop("median_house_value", axis=1)
y = df_train["median_house_value"]

# Split the data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values in 'total_bedrooms'
# Calculate the mean from the training set to prevent data leakage
mean_total_bedrooms_train = X_train['total_bedrooms'].mean()

# Apply imputation to both training and validation sets
X_train['total_bedrooms'].fillna(mean_total_bedrooms_train, inplace=True)
X_val['total_bedrooms'].fillna(mean_total_bedrooms_train, inplace=True) # Use train mean for validation set

# Initialize the XGBoost Regressor model
# objective='reg:squarederror' is standard for regression tasks (minimizes MSE)
# n_estimators=100 is a reasonable default for a simple model without hyperparameter tuning
# random_state for reproducibility
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)

# Train the model
model_xgb.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model_xgb.predict(X_val)

# Evaluate the model using Root Mean Squared Error (RMSE)
# This is the specified metric for the task
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

# Print the final validation performance in the required format
print(f"Final Validation Performance: {rmse}")

