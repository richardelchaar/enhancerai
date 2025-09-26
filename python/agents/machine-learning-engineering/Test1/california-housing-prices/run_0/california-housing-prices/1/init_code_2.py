
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the datasets
train_df = pd.read_csv('./input/train.csv')

# Separate target variable
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Preprocessing: Handle missing values
# Impute missing 'total_bedrooms' with the median
# Using median as it's more robust to outliers
X['total_bedrooms'].fillna(X['total_bedrooms'].median(), inplace=True)

# Split the training data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Regressor
# objective='reg:squarederror' is the recommended objective for regression tasks to minimize squared error.
# Setting random_state for reproducibility.
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = xgb_model.predict(X_val)

# Evaluate the model using RMSE
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the validation performance
print(f'Final Validation Performance: {rmse_val}')
