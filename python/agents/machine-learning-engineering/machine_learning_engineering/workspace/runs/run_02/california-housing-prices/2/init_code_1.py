
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# --- Preprocessing ---
# Identify features and target
TARGET = 'median_house_value'
# All columns except the target are features
FEATURES = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income'
]

# Handle missing values in 'total_bedrooms'
# Impute with the median value from the training data
median_total_bedrooms = train_df['total_bedrooms'].median()
train_df['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Prepare data for XGBoost
X = train_df[FEATURES]
y = train_df[TARGET]

# Split training data into training and validation sets
# Using a fixed random_state for reproducibility and consistent validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Initialize and Train XGBoost Model ---
# Initialize XGBoost Regressor with parameters similar to the example,
# suitable for a simple first solution.
# 'objective': 'reg:squarederror' is used for regression tasks to minimize squared error.
# 'n_jobs': -1 utilizes all available CPU cores, which is generally good practice.
model = xgb.XGBRegressor(objective='reg:squarederror',
                         n_estimators=100,
                         learning_rate=0.1,
                         max_depth=5,
                         random_state=42,
                         n_jobs=-1)

# Train the model on the training data
model.fit(X_train, y_train)

# --- Evaluate Model on Validation Set ---
# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Calculate Root Mean Squared Error (RMSE)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance in the specified format
print(f'Final Validation Performance: {rmse_val}')
