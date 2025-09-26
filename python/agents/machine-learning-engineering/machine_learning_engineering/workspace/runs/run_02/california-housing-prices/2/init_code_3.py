
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# --- Preprocessing ---
# Identify features and target
TARGET = 'median_house_value'
FEATURES = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income'
]

# Handle missing values in 'total_bedrooms'
# Impute with the median value from the training data.
# This is a common practice for this dataset as 'total_bedrooms' often has a few NaNs.
median_total_bedrooms = train_df['total_bedrooms'].median()
train_df['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Prepare data for CatBoost
X = train_df[FEATURES]
y = train_df[TARGET]

# Split training data into training and validation sets
# Using a fixed random_state for reproducibility and consistent validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Initialize and Train CatBoost Model ---
# CatBoost can automatically handle categorical features if specified.
# In this specific dataset, all features selected are numerical,
# so the `cat_features` parameter will be an empty list or not provided.
categorical_features_indices = [] # No categorical features in the selected list

# Initialize CatBoost Regressor with parameters from the example description.
model = CatBoostRegressor(iterations=100,               # Number of boosting rounds
                          learning_rate=0.1,             # Step size shrinkage
                          depth=5,                       # Depth of the tree
                          loss_function='RMSE',          # Objective for regression
                          random_seed=42,                # For reproducibility
                          verbose=0,                     # Suppress verbose output during training
                          cat_features=categorical_features_indices) # Specify categorical features if any

# Train the model on the training data
model.fit(X_train, y_train)

# --- Evaluate Model on Validation Set ---
# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Calculate Root Mean Squared Error (RMSE)
# RMSE is the specified metric for this task.
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance in the specified format
print(f'Final Validation Performance: {rmse_val}')
