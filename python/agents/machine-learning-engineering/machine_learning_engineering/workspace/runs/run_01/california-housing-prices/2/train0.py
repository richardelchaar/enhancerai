

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor # Import XGBoost
from sklearn.metrics import mean_squared_error

# 1. Load Data
# Assuming train.csv and test.csv are available in the ./input/ directory
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# 2. Preprocessing
# Separate features (X) and target (y) from the training data
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Impute missing 'total_bedrooms' with the median from the training data to prevent data leakage.
# This ensures that test data imputation uses statistics only from the training data.
if 'total_bedrooms' in X.columns:
    median_total_bedrooms_train = X['total_bedrooms'].median()
    X['total_bedrooms'].fillna(median_total_bedrooms_train, inplace=True)
    # Apply the same imputation to the test set using the median from the training data
    if 'total_bedrooms' in test_df.columns:
        test_df['total_bedrooms'].fillna(median_total_bedrooms_train, inplace=True)

# 3. Split data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize the models
# Initialize LightGBM Regressor (from base solution)
model_lightgbm = LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1, random_state=42)

# Initialize XGBoost Regressor (from reference solution)
# Using objective='reg:squarederror' as recommended for regression tasks aiming at RMSE.
model_xgboost = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)

# 5. Train the models
print("Training LightGBM model...")
model_lightgbm.fit(X_train, y_train)
print("LightGBM training complete.")

print("Training XGBoost model...")
model_xgboost.fit(X_train, y_train)
print("XGBoost training complete.")

# 6. Make predictions on the validation set for each model
y_pred_val_lgbm = model_lightgbm.predict(X_val)
y_pred_val_xgboost = model_xgboost.predict(X_val)

# 7. Ensemble the predictions
# A simple averaging ensemble strategy is used here.
y_pred_val_ensembled = (y_pred_val_lgbm + y_pred_val_xgboost) / 2

# 8. Evaluate the ensembled model using Root Mean Squared Error (RMSE)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val_ensembled))

# Print the validation performance in the required format
print(f"Final Validation Performance: {rmse_val}")

