
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training dataset
train_df = pd.read_csv("./input/train.csv")

# Separate features (X) and the target variable (y)
# The target variable is 'median_house_value'
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# --- Preprocessing for Missing Values ---
# The 'total_bedrooms' column is known to have missing values in this dataset.
# Impute these missing values with the median of the column to prevent data leakage
# from the validation set, the median is calculated only from the training features.
median_total_bedrooms = X['total_bedrooms'].median()
X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# --- Data Splitting ---
# Split the processed training data into training and a hold-out validation set.
# This allows us to evaluate the model's performance on unseen data.
# A test_size of 0.2 means 20% of the data will be used for validation.
# random_state ensures reproducibility of the split.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Initialization and Training ---

# Initialize and train LightGBM Regressor model
model_lgb = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1 # Use all available cores
)
print("Training LightGBM model...")
model_lgb.fit(X_train, y_train)
print("LightGBM model training complete.")

# Initialize and train XGBoost Regressor model
model_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
print("Training XGBoost model...")
model_xgb.fit(X_train, y_train)
print("XGBoost model training complete.")

# --- Model Prediction and Ensembling ---
# Make predictions from LightGBM on the hold-out validation set
y_pred_lgb = model_lgb.predict(X_val)

# Make predictions from XGBoost on the hold-out validation set
y_pred_xgb = model_xgb.predict(X_val)

# Ensemble the predictions by simple averaging
# This combines the strengths of both models
y_pred_ensemble = (y_pred_lgb + y_pred_xgb) / 2

# --- Model Evaluation ---
# Calculate the Root Mean Squared Error (RMSE) for the ensembled predictions.
# This is the specified metric for the task.
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))

# Print the final validation performance in the required format
print(f"Final Validation Performance: {rmse_val}")
