
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from xgboost import XGBRegressor # Import XGBoost
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load the datasets
# Assuming the files are in the './input/' directory as per common Kaggle setup
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate target variable from features
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Split the training data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical features for imputation (all features except target are numerical here)

numerical_features = X_train.select_dtypes(include=np.number).columns

# Explicit imputation block removed as per the improvement plan.
# LightGBM and XGBoost models can handle missing values internally.
# The following variables are now assigned the numerical features directly,
# allowing the models to handle NaNs internally.
X_train_imputed = X_train[numerical_features].copy()
X_val_imputed = X_val[numerical_features].copy()
test_df_imputed = test_df[numerical_features].copy()


# --- Model Training ---

# 1. Initialize and train the LightGBM Regressor model (from base solution)
model_lightgbm = lgb.LGBMRegressor(objective='regression', n_estimators=500, 
                                   learning_rate=0.05, random_state=42, n_jobs=-1)
model_lightgbm.fit(X_train_imputed, y_train)

# 2. Initialize and train the XGBoost Regressor model (from reference solution)
model_xgboost = XGBRegressor(objective='reg:squarederror', n_estimators=500, 
                             learning_rate=0.05, random_state=42, n_jobs=-1)
model_xgboost.fit(X_train_imputed, y_train)

# --- Prediction and Ensemble ---

# Make predictions on the validation set for both models
y_val_pred_lightgbm = model_lightgbm.predict(X_val_imputed)
y_val_pred_xgboost = model_xgboost.predict(X_val_imputed)

# Ensemble the predictions by simple averaging
y_val_pred_ensembled = (y_val_pred_lightgbm + y_val_pred_xgboost) / 2

# Calculate Root Mean Squared Error (RMSE) on the ensembled validation predictions
rmse_val_ensembled = np.sqrt(mean_squared_error(y_val, y_val_pred_ensembled))

# Print the final ensembled validation performance
print(f"Final Validation Performance: {rmse_val_ensembled}")

# Generate predictions for the test dataset for submission (optional, but good practice for Kaggle)
test_predictions_lightgbm = model_lightgbm.predict(test_df_imputed)
test_predictions_xgboost = model_xgboost.predict(test_df_imputed)

# Ensemble test predictions
final_test_predictions = (test_predictions_lightgbm + test_predictions_xgboost) / 2

# Create submission file (uncomment to save if needed)
# submission_df = pd.DataFrame({'median_house_value': final_test_predictions})
# submission_df.to_csv('submission.csv', index=False)
