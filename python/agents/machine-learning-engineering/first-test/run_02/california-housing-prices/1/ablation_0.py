
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the datasets
train_df = pd.read_csv('input/train.csv')

# Separate target variable from features
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Handle missing values in 'total_bedrooms'
median_total_bedrooms = X['total_bedrooms'].median()
X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Base Solution: Ensemble (LightGBM + XGBoost) ---
print("--- Base Solution: Ensemble (LightGBM + XGBoost) ---")

# Initialize and train LightGBM
model_lgbm = lgb.LGBMRegressor(random_state=42)
model_lgbm.fit(X_train, y_train)
y_pred_val_lgbm = model_lgbm.predict(X_val)

# Initialize and train XGBoost
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model_xgb.fit(X_train, y_train)
y_pred_val_xgb = model_xgb.predict(X_val)

# Ensemble predictions
y_pred_val_ensemble = (y_pred_val_lgbm + y_pred_val_xgb) / 2
rmse_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble))
print(f"Ensemble RMSE (Base Performance): {rmse_ensemble}")

# --- Ablation 1: Only LightGBM ---
print("\n--- Ablation 1: Only LightGBM ---")
# Predictions for LightGBM are already computed
rmse_lgbm_only = np.sqrt(mean_squared_error(y_val, y_pred_val_lgbm))
print(f"LightGBM Only RMSE: {rmse_lgbm_only}")
print(f"Modification effect: Disabling XGBoost and ensemble reduced performance by {rmse_lgbm_only - rmse_ensemble:.2f} RMSE.")

# --- Ablation 2: Only XGBoost ---
print("\n--- Ablation 2: Only XGBoost ---")
# Predictions for XGBoost are already computed
rmse_xgb_only = np.sqrt(mean_squared_error(y_val, y_pred_val_xgb))
print(f"XGBoost Only RMSE: {rmse_xgb_only}")
print(f"Modification effect: Disabling LightGBM and ensemble reduced performance by {rmse_xgb_only - rmse_ensemble:.2f} RMSE.")

print("\n--- Ablation Study Conclusion ---")
print(f"Overall performance (Ensemble RMSE): {rmse_ensemble}")
print(f"LightGBM Only RMSE: {rmse_lgbm_only}")
print(f"XGBoost Only RMSE: {rmse_xgb_only}")

# Determine which part contributes the most to the overall performance
# The "overall performance" is the ensemble RMSE, as it's the result of the full solution.
# A lower RMSE indicates better performance.

best_individual_rmse = min(rmse_lgbm_only, rmse_xgb_only)

if rmse_ensemble < best_individual_rmse:
    # Ensemble performs better than both individual models
    # Check if the improvement is significant (e.g., > 0.5% relative improvement)
    if (best_individual_rmse - rmse_ensemble) / rmse_ensemble > 0.005:
        print("\nThe ensemble strategy (combining LightGBM and XGBoost) contributes the most to the overall performance.")
    else:
        # If ensemble is only marginally better, the best individual model is the main driver
        if rmse_lgbm_only < rmse_xgb_only:
            print("\nLightGBM contributes the most to the overall performance, with the ensemble providing a marginal improvement.")
        else:
            print("\nXGBoost contributes the most to the overall performance, with the ensemble providing a marginal improvement.")
elif rmse_lgbm_only < rmse_xgb_only:
    print("\nLightGBM contributes the most to the overall performance.")
else:
    print("\nXGBoost contributes the most to the overall performance.")

