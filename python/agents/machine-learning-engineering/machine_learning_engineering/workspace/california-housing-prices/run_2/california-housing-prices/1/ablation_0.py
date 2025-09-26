
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training data
try:
    df_train = pd.read_csv("./input/train.csv")
except FileNotFoundError:
    df_train = pd.read_csv("train.csv")

# Separate features (X) and target (y)
X_full = df_train.drop("median_house_value", axis=1)
y_full = df_train["median_house_value"]

# --- Baseline Performance (Original Solution) ---
# Split the data into training and validation sets for the baseline
X_train_base, X_val_base, y_train_base, y_val_base = train_test_split(X_full.copy(), y_full.copy(), test_size=0.2, random_state=42)

# Impute missing values in 'total_bedrooms' with the mean from the training set
mean_total_bedrooms_base = X_train_base['total_bedrooms'].mean()
X_train_base['total_bedrooms'].fillna(mean_total_bedrooms_base, inplace=True)
X_val_base['total_bedrooms'].fillna(mean_total_bedrooms_base, inplace=True)

# LightGBM Model
lgbm_base = lgb.LGBMRegressor(random_state=42)
lgbm_base.fit(X_train_base, y_train_base)
y_pred_lgbm_base = lgbm_base.predict(X_val_base)

# XGBoost Model
model_xgb_base = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model_xgb_base.fit(X_train_base, y_train_base)
y_pred_xgb_base = model_xgb_base.predict(X_val_base)

# Simple averaging ensemble
y_pred_ensemble_base = (y_pred_lgbm_base + y_pred_xgb_base) / 2
rmse_ensemble_base = np.sqrt(mean_squared_error(y_val_base, y_pred_ensemble_base))
print(f"Baseline (Original Solution) RMSE: {rmse_ensemble_base:.6f}")

# --- Ablation 1: LightGBM Only (Remove XGBoost and Ensemble) ---
# Re-split data to ensure a clean slate and consistent random state
X_train_ab1, X_val_ab1, y_train_ab1, y_val_ab1 = train_test_split(X_full.copy(), y_full.copy(), test_size=0.2, random_state=42)
mean_total_bedrooms_ab1 = X_train_ab1['total_bedrooms'].mean() # Re-calculate mean for consistency in new split
X_train_ab1['total_bedrooms'].fillna(mean_total_bedrooms_ab1, inplace=True)
X_val_ab1['total_bedrooms'].fillna(mean_total_bedrooms_ab1, inplace=True)

# Only train and predict with LightGBM
lgbm_ab1 = lgb.LGBMRegressor(random_state=42)
lgbm_ab1.fit(X_train_ab1, y_train_ab1)
y_pred_lgbm_ab1 = lgbm_ab1.predict(X_val_ab1)

rmse_lgbm_only = np.sqrt(mean_squared_error(y_val_ab1, y_pred_lgbm_ab1))
print(f"Ablation 1 (LightGBM Only) RMSE: {rmse_lgbm_only:.6f}")

# --- Ablation 2: Change Imputation Method for 'total_bedrooms' to Median ---
# Re-split data to ensure a clean slate before applying a different imputation
X_train_ab2, X_val_ab2, y_train_ab2, y_val_ab2 = train_test_split(X_full.copy(), y_full.copy(), test_size=0.2, random_state=42)

# Impute missing values in 'total_bedrooms' with the MEDIAN from the training set
median_total_bedrooms_ab2 = X_train_ab2['total_bedrooms'].median() # Key change here
X_train_ab2['total_bedrooms'].fillna(median_total_bedrooms_ab2, inplace=True)
X_val_ab2['total_bedrooms'].fillna(median_total_bedrooms_ab2, inplace=True)

# Re-train both models with the new imputation strategy
lgbm_ab2 = lgb.LGBMRegressor(random_state=42)
lgbm_ab2.fit(X_train_ab2, y_train_ab2)
y_pred_lgbm_ab2 = lgbm_ab2.predict(X_val_ab2)

model_xgb_ab2 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model_xgb_ab2.fit(X_train_ab2, y_train_ab2)
y_pred_xgb_ab2 = model_xgb_ab2.predict(X_val_ab2)

# Ensemble predictions with new imputation
y_pred_ensemble_ab2 = (y_pred_lgbm_ab2 + y_pred_xgb_ab2) / 2
rmse_median_imputation = np.sqrt(mean_squared_error(y_val_ab2, y_pred_ensemble_ab2))
print(f"Ablation 2 (Median Imputation) RMSE: {rmse_median_imputation:.6f}")

# --- Conclusion: Which part contributes the most to the overall performance? ---
# We assess contribution by how much performance (RMSE) *worsens* when a component is removed or changed.
# A larger increase in RMSE indicates a more significant positive contribution from the original component.

performance_change_ensemble_removal = rmse_lgbm_only - rmse_ensemble_base
performance_change_imputation_strategy = rmse_median_imputation - rmse_ensemble_base

if performance_change_ensemble_removal >= 0 and performance_change_imputation_strategy >= 0:
    # Both ablations worsened performance (RMSE increased), meaning original components were beneficial.
    if performance_change_ensemble_removal > performance_change_imputation_strategy:
        print("The ensemble (XGBoost + LightGBM combination) contributes the most to the overall performance, as its removal led to the largest performance drop.")
    else:
        print("The mean imputation method for 'total_bedrooms' contributes the most to the overall performance, as changing it to median led to the largest performance drop.")
elif performance_change_ensemble_removal < 0 and performance_change_imputation_strategy >= 0:
    # Removing ensemble *improved* performance, while changing imputation *worsened* it.
    print("The mean imputation method for 'total_bedrooms' contributes positively to the overall performance.")
    print(f"Note: Removing the ensemble improved performance by {abs(performance_change_ensemble_removal):.6f}, suggesting the ensemble might be suboptimal.")
elif performance_change_ensemble_removal >= 0 and performance_change_imputation_strategy < 0:
    # Removing ensemble *worsened* performance, while changing imputation *improved* it.
    print("The ensemble (XGBoost + LightGBM combination) contributes the most to the overall performance.")
    print(f"Note: Changing imputation to median improved performance by {abs(performance_change_imputation_strategy):.6f}, suggesting mean imputation might be suboptimal.")
else: # Both ablations improved performance (RMSE decreased), meaning original components were suboptimal.
    print("Neither original component contributed positively compared to its ablated version (both ablations improved performance).")
    print(f"Note: Removing the ensemble improved performance by {abs(performance_change_ensemble_removal):.6f}.")
    print(f"Note: Changing imputation to median improved performance by {abs(performance_change_imputation_strategy):.6f}.")
