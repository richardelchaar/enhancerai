
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import os

# Load the training data
train_df = pd.read_csv('./input/train.csv')

# Separate target variable from features
X_full = train_df.drop('median_house_value', axis=1)
y_full = train_df['median_house_value']

# Split the data into training and validation sets
X_train_orig, X_val_orig, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# Dictionary to store RMSE results for comparison
results = {}

# --- Base Solution: Ensemble with Median Imputation ---
print("--- Base Solution: Ensemble with Median Imputation ---")

# Create copies of dataframes to avoid modifying originals for subsequent ablations
X_train_base = X_train_orig.copy()
X_val_base = X_val_orig.copy()

# Apply median imputation
imputer_base = SimpleImputer(strategy='median')
X_train_base['total_bedrooms'] = imputer_base.fit_transform(X_train_base[['total_bedrooms']])
X_val_base['total_bedrooms'] = imputer_base.transform(X_val_base[['total_bedrooms']])

# Initialize and train LightGBM Regressor
lgbm_model_base = lgb.LGBMRegressor(objective='regression_l2', random_state=42)
lgbm_model_base.fit(X_train_base, y_train)

# Initialize and train XGBoost Regressor
xgb_model_base = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model_base.fit(X_train_base, y_train)

# Make predictions on the validation set
y_pred_lgbm_base = lgbm_model_base.predict(X_val_base)
y_pred_xgb_base = xgb_model_base.predict(X_val_base)

# Ensemble predictions by averaging
y_pred_ensemble_base = (y_pred_lgbm_base + y_pred_xgb_base) / 2

# Evaluate the ensembled model
rmse_base = np.sqrt(mean_squared_error(y_val, y_pred_ensemble_base))
results['Base Solution (Ensemble + Imputation)'] = rmse_base
print(f'RMSE for Base Solution (Ensemble + Imputation): {rmse_base:.4f}\n')


# --- Ablation 1: Only LightGBM (with Median Imputation) ---
print("--- Ablation 1: Only LightGBM (with Median Imputation) ---")

X_train_lgbm_only = X_train_orig.copy()
X_val_lgbm_only = X_val_orig.copy()

# Apply median imputation (re-fit imputer for independent ablation context)
imputer_lgbm_only = SimpleImputer(strategy='median')
X_train_lgbm_only['total_bedrooms'] = imputer_lgbm_only.fit_transform(X_train_lgbm_only[['total_bedrooms']])
X_val_lgbm_only['total_bedrooms'] = imputer_lgbm_only.transform(X_val_lgbm_only[['total_bedrooms']])

# Train only LightGBM
lgbm_model_solo = lgb.LGBMRegressor(objective='regression_l2', random_state=42)
lgbm_model_solo.fit(X_train_lgbm_only, y_train)

# Make predictions
y_pred_lgbm_solo = lgbm_model_solo.predict(X_val_lgbm_only)

# Evaluate
rmse_lgbm_solo = np.sqrt(mean_squared_error(y_val, y_pred_lgbm_solo))
results['Ablation 1 (LGBM Only + Imputation)'] = rmse_lgbm_solo
print(f'RMSE for Ablation 1 (LGBM Only + Imputation): {rmse_lgbm_solo:.4f}\n')


# --- Ablation 2: Only XGBoost (with Median Imputation) ---
print("--- Ablation 2: Only XGBoost (with Median Imputation) ---")

X_train_xgb_only = X_train_orig.copy()
X_val_xgb_only = X_val_orig.copy()

# Apply median imputation (re-fit imputer for independent ablation context)
imputer_xgb_only = SimpleImputer(strategy='median')
X_train_xgb_only['total_bedrooms'] = imputer_xgb_only.fit_transform(X_train_xgb_only[['total_bedrooms']])
X_val_xgb_only['total_bedrooms'] = imputer_xgb_only.transform(X_val_xgb_only[['total_bedrooms']])

# Train only XGBoost
xgb_model_solo = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model_solo.fit(X_train_xgb_only, y_train)

# Make predictions
y_pred_xgb_solo = xgb_model_solo.predict(X_val_xgb_only)

# Evaluate
rmse_xgb_solo = np.sqrt(mean_squared_error(y_val, y_pred_xgb_solo))
results['Ablation 2 (XGBoost Only + Imputation)'] = rmse_xgb_solo
print(f'RMSE for Ablation 2 (XGBoost Only + Imputation): {rmse_xgb_solo:.4f}\n')


# --- Ablation 3: Ensemble without Imputation ---
print("--- Ablation 3: Ensemble without Imputation ---")

# Use original dataframes, which may contain NaNs
X_train_no_impute = X_train_orig.copy()
X_val_no_impute = X_val_orig.copy()

# LightGBM and XGBoost can handle NaNs internally, so no explicit imputation is applied here.

# Initialize and train LightGBM Regressor
lgbm_model_no_impute = lgb.LGBMRegressor(objective='regression_l2', random_state=42)
lgbm_model_no_impute.fit(X_train_no_impute, y_train)

# Initialize and train XGBoost Regressor
xgb_model_no_impute = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model_no_impute.fit(X_train_no_impute, y_train)

# Make predictions
y_pred_lgbm_no_impute = lgbm_model_no_impute.predict(X_val_no_impute)
y_pred_xgb_no_impute = xgb_model_no_impute.predict(X_val_no_impute)

# Ensemble predictions
y_pred_ensemble_no_impute = (y_pred_lgbm_no_impute + y_pred_xgb_no_impute) / 2

# Evaluate
rmse_no_impute = np.sqrt(mean_squared_error(y_val, y_pred_ensemble_no_impute))
results['Ablation 3 (Ensemble + No Imputation)'] = rmse_no_impute
print(f'RMSE for Ablation 3 (Ensemble + No Imputation): {rmse_no_impute:.4f}\n')


# --- Summary and Conclusion ---
print("--- Ablation Study Summary ---")
for label, rmse_value in results.items():
    print(f'{label}: {rmse_value:.4f}')

best_performance_label = min(results, key=results.get)
best_rmse = results[best_performance_label]

print("\n--- Conclusion on Most Contributing Part ---")
if best_performance_label == 'Base Solution (Ensemble + Imputation)':
    print(f"The combination of ensembling LightGBM and XGBoost models, along with median imputation for 'total_bedrooms', contributes the most to performance, achieving the lowest RMSE of {best_rmse:.4f}.")
elif best_performance_label == 'Ablation 1 (LGBM Only + Imputation)':
    print(f"The LightGBM model, combined with median imputation for 'total_bedrooms', contributes the most to performance, achieving the lowest RMSE of {best_rmse:.4f}. This suggests LightGBM alone performs better than the ensemble or XGBoost alone.")
elif best_performance_label == 'Ablation 2 (XGBoost Only + Imputation)':
    print(f"The XGBoost model, combined with median imputation for 'total_bedrooms', contributes the most to performance, achieving the lowest RMSE of {best_rmse:.4f}. This suggests XGBoost alone performs better than the ensemble or LightGBM alone.")
elif best_performance_label == 'Ablation 3 (Ensemble + No Imputation)':
    print(f"Ensembling LightGBM and XGBoost models contributes the most, and allowing models to handle missing 'total_bedrooms' values internally performs better than explicit median imputation, achieving the lowest RMSE of {best_rmse:.4f}.")

