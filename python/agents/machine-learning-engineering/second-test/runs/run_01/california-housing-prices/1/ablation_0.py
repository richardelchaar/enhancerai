
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
try:
    train_df_orig = pd.read_csv("./input/train.csv")
except FileNotFoundError:
    train_df_orig = pd.read_csv("train.csv")

# Identify features and target
TARGET_COL = 'median_house_value'
FEATURES = [col for col in train_df_orig.columns if col != TARGET_COL]

# --- BASELINE: Original Solution (Imputation + Ensemble) ---
train_df_base = train_df_orig.copy()
median_total_bedrooms_train = train_df_base['total_bedrooms'].median()
train_df_base['total_bedrooms'].fillna(median_total_bedrooms_train, inplace=True)

X_base = train_df_base[FEATURES]
y_base = train_df_base[TARGET_COL]
X_train_base, X_val_base, y_train_base, y_val_base = train_test_split(X_base, y_base, test_size=0.2, random_state=42)

lgbm_model_base = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, n_jobs=-1)
lgbm_model_base.fit(X_train_base, y_train_base)
y_val_pred_lgbm_base = lgbm_model_base.predict(X_val_base)

xgb_model_base = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_jobs=-1)
xgb_model_base.fit(X_train_base, y_train_base)
y_val_pred_xgb_base = xgb_model_base.predict(X_val_base)

y_val_pred_ensemble_base = (y_val_pred_lgbm_base + y_val_pred_xgb_base) / 2
rmse_base = np.sqrt(mean_squared_error(y_val_base, y_val_pred_ensemble_base))
print(f"Base Case RMSE (Imputation + Ensemble): {rmse_base}")

# --- ABLATION 1: No total_bedrooms Imputation (Ensemble still used) ---
train_df_ablation1 = train_df_orig.copy() # Start fresh with original data
# No imputation for 'total_bedrooms'. Models (LGBM, XGBoost) can handle NaNs by default.

X_ablation1 = train_df_ablation1[FEATURES]
y_ablation1 = train_df_ablation1[TARGET_COL]
X_train_ablation1, X_val_ablation1, y_train_ablation1, y_val_ablation1 = train_test_split(X_ablation1, y_ablation1, test_size=0.2, random_state=42)

lgbm_model_ablation1 = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, n_jobs=-1)
lgbm_model_ablation1.fit(X_train_ablation1, y_train_ablation1)
y_val_pred_lgbm_ablation1 = lgbm_model_ablation1.predict(X_val_ablation1)

xgb_model_ablation1 = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_jobs=-1)
xgb_model_ablation1.fit(X_train_ablation1, y_train_ablation1)
y_val_pred_xgb_ablation1 = xgb_model_ablation1.predict(X_val_ablation1)

y_val_pred_ensemble_ablation1 = (y_val_pred_lgbm_ablation1 + y_val_pred_xgb_ablation1) / 2
rmse_ablation1 = np.sqrt(mean_squared_error(y_val_ablation1, y_val_pred_ensemble_ablation1))
print(f"Ablation 1 RMSE (No Imputation, Ensemble): {rmse_ablation1}")

# --- ABLATION 2: Imputation, LightGBM Only (No XGBoost, No Ensemble) ---
train_df_ablation2 = train_df_orig.copy()
median_total_bedrooms_train_ablation2 = train_df_ablation2['total_bedrooms'].median()
train_df_ablation2['total_bedrooms'].fillna(median_total_bedrooms_train_ablation2, inplace=True)

X_ablation2 = train_df_ablation2[FEATURES]
y_ablation2 = train_df_ablation2[TARGET_COL]
X_train_ablation2, X_val_ablation2, y_train_ablation2, y_val_ablation2 = train_test_split(X_ablation2, y_ablation2, test_size=0.2, random_state=42)

lgbm_model_ablation2 = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, n_jobs=-1)
lgbm_model_ablation2.fit(X_train_ablation2, y_train_ablation2)
y_val_pred_lgbm_ablation2 = lgbm_model_ablation2.predict(X_val_ablation2)

rmse_ablation2 = np.sqrt(mean_squared_error(y_val_ablation2, y_val_pred_lgbm_ablation2))
print(f"Ablation 2 RMSE (Imputation, LightGBM Only): {rmse_ablation2}")

# --- Determine Contribution ---
degradation_imputation = rmse_ablation1 - rmse_base
degradation_ensembling = rmse_ablation2 - rmse_base

print(f"Performance degradation (RMSE increase) from removing Imputation: {degradation_imputation:.4f}")
print(f"Performance degradation (RMSE increase) from removing Ensembling: {degradation_ensembling:.4f}")

if degradation_imputation > degradation_ensembling:
    print("The imputation of 'total_bedrooms' contributes more to the overall performance.")
elif degradation_ensembling > degradation_imputation:
    print("The ensembling of LightGBM and XGBoost models contributes more to the overall performance.")
else:
    print("Both imputation and ensembling contribute similarly to the overall performance, or their impact is negligible.")
