
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load the datasets
train_df = pd.read_csv("./input/train.csv")

# Separate target variable from features in the training data
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Identify numerical features for imputation.
numerical_cols = X.select_dtypes(include=np.number).columns

# Initialize SimpleImputer to handle missing values.
imputer = SimpleImputer(strategy='median')

# Fit the imputer on the training features and transform them.
X_imputed_array = imputer.fit_transform(X[numerical_cols])
X = pd.DataFrame(X_imputed_array, columns=numerical_cols, index=X.index)

# Split the processed training data into training and validation sets.
X_train_split, X_val, y_train_split, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Base Models Initialization (used across ablations) ---
lgbm_base_model = lgb.LGBMRegressor(objective='regression',
                                    metric='rmse',
                                    n_estimators=100,
                                    learning_rate=0.1,
                                    num_leaves=31,
                                    random_state=42,
                                    n_jobs=-1)

xgb_base_model = xgb.XGBRegressor(objective='reg:squarederror',
                                  n_estimators=100,
                                  learning_rate=0.1,
                                  max_depth=5,
                                  random_state=42,
                                  n_jobs=-1)

# --- Ablation Study ---

print("--- Ablation Study Results ---")

# Scenario 1: Original Solution (LightGBM + XGBoost Ensemble)
print("\nScenario 1: Original Solution (LightGBM + XGBoost Ensemble)")
lgbm_base_model.fit(X_train_split, y_train_split)
y_pred_val_lgbm_s1 = lgbm_base_model.predict(X_val)

xgb_base_model.fit(X_train_split, y_train_split)
y_pred_val_xgb_s1 = xgb_base_model.predict(X_val)

y_pred_val_ensemble_s1 = (y_pred_val_lgbm_s1 + y_pred_val_xgb_s1) / 2
rmse_val_ensemble_s1 = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble_s1))
print(f"Original Ensemble RMSE: {rmse_val_ensemble_s1:.4f}")


# Scenario 2: Ablation - Use only LightGBM (remove XGBoost and ensembling)
print("\nScenario 2: Ablation - Only LightGBM Model")
# LightGBM is already trained from Scenario 1, or can be retrained.
# For simplicity and to reflect 'disabling' a part, we use its predictions directly.
y_pred_val_lgbm_s2 = lgbm_base_model.predict(X_val) # Reuse prediction from S1 for efficiency
rmse_val_lgbm_s2 = np.sqrt(mean_squared_error(y_val, y_pred_val_lgbm_s2))
print(f"LightGBM Only RMSE: {rmse_val_lgbm_s2:.4f}")
print(f"Modification: Removed XGBoost model and simple averaging ensemble.")
print(f"Effect: Performance changed from {rmse_val_ensemble_s1:.4f} to {rmse_val_lgbm_s2:.4f}.")


# Scenario 3: Ablation - Use only XGBoost (remove LightGBM and ensembling)
print("\nScenario 3: Ablation - Only XGBoost Model")
# XGBoost is already trained from Scenario 1, or can be retrained.
y_pred_val_xgb_s3 = xgb_base_model.predict(X_val) # Reuse prediction from S1 for efficiency
rmse_val_xgb_s3 = np.sqrt(mean_squared_error(y_val, y_pred_val_xgb_s3))
print(f"XGBoost Only RMSE: {rmse_val_xgb_s3:.4f}")
print(f"Modification: Removed LightGBM model and simple averaging ensemble.")
print(f"Effect: Performance changed from {rmse_val_ensemble_s1:.4f} to {rmse_val_xgb_s3:.4f}.")


# Conclusion about contributions
print("\n--- Contribution Analysis ---")
if rmse_val_ensemble_s1 < min(rmse_val_lgbm_s2, rmse_val_xgb_s3):
    print("The simple averaging ensemble of LightGBM and XGBoost contributes positively, resulting in the best performance.")
    if rmse_val_lgbm_s2 < rmse_val_xgb_s3:
        print(f"Among individual models, LightGBM (RMSE: {rmse_val_lgbm_s2:.4f}) performs better than XGBoost (RMSE: {rmse_val_xgb_s3:.4f}).")
        print(f"The LightGBM model appears to be the most significant contributor to the overall performance in the ensemble, as its individual performance is closer to the ensemble's and superior to XGBoost's.")
    else:
        print(f"Among individual models, XGBoost (RMSE: {rmse_val_xgb_s3:.4f}) performs better than LightGBM (RMSE: {rmse_val_lgbm_s2:.4f}).")
        print(f"The XGBoost model appears to be the most significant contributor to the overall performance in the ensemble, as its individual performance is closer to the ensemble's and superior to LightGBM's.")
else:
    if rmse_val_lgbm_s2 < rmse_val_xgb_s3:
        print(f"The LightGBM model (RMSE: {rmse_val_lgbm_s2:.4f}) is the most significant contributor to the overall performance, as it outperforms the ensemble (RMSE: {rmse_val_ensemble_s1:.4f}) and XGBoost (RMSE: {rmse_val_xgb_s3:.4f}). The ensemble degraded performance slightly.")
    else:
        print(f"The XGBoost model (RMSE: {rmse_val_xgb_s3:.4f}) is the most significant contributor to the overall performance, as it outperforms the ensemble (RMSE: {rmse_val_ensemble_s1:.4f}) and LightGBM (RMSE: {rmse_val_lgbm_s2:.4f}). The ensemble degraded performance slightly.")
