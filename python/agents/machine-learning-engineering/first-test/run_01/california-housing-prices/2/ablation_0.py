
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load the datasets
# Assuming the files are in the './input/' directory as per common Kaggle setup
train_df = pd.read_csv("./input/train.csv")

# Separate target variable from features
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Split the training data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical features (all features except target are numerical here)
numerical_features = X_train.select_dtypes(include=np.number).columns


# --- Baseline Performance (Original Solution) ---

# Impute missing values for baseline
imputer_baseline = SimpleImputer(strategy='median')
X_train_imputed_baseline = pd.DataFrame(imputer_baseline.fit_transform(X_train[numerical_features]),
                                       columns=numerical_features, index=X_train.index)
X_val_imputed_baseline = pd.DataFrame(imputer_baseline.transform(X_val[numerical_features]),
                                     columns=numerical_features, index=X_val.index)

# Train LightGBM for baseline
model_lightgbm_baseline = lgb.LGBMRegressor(objective='regression', n_estimators=500,
                                            learning_rate=0.05, random_state=42, n_jobs=-1)
model_lightgbm_baseline.fit(X_train_imputed_baseline, y_train)

# Train XGBoost for baseline
model_xgboost_baseline = XGBRegressor(objective='reg:squarederror', n_estimators=500,
                                     learning_rate=0.05, random_state=42, n_jobs=-1)
model_xgboost_baseline.fit(X_train_imputed_baseline, y_train)

# Predict and ensemble for baseline
y_val_pred_lightgbm_baseline = model_lightgbm_baseline.predict(X_val_imputed_baseline)
y_val_pred_xgboost_baseline = model_xgboost_baseline.predict(X_val_imputed_baseline)
y_val_pred_ensembled_baseline = (y_val_pred_lightgbm_baseline + y_val_pred_xgboost_baseline) / 2

rmse_val_ensembled_baseline = np.sqrt(mean_squared_error(y_val, y_val_pred_ensembled_baseline))
print(f"Baseline Performance (Ensemble with Imputation): {rmse_val_ensembled_baseline:.4f}")


# --- Ablation 1: LightGBM Only (Remove XGBoost and Ensemble) ---

# Use the same imputed data as baseline
X_train_imputed_ablation1 = X_train_imputed_baseline
X_val_imputed_ablation1 = X_val_imputed_baseline

# Train only LightGBM
model_lightgbm_ablation1 = lgb.LGBMRegressor(objective='regression', n_estimators=500,
                                             learning_rate=0.05, random_state=42, n_jobs=-1)
model_lightgbm_ablation1.fit(X_train_imputed_ablation1, y_train)

# Predict with LightGBM only
y_val_pred_lightgbm_ablation1 = model_lightgbm_ablation1.predict(X_val_imputed_ablation1)

rmse_val_lightgbm_only = np.sqrt(mean_squared_error(y_val, y_val_pred_lightgbm_ablation1))
print(f"Ablation 1 Performance (LightGBM Only, with Imputation): {rmse_val_lightgbm_only:.4f}")


# --- Ablation 2: Remove Imputation (Original Ensemble, but without SimpleImputer) ---

# Use original data with NaNs for training and validation
X_train_no_impute = X_train[numerical_features]
X_val_no_impute = X_val[numerical_features]

# Train LightGBM without imputation
model_lightgbm_no_impute = lgb.LGBMRegressor(objective='regression', n_estimators=500,
                                             learning_rate=0.05, random_state=42, n_jobs=-1)
model_lightgbm_no_impute.fit(X_train_no_impute, y_train)

# Train XGBoost without imputation
# FIX: Changed 'reg:squareerror' to 'reg:squarederror'
model_xgboost_no_impute = XGBRegressor(objective='reg:squarederror', n_estimators=500,
                                      learning_rate=0.05, random_state=42, n_jobs=-1)
model_xgboost_no_impute.fit(X_train_no_impute, y_train)

# Predict and ensemble without imputation
y_val_pred_lightgbm_no_impute = model_lightgbm_no_impute.predict(X_val_no_impute)
y_val_pred_xgboost_no_impute = model_xgboost_no_impute.predict(X_val_no_impute)
y_val_pred_ensembled_no_impute = (y_val_pred_lightgbm_no_impute + y_val_pred_xgboost_no_impute) / 2

rmse_val_ensembled_no_impute = np.sqrt(mean_squared_error(y_val, y_val_pred_ensembled_no_impute))
print(f"Ablation 2 Performance (Ensemble without Imputation): {rmse_val_ensembled_no_impute:.4f}")


# --- Conclusion ---

print("\n--- Ablation Study Summary ---")
print(f"Baseline (Ensemble with Imputation):              {rmse_val_ensembled_baseline:.4f}")
print(f"Ablation 1 (LightGBM Only, with Imputation):      {rmse_val_lightgbm_only:.4f}")
print(f"Ablation 2 (Ensemble without Imputation):         {rmse_val_ensembled_no_impute:.4f}")

# Determine which part contributes most
print("\n--- Contribution Analysis ---")

# Contribution of Ensemble: Compare Baseline to LightGBM Only
# A positive value means ensembling worsened performance (LightGBM only was better)
# A negative value means ensembling improved performance (Baseline was better)
ensemble_contribution_change = rmse_val_ensembled_baseline - rmse_val_lightgbm_only
if ensemble_contribution_change < 0: # Baseline (ensembled) is better (lower RMSE)
    print(f"Ensembling multiple models improved performance by: {-ensemble_contribution_change:.4f} RMSE.")
else: # LightGBM only is better or equal
    print(f"Ensembling multiple models worsened performance by: {ensemble_contribution_change:.4f} RMSE (LightGBM only was better/equal).")

# Contribution of Imputation: Compare Baseline to Ensemble without Imputation
# A positive value means imputation worsened performance (Ensemble without imputation was better)
# A negative value means imputation improved performance (Baseline was better)
imputation_contribution_change = rmse_val_ensembled_baseline - rmse_val_ensembled_no_impute
if imputation_contribution_change < 0: # Baseline (with imputation) is better (lower RMSE)
    print(f"Imputation improved performance by: {-imputation_contribution_change:.4f} RMSE.")
else: # Ensemble without imputation is better or equal
    print(f"Imputation worsened performance by: {imputation_contribution_change:.4f} RMSE (Ensemble without imputation was better/equal).")

# Simplified conclusion based on overall best
if rmse_val_ensembled_baseline <= rmse_val_lightgbm_only and rmse_val_ensembled_baseline <= rmse_val_ensembled_no_impute:
    print("\nBased on this ablation study, the baseline (ensemble with imputation) is the best performing configuration.")
elif rmse_val_lightgbm_only <= rmse_val_ensembled_baseline and rmse_val_lightgbm_only <= rmse_val_ensembled_no_impute:
    print("\nBased on this ablation study, LightGBM only (with imputation) is the best performing configuration.")
else:
    print("\nBased on this ablation study, the ensemble without imputation is the best performing configuration.")


# Final Validation Performance
final_validation_score = rmse_val_ensembled_baseline
print(f'Final Validation Performance: {final_validation_score}')
