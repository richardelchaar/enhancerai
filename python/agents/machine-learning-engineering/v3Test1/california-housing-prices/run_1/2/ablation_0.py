
# Suppress verbose model output to prevent token explosion
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
# Suppress LightGBM verbosity
os.environ['LIGHTGBM_VERBOSITY'] = '-1'
# Suppress XGBoost verbosity  
os.environ['XGBOOST_VERBOSITY'] = '0'
# Suppress sklearn warnings
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)


import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Define RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate target variable
X_train_raw = train_df.drop('median_house_value', axis=1)
y_train = train_df['median_house_value']
X_test_raw = test_df.copy()

# Function to preprocess data based on imputation strategy
def preprocess_data(X_train_raw, X_test_raw, imputation_strategy='median'):
    # Combine train and test for consistent preprocessing
    all_data = pd.concat([X_train_raw, X_test_raw], ignore_index=True)

    # Feature Engineering
    all_data['rooms_per_household'] = all_data['total_rooms'] / all_data['households']
    all_data['population_per_household'] = all_data['population'] / all_data['households']
    all_data['bedrooms_per_room'] = all_data['total_bedrooms'] / all_data['total_rooms']

    # Imputation
    # Replace potential infinities with NaN (can occur from division by zero)
    all_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    if imputation_strategy == 'mean_bedrooms_only':
        # Impute 'total_bedrooms' with its mean
        mean_total_bedrooms = all_data['total_bedrooms'].mean()
        all_data['total_bedrooms'].fillna(mean_total_bedrooms, inplace=True)
        
        # For any other columns that might still have NaNs (e.g., engineered features if denominator was 0),
        # use median imputation as per baseline for consistency on other features.
        # Create a new imputer instance for general median imputation on remaining NaNs
        median_imputer_for_others = SimpleImputer(strategy='median')
        all_data_imputed = pd.DataFrame(median_imputer_for_others.fit_transform(all_data), columns=all_data.columns)
    else: # 'median' (baseline) or 'mean' globally if specified
        imputer = SimpleImputer(strategy=imputation_strategy)
        all_data_imputed = pd.DataFrame(imputer.fit_transform(all_data), columns=all_data.columns)

    # Scaling
    scaler = StandardScaler()
    all_data_scaled = pd.DataFrame(scaler.fit_transform(all_data_imputed), columns=all_data_imputed.columns)

    # Split back into train and test
    X_train_processed = all_data_scaled.iloc[:len(X_train_raw)]
    X_test_processed = all_data_scaled.iloc[len(X_train_raw):]

    return X_train_processed, X_test_processed

# Define KFold for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Model parameters (suppressing verbose output)
lgb_params = {
    'objective': 'regression_l1', # MAE, robust to outliers
    'metric': 'rmse',
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1, # Suppress verbose output
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 0.1, # L2 regularization
    'alpha': 0.1,  # L1 regularization
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0 # Suppress verbose output
}

# --- Baseline Model (LGBM + XGBoost, Median Imputation) ---
print("Running Baseline Model...")
X_processed_baseline, X_test_processed_baseline = preprocess_data(X_train_raw, X_test_raw, imputation_strategy='median')

oof_preds_lgb_baseline = np.zeros(len(X_processed_baseline))
test_preds_lgb_baseline = np.zeros(len(X_test_processed_baseline))
oof_preds_xgb_baseline = np.zeros(len(X_processed_baseline))
test_preds_xgb_baseline = np.zeros(len(X_test_processed_baseline))

for fold, (train_index, val_index) in enumerate(kf.split(X_processed_baseline, y_train)):
    X_train, X_val = X_processed_baseline.iloc[train_index], X_processed_baseline.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train_fold,
                  eval_set=[(X_val, y_val_fold)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])

    oof_preds_lgb_baseline[val_index] = lgb_model.predict(X_val)
    test_preds_lgb_baseline += lgb_model.predict(X_test_processed_baseline) / kf.n_splits

    # XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train_fold,
                  eval_set=[(X_val, y_val_fold)],
                  early_stopping_rounds=100,
                  verbose=False)

    oof_preds_xgb_baseline[val_index] = xgb_model.predict(X_val)
    test_preds_xgb_baseline += xgb_model.predict(X_test_processed_baseline) / kf.n_splits

# Ensemble predictions for baseline
oof_ensemble_baseline = (oof_preds_lgb_baseline + oof_preds_xgb_baseline) / 2
test_ensemble_baseline = (test_preds_lgb_baseline + test_preds_xgb_baseline) / 2
baseline_rmse = rmse(y_train, oof_ensemble_baseline)
print(f"Baseline RMSE (LGBM + XGBoost, Median Imputation): {baseline_rmse:.4f}")

# Store results
results = {'Baseline': baseline_rmse}
ablation_performance = {}

# --- Ablation 1: LightGBM only (no XGBoost in ensemble) ---
print("\nRunning Ablation 1: LightGBM only...")
oof_preds_lgb_ablation1 = np.zeros(len(X_processed_baseline))
test_preds_lgb_ablation1 = np.zeros(len(X_test_processed_baseline))

for fold, (train_index, val_index) in enumerate(kf.split(X_processed_baseline, y_train)):
    X_train, X_val = X_processed_baseline.iloc[train_index], X_processed_baseline.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    lgb_model_ablation1 = lgb.LGBMRegressor(**lgb_params)
    lgb_model_ablation1.fit(X_train, y_train_fold,
                            eval_set=[(X_val, y_val_fold)],
                            callbacks=[lgb.early_stopping(100, verbose=False)])

    oof_preds_lgb_ablation1[val_index] = lgb_model_ablation1.predict(X_val)
    test_preds_lgb_ablation1 += lgb_model_ablation1.predict(X_test_processed_baseline) / kf.n_splits

ablation1_rmse = rmse(y_train, oof_preds_lgb_ablation1)
print(f"Ablation 1 RMSE (LightGBM only, Median Imputation): {ablation1_rmse:.4f}")
ablation_performance['Model Choice - LightGBM Only'] = ablation1_rmse

# --- Ablation 2: Mean Imputation for total_bedrooms (Ensemble LGBM+XGBoost) ---
print("\nRunning Ablation 2: Mean Imputation for total_bedrooms...")
X_processed_ablation2, X_test_processed_ablation2 = preprocess_data(X_train_raw, X_test_raw, imputation_strategy='mean_bedrooms_only')

oof_preds_lgb_ablation2 = np.zeros(len(X_processed_ablation2))
test_preds_lgb_ablation2 = np.zeros(len(X_test_processed_ablation2))
oof_preds_xgb_ablation2 = np.zeros(len(X_processed_ablation2))
test_preds_xgb_ablation2 = np.zeros(len(X_test_processed_ablation2))

for fold, (train_index, val_index) in enumerate(kf.split(X_processed_ablation2, y_train)):
    X_train, X_val = X_processed_ablation2.iloc[train_index], X_processed_ablation2.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    # LightGBM
    lgb_model_ablation2 = lgb.LGBMRegressor(**lgb_params)
    lgb_model_ablation2.fit(X_train, y_train_fold,
                            eval_set=[(X_val, y_val_fold)],
                            callbacks=[lgb.early_stopping(100, verbose=False)])

    oof_preds_lgb_ablation2[val_index] = lgb_model_ablation2.predict(X_val)
    test_preds_lgb_ablation2 += lgb_model_ablation2.predict(X_test_processed_ablation2) / kf.n_splits

    # XGBoost
    xgb_model_ablation2 = xgb.XGBRegressor(**xgb_params)
    xgb_model_ablation2.fit(X_train, y_train_fold,
                            eval_set=[(X_val, y_val_fold)],
                            early_stopping_rounds=100,
                            verbose=False)

    oof_preds_xgb_ablation2[val_index] = xgb_model_ablation2.predict(X_val)
    test_preds_xgb_ablation2 += xgb_model_ablation2.predict(X_test_processed_ablation2) / kf.n_splits

oof_ensemble_ablation2 = (oof_preds_lgb_ablation2 + oof_preds_xgb_ablation2) / 2
test_ensemble_ablation2 = (test_preds_lgb_ablation2 + test_preds_xgb_ablation2) / 2
ablation2_rmse = rmse(y_train, oof_ensemble_ablation2)
print(f"Ablation 2 RMSE (LGBM + XGBoost, Mean Imputation for total_bedrooms): {ablation2_rmse:.4f}")
ablation_performance['Imputation Strategy - Mean for Bedrooms'] = ablation2_rmse


# --- Summary ---
print("\n--- Ablation Study Summary ---")
print(f"Baseline RMSE: {baseline_rmse:.4f}")
for ablation_name, perf in ablation_performance.items():
    print(f"{ablation_name} RMSE: {perf:.4f}")

# Determine the largest change
largest_change = 0
largest_change_ablation = "None"

for ablation_name, perf in ablation_performance.items():
    change = abs(perf - baseline_rmse)
    if change > largest_change:
        largest_change = change
        largest_change_ablation = ablation_name

print(f"\nThe component whose ablation resulted in the largest change in performance is: {largest_change_ablation}")
print(f"Change magnitude: {largest_change:.4f}")

# Final Prediction (using baseline model's test predictions)
final_predictions = test_ensemble_baseline

# Save submission
submission_df = pd.DataFrame({'median_house_value': final_predictions})
submission_df.to_csv("submission.csv", index=False)
print("Submission file created successfully!")

final_validation_score = baseline_rmse
print(f'Final Validation Performance: {final_validation_score}')
