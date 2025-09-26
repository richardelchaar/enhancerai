
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import subprocess
import sys
import os

# Install catboost if not already installed
try:
    from catboost import CatBoostRegressor
except ImportError:
    print("CatBoost not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
    from catboost import CatBoostRegressor

# Install lightgbm if not already installed
try:
    import lightgbm as lgb
except ImportError:
    print("LightGBM not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
    import lightgbm as lgb

# --- Configuration ---
TRAIN_FILE = './input/train.csv'
TARGET_COLUMN = 'median_house_value'
RANDOM_SEED = 42
VALIDATION_SPLIT_RATIO = 0.2

# Create dummy input directory and files if they don't exist for local testing
# This is for reproducibility/local execution outside Kaggle environment
if not os.path.exists('./input'):
    os.makedirs('./input')
    # Create dummy train.csv if it doesn't exist
    if not os.path.exists(TRAIN_FILE):
        print(f"Creating dummy {TRAIN_FILE} for testing purposes.")
        dummy_data = {
            'feature_1': np.random.rand(100),
            'feature_2': np.random.rand(100) * 10,
            'total_bedrooms': np.random.rand(100) * 500, # Will have some NaNs
            'median_house_value': np.random.rand(100) * 100000 + 50000
        }
        dummy_data['total_bedrooms'][::10] = np.nan # Introduce some NaNs
        pd.DataFrame(dummy_data).to_csv(TRAIN_FILE, index=False)


# --- 1. Load Data ---
train_df = pd.read_csv(TRAIN_FILE)

# Separate target variable from training features
X_full = train_df.drop(TARGET_COLUMN, axis=1)
y_full = train_df[TARGET_COLUMN]

# --- 2. Handle Missing Values (Baseline Imputation) ---
imputer = SimpleImputer(strategy='median')
X_imputed_full = imputer.fit_transform(X_full)
X_imputed_full = pd.DataFrame(X_imputed_full, columns=X_full.columns)

# --- 3. Split Data for Validation ---
X_train, X_val, y_train, y_val = train_test_split(
    X_imputed_full, y_full, test_size=VALIDATION_SPLIT_RATIO, random_state=RANDOM_SEED
)

# Store results for comparison
ablation_results = {}

# --- Baseline: Full Ensemble Performance ---
print("--- Running Baseline: Full Ensemble ---")

# CatBoost Regressor
model_cb = CatBoostRegressor(
    loss_function='RMSE',
    random_seed=RANDOM_SEED,
    verbose=False,
    iterations=1000,
    learning_rate=0.05,
    early_stopping_rounds=50
)
model_cb.fit(X_train, y_train, eval_set=(X_val, y_val))
cb_val_predictions = model_cb.predict(X_val)

# LightGBM Model
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': RANDOM_SEED,
    'boosting_type': 'gbdt',
}
model_lgb = lgb.LGBMRegressor(**lgb_params)
model_lgb.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
lgb_val_predictions = model_lgb.predict(X_val)

# Ensembling (Simple Averaging)
ensemble_val_predictions = (cb_val_predictions + lgb_val_predictions) / 2
baseline_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_predictions))
ablation_results['Full Ensemble (Baseline)'] = baseline_rmse
print(f"Baseline (Full Ensemble) RMSE: {baseline_rmse:.4f}\n")


# --- Ablation 1: Only CatBoost Model (remove LightGBM from ensemble) ---
print("--- Running Ablation 1: CatBoost Only ---")
# CatBoost model is already trained from baseline
ablation1_rmse = np.sqrt(mean_squared_error(y_val, cb_val_predictions))
ablation_results['CatBoost Only'] = ablation1_rmse
print(f"CatBoost Only RMSE: {ablation1_rmse:.4f}\n")


# --- Ablation 2: Only LightGBM Model (remove CatBoost from ensemble) ---
print("--- Running Ablation 2: LightGBM Only ---")
# LightGBM model is already trained from baseline
ablation2_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_predictions))
ablation_results['LightGBM Only'] = ablation2_rmse
print(f"LightGBM Only RMSE: {ablation2_rmse:.4f}\n")


# --- Ablation 3: No Missing Value Imputation ---
# This part ablates the missing value handling strategy.
print("--- Running Ablation 3: No Missing Value Imputation ---")
# Reset data to original, then split without imputation
X_no_impute = X_full.copy() # Use original X_full which might have NaNs
# Drop rows with NaNs as models usually don't handle them directly
# This is a strong ablation, effectively removing data for rows with missing values
X_no_impute_dropped, y_no_impute_dropped = X_no_impute.dropna(), y_full[X_no_impute.dropna().index]

# If there are no NaNs or too few, this ablation might not be impactful or might fail
if X_no_impute_dropped.empty:
    print("Warning: All rows removed due to NaNs when trying 'No Missing Value Imputation' ablation. Skipping this ablation.")
    ablation_results['No Imputation (Dropped NaNs)'] = float('inf') # Indicate failure/high error
else:
    X_train_no_impute, X_val_no_impute, y_train_no_impute, y_val_no_impute = train_test_split(
        X_no_impute_dropped, y_no_impute_dropped, test_size=VALIDATION_SPLIT_RATIO, random_state=RANDOM_SEED
    )

    # Re-train CatBoost
    model_cb_no_impute = CatBoostRegressor(
        loss_function='RMSE', random_seed=RANDOM_SEED, verbose=False,
        iterations=1000, learning_rate=0.05, early_stopping_rounds=50
    )
    model_cb_no_impute.fit(X_train_no_impute, y_train_no_impute, eval_set=(X_val_no_impute, y_val_no_impute))
    cb_val_predictions_no_impute = model_cb_no_impute.predict(X_val_no_impute)

    # Re-train LightGBM
    model_lgb_no_impute = lgb.LGBMRegressor(**lgb_params)
    model_lgb_no_impute.fit(X_train_no_impute, y_train_no_impute,
                            eval_set=[(X_val_no_impute, y_val_no_impute)],
                            eval_metric='rmse',
                            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    lgb_val_predictions_no_impute = model_lgb_no_impute.predict(X_val_no_impute)

    # Ensemble without imputation
    ensemble_val_predictions_no_impute = (cb_val_predictions_no_impute + lgb_val_predictions_no_impute) / 2
    ablation3_rmse = np.sqrt(mean_squared_error(y_val_no_impute, ensemble_val_predictions_no_impute))
    ablation_results['No Imputation (Dropped NaNs)'] = ablation3_rmse
    print(f"No Imputation (Dropped NaNs) Ensemble RMSE: {ablation3_rmse:.4f}\n")


# --- Summary of Ablation Study ---
print("\n--- Ablation Study Results ---")
for name, rmse in ablation_results.items():
    print(f"{name}: {rmse:.4f}")

# Determine contribution
best_rmse_overall = baseline_rmse
best_approach = 'Full Ensemble (Baseline)'

if ablation_results['CatBoost Only'] < best_rmse_overall:
    best_rmse_overall = ablation_results['CatBoost Only']
    best_approach = 'CatBoost Only'
if ablation_results['LightGBM Only'] < best_rmse_overall:
    best_rmse_overall = ablation_results['LightGBM Only']
    best_approach = 'LightGBM Only'
if 'No Imputation (Dropped NaNs)' in ablation_results and ablation_results['No Imputation (Dropped NaNs)'] < best_rmse_overall:
    best_rmse_overall = ablation_results['No Imputation (Dropped NaNs)']
    best_approach = 'No Imputation (Dropped NaNs)'

print("\n--- Contribution Analysis ---")
print(f"Baseline (Full Ensemble) RMSE: {baseline_rmse:.4f}")
print(f"CatBoost Only RMSE: {ablation_results['CatBoost Only']:.4f}")
print(f"LightGBM Only RMSE: {ablation_results['LightGBM Only']:.4f}")
if 'No Imputation (Dropped NaNs)' in ablation_results and ablation_results['No Imputation (Dropped NaNs)'] != float('inf'):
    print(f"No Imputation (Dropped NaNs) Ensemble RMSE: {ablation_results['No Imputation (Dropped NaNs)']:.4f}")


if baseline_rmse < ablation_results['CatBoost Only'] and baseline_rmse < ablation_results['LightGBM Only']:
    print("\nEnsembling both CatBoost and LightGBM models contributes positively to the overall performance, as the ensemble achieves a lower RMSE than either model individually.")
elif ablation_results['CatBoost Only'] < baseline_rmse and ablation_results['CatBoost Only'] < ablation_results['LightGBM Only']:
    print("\nCatBoost is the primary contributor, as it achieves the best performance even when run alone, outperforming the ensemble.")
elif ablation_results['LightGBM Only'] < baseline_rmse and ablation_results['LightGBM Only'] < ablation_results['CatBoost Only']:
    print("\nLightGBM is the primary contributor, as it achieves the best performance even when run alone, outperforming the ensemble.")
else:
    print("\nThe contributions of individual models are balanced, with the ensemble offering a slight improvement or similar performance to the best individual model.")

if 'No Imputation (Dropped NaNs)' in ablation_results and ablation_results['No Imputation (Dropped NaNs)'] != float('inf'):
    if ablation_results['No Imputation (Dropped NaNs)'] > baseline_rmse:
        print("\nMissing value imputation (median strategy) is a critical part of the solution, significantly improving performance compared to dropping rows with NaNs.")
    else:
        print("\nSkipping missing value imputation (by dropping rows) performed better or similarly, suggesting that the imputation strategy might not be optimal or the missing data is not detrimental.")
