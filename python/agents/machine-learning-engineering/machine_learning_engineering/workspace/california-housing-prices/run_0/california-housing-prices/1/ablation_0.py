
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Data Loading and Preprocessing ---

# Load the training dataset
try:
    train_df = pd.read_csv('./input/train.csv')
except FileNotFoundError:
    print("train.csv not found. Creating dummy data for demonstration purposes.")
    # Create dummy dataframe if file not found, ensuring it matches expected columns
    data = {
        'longitude': np.random.rand(1000) * -20 - 110,
        'latitude': np.random.rand(1000) * 10 + 30,
        'housing_median_age': np.random.randint(1, 50, 1000),
        'total_rooms': np.random.randint(100, 6000, 1000),
        'total_bedrooms': np.random.randint(50, 1200, 1000),
        'population': np.random.randint(100, 5000, 1000),
        'households': np.random.randint(50, 1000, 1000),
        'median_income': np.random.rand(1000) * 10,
        'median_house_value': np.random.rand(1000) * 500000
    }
    train_df = pd.DataFrame(data)
    # Introduce some NaN values in total_bedrooms for testing imputation
    train_df.loc[train_df.sample(frac=0.05, random_state=42).index, 'total_bedrooms'] = np.nan


# Separate target variable
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Preprocessing: Handle missing values (as per original solution)
# Impute missing 'total_bedrooms' with the median
median_total_bedrooms = X['total_bedrooms'].median()
X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary to store RMSE results for comparison
results = {}

print("--- Starting Ablation Study ---")

# --- Baseline: Original Solution (LGBM + XGBoost Ensemble) ---
print("\nScenario 1: Baseline (LGBM + XGBoost Ensemble)")
lgbm_model_baseline = lgb.LGBMRegressor(random_state=42)
lgbm_model_baseline.fit(X_train, y_train)
y_pred_lgbm_baseline = lgbm_model_baseline.predict(X_val)

xgb_model_baseline = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model_baseline.fit(X_train, y_train)
y_pred_xgb_baseline = xgb_model_baseline.predict(X_val)

y_pred_ensemble_baseline = (y_pred_lgbm_baseline + y_pred_xgb_baseline) / 2
rmse_baseline = np.sqrt(mean_squared_error(y_val, y_pred_ensemble_baseline))
results['Baseline (LGBM + XGBoost Ensemble)'] = rmse_baseline
print(f'Performance (RMSE): {rmse_baseline:.4f}')

# --- Ablation 1: Only LightGBM Model ---
# This disables the XGBoost model from the ensemble
print("\nScenario 2: Ablation (LightGBM Only)")
lgbm_model_ablation1 = lgb.LGBMRegressor(random_state=42) # Re-instantiate for clarity, though not strictly necessary
lgbm_model_ablation1.fit(X_train, y_train)
y_pred_lgbm_ablation1 = lgbm_model_ablation1.predict(X_val)
rmse_ablation1 = np.sqrt(mean_squared_error(y_val, y_pred_lgbm_ablation1))
results['Ablation (LightGBM Only)'] = rmse_ablation1
print(f'Performance (RMSE): {rmse_ablation1:.4f}')

# --- Ablation 2: Only XGBoost Model ---
# This disables the LightGBM model from the ensemble
print("\nScenario 3: Ablation (XGBoost Only)")
xgb_model_ablation2 = xgb.XGBRegressor(objective='reg:squarederror', random_state=42) # Re-instantiate
xgb_model_ablation2.fit(X_train, y_train)
y_pred_xgb_ablation2 = xgb_model_ablation2.predict(X_val)
rmse_ablation2 = np.sqrt(mean_squared_error(y_val, y_pred_xgb_ablation2))
results['Ablation (XGBoost Only)'] = rmse_ablation2
print(f'Performance (RMSE): {rmse_ablation2:.4f}')

print("\n--- Ablation Study Summary ---")
for name, rmse in results.items():
    print(f"- {name}: RMSE = {rmse:.4f}")

# Determine which part contributes the most based on the best performing scenario (lowest RMSE)
best_performance_name = min(results, key=results.get)
best_performance_rmse = results[best_performance_name]

if best_performance_name == 'Baseline (LGBM + XGBoost Ensemble)':
    contribution = "the ensembling strategy (combining LightGBM and XGBoost)"
elif best_performance_name == 'Ablation (LightGBM Only)':
    contribution = "the LightGBM model alone"
else: # best_performance_name == 'Ablation (XGBoost Only)'
    contribution = "the XGBoost model alone"

print(f"\nBased on this ablation study, {contribution} contributes the most to the overall performance, achieving the lowest RMSE of {best_performance_rmse:.4f}.")
