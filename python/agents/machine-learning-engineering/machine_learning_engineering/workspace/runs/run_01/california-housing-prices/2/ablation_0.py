
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Store results for comparison
results = {}

# --- Baseline Solution ---
train_df = pd.read_csv("./input/train.csv")

X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Impute missing 'total_bedrooms' with the median from the training data
if 'total_bedrooms' in X.columns:
    median_total_bedrooms_train = X['total_bedrooms'].median()
    X['total_bedrooms'].fillna(median_total_bedrooms_train, inplace=True)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model_lightgbm = LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1, random_state=42)
model_xgboost = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)

model_lightgbm.fit(X_train, y_train)
model_xgboost.fit(X_train, y_train)

y_pred_val_lgbm = model_lightgbm.predict(X_val)
y_pred_val_xgboost = model_xgboost.predict(X_val)

y_pred_val_ensembled = (y_pred_val_lgbm + y_pred_val_xgboost) / 2
rmse_baseline = np.sqrt(mean_squared_error(y_val, y_pred_val_ensembled))
print(f"Baseline Validation RMSE: {rmse_baseline}")
results['Baseline'] = rmse_baseline


# --- Ablation 1: No total_bedrooms imputation ---
# Reload data to ensure a clean slate for the ablation
train_df_ablation1 = pd.read_csv("./input/train.csv")
X_ablation1 = train_df_ablation1.drop("median_house_value", axis=1)
y_ablation1 = train_df_ablation1["median_house_value"]

# Skipping the imputation step for total_bedrooms for this ablation.
# LightGBM and XGBoost models are generally capable of handling NaN values.

X_train_ablation1, X_val_ablation1, y_train_ablation1, y_val_ablation1 = train_test_split(X_ablation1, y_ablation1, test_size=0.2, random_state=42)

model_lightgbm_ablation1 = LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1, random_state=42)
model_xgboost_ablation1 = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)

model_lightgbm_ablation1.fit(X_train_ablation1, y_train_ablation1)
model_xgboost_ablation1.fit(X_train_ablation1, y_train_ablation1)

y_pred_val_lgbm_ablation1 = model_lightgbm_ablation1.predict(X_val_ablation1)
y_pred_val_xgboost_ablation1 = model_xgboost_ablation1.predict(X_val_ablation1)

y_pred_val_ensembled_ablation1 = (y_pred_val_lgbm_ablation1 + y_pred_val_xgboost_ablation1) / 2
rmse_ablation1 = np.sqrt(mean_squared_error(y_val_ablation1, y_pred_val_ensembled_ablation1))
print(f"Ablation (No total_bedrooms imputation) Validation RMSE: {rmse_ablation1}")
results['No Imputation'] = rmse_ablation1


# --- Ablation 2: LightGBM Only (No Ensembling) ---
# Reload data to ensure a clean slate for the ablation
train_df_ablation2 = pd.read_csv("./input/train.csv")
X_ablation2 = train_df_ablation2.drop("median_house_value", axis=1)
y_ablation2 = train_df_ablation2["median_house_value"]

# Apply original imputation for total_bedrooms to isolate the ensembling effect
if 'total_bedrooms' in X_ablation2.columns:
    median_total_bedrooms_train_ablation2 = X_ablation2['total_bedrooms'].median()
    X_ablation2['total_bedrooms'].fillna(median_total_bedrooms_train_ablation2, inplace=True)

X_train_ablation2, X_val_ablation2, y_train_ablation2, y_val_ablation2 = train_test_split(X_ablation2, y_ablation2, test_size=0.2, random_state=42)

model_lightgbm_ablation2 = LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1, random_state=42)
# XGBoost model is excluded, and thus ensembling is effectively removed in this ablation

model_lightgbm_ablation2.fit(X_train_ablation2, y_train_ablation2)

# Only LightGBM predictions are used
y_pred_val_ensembled_ablation2 = model_lightgbm_ablation2.predict(X_val_ablation2)
rmse_ablation2 = np.sqrt(mean_squared_error(y_val_ablation2, y_pred_val_ensembled_ablation2))
print(f"Ablation (LightGBM Only) Validation RMSE: {rmse_ablation2}")
results['LightGBM Only'] = rmse_ablation2


# --- Determine contribution ---
degradations = {}
# Calculate degradation for each ablation relative to the baseline
degradations['Imputation of total_bedrooms'] = results['No Imputation'] - results['Baseline']
degradations['Ensembling with XGBoost'] = results['LightGBM Only'] - results['Baseline']

# Find the ablation that caused the largest increase in RMSE (largest positive degradation)
max_degradation_key = None
max_degradation_value = -np.inf

for key, value in degradations.items():
    if value > max_degradation_value:
        max_degradation_value = value
        max_degradation_key = key

# Print the conclusion
if max_degradation_value > 0.001: # Use a small threshold to consider a degradation significant
    print(f"\nThe part of the code that contributes the most to the overall performance is: {max_degradation_key}.")
    print(f"Removing or altering this part resulted in an RMSE increase of approximately {max_degradation_value:.4f}.")
elif abs(max_degradation_value) <= 0.001:
    print("\nBased on this ablation study, the performance changes between the baseline and the modified versions are negligible.")
    print("This suggests that the ablated parts do not have a strong isolated positive or negative contribution in this context.")
else: # This scenario means one of the ablations actually improved performance
     min_rmse_key = min(results, key=results.get)
     print(f"\nSurprisingly, one or more ablations unexpectedly improved performance relative to the baseline.")
     print(f"The best performing setup was: {min_rmse_key} with RMSE: {results[min_rmse_key]:.4f}.")
     print(f"This indicates that the baseline might not be optimal, or the ablated parts have a negative contribution in the original setup.")

