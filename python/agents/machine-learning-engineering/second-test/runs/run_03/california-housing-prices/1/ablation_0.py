
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import warnings

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')

# Load data - ONLY training data for ablation study
train_df = pd.read_csv("./input/train.csv")

# Separate target variable
y_train_full = train_df['median_house_value']
X_train_full = train_df.drop('median_house_value', axis=1)

def run_experiment(X_data: pd.DataFrame, y_data: pd.Series, enable_feature_engineering: bool, imputation_strategy: str, description: str):
    """
    Runs a single experiment variation and returns its validation RMSE.
    """
    X_processed = X_data.copy()

    # --- Feature Engineering ---
    if enable_feature_engineering:
        # rooms_per_household
        X_processed['rooms_per_household'] = X_processed['total_rooms'] / X_processed['households']
        X_processed['rooms_per_household'].replace([np.inf, -np.inf], np.nan, inplace=True)

        # bedrooms_per_room
        X_processed['bedrooms_per_room'] = X_processed['total_bedrooms'] / X_processed['total_rooms']
        X_processed['bedrooms_per_room'].replace([np.inf, -np.inf], np.nan, inplace=True)

        # population_per_household
        X_processed['population_per_household'] = X_processed['population'] / X_processed['households']
        X_processed['population_per_household'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- Imputation ---
    imputer = SimpleImputer(strategy=imputation_strategy)
    numerical_cols = X_processed.select_dtypes(include=np.number).columns
    # Ensure all numerical columns are handled, even if they have no NaNs
    X_processed[numerical_cols] = imputer.fit_transform(X_processed[numerical_cols])


    # Validation split for model evaluation
    X_train, X_val, y_train, y_val = train_test_split(X_processed, y_data, test_size=0.2, random_state=42)

    # --- Model Training and Evaluation ---
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))

    print(f"[{description}] Validation RMSE: {rmse:.4f}")
    return rmse

print("--- Ablation Study Results ---")

# --- Baseline Experiment ---
baseline_rmse = run_experiment(
    X_train_full, y_train_full,
    enable_feature_engineering=True,
    imputation_strategy='median',
    description='Baseline (Full Feature Engineering, Median Imputation)'
)

# --- Ablation 1: Disable Feature Engineering ---
# This modification disables the creation of rooms_per_household, bedrooms_per_room, and population_per_household.
no_fe_rmse = run_experiment(
    X_train_full, y_train_full,
    enable_feature_engineering=False,
    imputation_strategy='median',
    description='Ablation 1 (No Feature Engineering, Median Imputation)'
)

# --- Ablation 2: Change Imputation Strategy to Mean ---
# This modification changes the SimpleImputer strategy from 'median' to 'mean'.
mean_impute_rmse = run_experiment(
    X_train_full, y_train_full,
    enable_feature_engineering=True,
    imputation_strategy='mean',
    description='Ablation 2 (Full Feature Engineering, Mean Imputation)'
)

print("\n--- Summary of Ablation Study ---")
print(f"Baseline RMSE (Full FE, Median Impute): {baseline_rmse:.4f}")
print(f"Ablation 1 RMSE (No FE, Median Impute): {no_fe_rmse:.4f}")
print(f"Ablation 2 RMSE (Full FE, Mean Impute): {mean_impute_rmse:.4f}")

# Calculate performance change from baseline for each ablation
loss_from_disabling_fe = no_fe_rmse - baseline_rmse
loss_from_changing_imputation = mean_impute_rmse - baseline_rmse

if loss_from_disabling_fe > 0:
    print(f"\nDisabling Feature Engineering increased RMSE by {loss_from_disabling_fe:.4f}.")
else:
    print(f"\nDisabling Feature Engineering decreased RMSE by {abs(loss_from_disabling_fe):.4f}.")

if loss_from_changing_imputation > 0:
    print(f"Changing Imputation Strategy from Median to Mean increased RMSE by {loss_from_changing_imputation:.4f}.")
else:
    print(f"Changing Imputation Strategy from Median to Mean decreased RMSE by {abs(loss_from_changing_imputation):.4f}.")

# Determine which part contributes most to the overall performance
# A larger positive 'loss' (increase in RMSE) indicates a greater positive contribution from the baseline component.
most_contributing_part_desc = ""
if loss_from_disabling_fe > loss_from_changing_imputation:
    if loss_from_disabling_fe > 0: # If disabling FE led to a significant increase in RMSE
        most_contributing_part_desc = "Feature Engineering (creating rooms_per_household, bedrooms_per_room, population_per_household)"
    else: # If disabling FE did not increase RMSE, or even decreased it.
        most_contributing_part_desc = "Neither Feature Engineering nor the specific choice of Imputation Strategy showed a dominant positive contribution."
elif loss_from_changing_imputation > loss_from_disabling_fe:
    if loss_from_changing_imputation > 0: # If changing imputation to mean led to a significant increase in RMSE
        most_contributing_part_desc = "The choice of Median Imputation strategy (over Mean Imputation)"
    else: # If changing imputation did not increase RMSE, or even decreased it.
        most_contributing_part_desc = "Neither Feature Engineering nor the specific choice of Imputation Strategy showed a dominant positive contribution."
else: # Losses are equal, or both non-positive/negative (i.e., no significant degradation from either ablation).
    most_contributing_part_desc = "Feature Engineering and the choice of Median Imputation strategy contributed similarly (or not significantly) to the performance, or their impact was minor in this specific comparison."

print(f"\nBased on this ablation study, {most_contributing_part_desc} appears to contribute most to the overall performance in this context.")
