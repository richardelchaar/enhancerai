

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from catboost import CatBoostRegressor
import os

# Define the input directory
input_dir = "./input"

# Load the training data
train_file_path = os.path.join(input_dir, "train.csv")
try:
    train_df = pd.read_csv(train_file_path)
    train_df.columns = train_df.columns.str.strip()
except FileNotFoundError:
    print(f"Error: train.csv not found at {train_file_path}. Please ensure it's in the '{input_dir}' directory.")
    exit()
except Exception as e:
    print(f"Error loading train.csv: {e}")
    exit()

# Define features (X) and target (y)
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income']
target = 'median_house_value'

# Check if train_df is not empty and has the required columns
if train_df.empty or not all(col in train_df.columns for col in features + [target]):
    print("Training data is empty or missing required features. Cannot perform ablation study.")
    exit()

# Create copies of the full dataset for each ablation to ensure independence
X_full = train_df[features].copy()
y_full = train_df[target].copy()

# Dictionary to store results of each ablation
ablation_results = {}

# --- Base Case: Original Solution ---
print("--- Running Base Case (Original Solution) ---")

X_base = X_full.copy()
y_base = y_full.copy()

# Preprocessing: Impute missing values in 'total_bedrooms' with the mean (ORIGINAL)
mean_total_bedrooms_base = X_base['total_bedrooms'].mean()
X_base['total_bedrooms'].fillna(mean_total_bedrooms_base, inplace=True)

# Split the data into training and validation sets
X_train_base, X_val_base, y_train_base, y_val_base = train_test_split(X_base, y_base, test_size=0.2, random_state=42)

# Initialize CatBoostRegressor with specified parameters
cat_model_base = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    early_stopping_rounds=100, # Original early stopping
    verbose=False
)

# Train the model
cat_model_base.fit(X_train_base, y_train_base, eval_set=(X_val_base, y_val_base))

# Make predictions on the validation set
y_pred_val_base = cat_model_base.predict(X_val_base)

# Calculate Root Mean Squared Error (RMSE)
rmse_base = np.sqrt(mean_squared_error(y_val_base, y_pred_val_base))
ablation_results['Base Case (Original)'] = rmse_base
print(f"Base Case Validation RMSE: {rmse_base:.4f}")


# --- Ablation 1: Using Median Imputation for 'total_bedrooms' ---
print("\n--- Running Ablation 1: Median Imputation for 'total_bedrooms' ---")

X_ablation1 = X_full.copy() # Start from original full dataset
y_ablation1 = y_full.copy()

# Preprocessing: Impute missing values in 'total_bedrooms' with the MEDIAN (ABLATION)
median_total_bedrooms_ablation1 = X_ablation1['total_bedrooms'].median()
X_ablation1['total_bedrooms'].fillna(median_total_bedrooms_ablation1, inplace=True)

# Split the data into training and validation sets
X_train_ablation1, X_val_ablation1, y_train_ablation1, y_val_ablation1 = train_test_split(X_ablation1, y_ablation1, test_size=0.2, random_state=42)

# Initialize CatBoostRegressor with original parameters (including early stopping)
cat_model_ablation1 = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    early_stopping_rounds=100,
    verbose=False
)

# Train the model
cat_model_ablation1.fit(X_train_ablation1, y_train_ablation1, eval_set=(X_val_ablation1, y_val_ablation1))

# Make predictions on the validation set
y_pred_val_ablation1 = cat_model_ablation1.predict(X_val_ablation1)

# Calculate RMSE
rmse_ablation1 = np.sqrt(mean_squared_error(y_val_ablation1, y_pred_val_ablation1))
ablation_results['Ablation 1 (Median Imputation)'] = rmse_ablation1
print(f"Ablation 1 (Median Imputation) Validation RMSE: {rmse_ablation1:.4f}")


# --- Ablation 2: Disabling Early Stopping ---
print("\n--- Running Ablation 2: No Early Stopping ---")

X_ablation2 = X_full.copy() # Start from original full dataset
y_ablation2 = y_full.copy()

# Preprocessing: Impute missing values in 'total_bedrooms' with the mean (ORIGINAL, consistent with base for this ablation)
mean_total_bedrooms_ablation2 = X_ablation2['total_bedrooms'].mean()
X_ablation2['total_bedrooms'].fillna(mean_total_bedrooms_ablation2, inplace=True)

# Split the data into training and validation sets
X_train_ablation2, X_val_ablation2, y_train_ablation2, y_val_ablation2 = train_test_split(X_ablation2, y_ablation2, test_size=0.2, random_state=42)

# Initialize CatBoostRegressor with specified parameters, NO EARLY STOPPING
cat_model_ablation2 = CatBoostRegressor(
    iterations=1000, # Will run for all iterations
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    # early_stopping_rounds removed
    verbose=False
)

# Train the model without eval_set for early stopping
cat_model_ablation2.fit(X_train_ablation2, y_train_ablation2) # eval_set removed

# Make predictions on the validation set
y_pred_val_ablation2 = cat_model_ablation2.predict(X_val_ablation2)

# Calculate RMSE
rmse_ablation2 = np.sqrt(mean_squared_error(y_val_ablation2, y_pred_val_ablation2))
ablation_results['Ablation 2 (No Early Stopping)'] = rmse_ablation2
print(f"Ablation 2 (No Early Stopping) Validation RMSE: {rmse_ablation2:.4f}")

# --- Summary and Contribution Analysis ---
print("\n--- Ablation Study Results Summary ---")
for ablation, rmse in ablation_results.items():
    print(f"- {ablation}: {rmse:.4f}")

# Determine the most impactful change by comparing against the base case
base_rmse = ablation_results['Base Case (Original)']

impacts = {}
for ablation, rmse in ablation_results.items():
    if ablation != 'Base Case (Original)':
        # Calculate the change in RMSE: positive means performance worsened, negative means it improved.
        impact = rmse - base_rmse
        impacts[ablation] = impact

print("\n--- Conclusion on Contribution ---")
if impacts:
    # Find the ablation that caused the largest increase in RMSE (worsening performance),
    # implying the original component was highly beneficial.
    most_degrading_ablation = max(impacts, key=impacts.get)
    max_degrading_impact = impacts[most_degrading_ablation]

    # Find the ablation that caused the largest decrease in RMSE (improving performance),
    # implying the original component might have been suboptimal.
    most_improving_ablation = min(impacts, key=impacts.get)
    max_improving_impact = impacts[most_improving_ablation]

    # Threshold for considering an impact significant
    significance_threshold = 0.001

    most_contributing_part_name = "None of the tested components significantly"

    if max_degrading_impact > significance_threshold:
        if "Median Imputation" in most_degrading_ablation:
            most_contributing_part_name = "Original Mean Imputation for 'total_bedrooms'"
        elif "No Early Stopping" in most_degrading_ablation:
            most_contributing_part_name = "Original Early Stopping mechanism"
        print(f"The modification in '{most_degrading_ablation}' (RMSE change: +{max_degrading_impact:.4f}) caused the largest degradation in performance.")
        print(f"This indicates that the '{most_contributing_part_name}' contributes the most positively to the overall performance of the original solution among the tested components.")
    elif abs(max_improving_impact) > significance_threshold:
        if "Median Imputation" in most_improving_ablation:
            most_contributing_part_name = "Original Mean Imputation for 'total_bedrooms'"
            print(f"The modification in '{most_improving_ablation}' (RMSE change: {max_improving_impact:.4f}) caused the largest improvement in performance.")
            print(f"This suggests that 'Median Imputation' is better than the original 'Mean Imputation' for 'total_bedrooms', implying the original component was suboptimal.")
        elif "No Early Stopping" in most_improving_ablation:
            most_contributing_part_name = "Original Early Stopping mechanism"
            print(f"The modification in '{most_improving_ablation}' (RMSE change: {max_improving_impact:.4f}) caused the largest improvement in performance.")
            print(f"This suggests that disabling early stopping (i.e., training for more iterations) improves performance, indicating the original early stopping might have been too aggressive or the model benefits from longer training.")
    else:
        print("The tested modifications had a minor impact on performance relative to the base case.")

else:
    print("No ablations were performed or recorded results for comparison.")

