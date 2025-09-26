
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Define file path for training data
TRAIN_FILE = "./input/train.csv"

def run_experiment(X_train, X_val, y_train, y_val, config):
    """
    Runs a single experiment based on the given configuration and returns validation RMSE.
    """
    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include='object').columns.tolist()

    pipelines_to_train_and_predict = []

    # --- Preprocessing for Base Model Approach (always uses median for numerical, drops categorical) ---
    base_numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    base_preprocessor = ColumnTransformer(
        transformers=[
            ('num', base_numerical_transformer, numerical_features)
        ],
        remainder='drop' # Explicitly drop categorical columns, matching base solution's effect
    )

    # --- Preprocessing for Reference Model Approach (can vary numerical imputer strategy) ---
    ref_numerical_imputer_strategy = config.get('ref_numerical_imputer_strategy', 'median')
    reference_numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=ref_numerical_imputer_strategy))
    ])

    reference_categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    reference_preprocessor = ColumnTransformer(
        transformers=[
            ('num', reference_numerical_transformer, numerical_features),
            ('cat', reference_categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # --- Models ---
    base_model_regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    reference_model_regressor = RandomForestRegressor(random_state=42, n_jobs=-1)

    if config['run_base_pipeline']:
        base_pipeline = Pipeline(steps=[('preprocessor', base_preprocessor),
                                        ('regressor', base_model_regressor)])
        pipelines_to_train_and_predict.append(base_pipeline)

    if config['run_reference_pipeline']:
        reference_pipeline = Pipeline(steps=[('preprocessor', reference_preprocessor),
                                             ('regressor', reference_model_regressor)])
        pipelines_to_train_and_predict.append(reference_pipeline)

    # Train pipelines
    for pipe in pipelines_to_train_and_predict:
        pipe.fit(X_train, y_train)

    # Make predictions on validation set
    all_predictions = []
    for pipe in pipelines_to_train_and_predict:
        all_predictions.append(pipe.predict(X_val))

    if len(all_predictions) > 1 and config['ensemble_predictions']:
        # Ensemble by averaging predictions if multiple pipelines and ensemble is enabled
        y_pred_final_val = np.mean(all_predictions, axis=0)
    elif len(all_predictions) == 1:
        # Use single pipeline's predictions
        y_pred_final_val = all_predictions[0]
    else: # Should not happen if at least one pipeline is configured to run
        raise ValueError("No predictions were generated. Check configuration.")

    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_final_val))
    return rmse_val

# Main script for ablation study
if __name__ == "__main__":
    # Ensure the input directory exists for the dummy data if needed
    os.makedirs(os.path.dirname(TRAIN_FILE), exist_ok=True)

    # 1. Load the datasets (or create dummy data if file is missing)
    if not os.path.exists(TRAIN_FILE):
        print(f"Warning: {TRAIN_FILE} not found. Creating dummy data for demonstration purposes.")
        # Create dummy data resembling housing dataset for demonstration
        dummy_data = {
            'longitude': np.random.uniform(-125, -114, 1000),
            'latitude': np.random.uniform(32, 42, 1000),
            'housing_median_age': np.random.randint(1, 52, 1000),
            'total_rooms': np.random.randint(10, 10000, 1000),
            'total_bedrooms': np.random.randint(1, 2000, 1000),
            'population': np.random.randint(3, 30000, 1000),
            'households': np.random.randint(1, 5000, 1000),
            'median_income': np.random.uniform(0.5, 15, 1000),
            'ocean_proximity': np.random.choice(['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'], 1000),
            'median_house_value': np.random.uniform(15000, 500000, 1000)
        }
        dummy_df = pd.DataFrame(dummy_data)
        # Introduce some NaNs to test imputation
        dummy_df.loc[::10, 'total_bedrooms'] = np.nan
        dummy_df.loc[::5, 'ocean_proximity'] = np.nan
        dummy_df.to_csv(TRAIN_FILE, index=False)
        print("Dummy train.csv created at ./input/train.csv.")

    train_df = pd.read_csv(TRAIN_FILE)

    # Separate target variable from features
    X = train_df.drop("median_house_value", axis=1)
    y = train_df["median_house_value"]

    # Split the training data into training and validation sets for evaluation
    X_train_full, X_val, y_train_full, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    # --- Baseline: Original Ensemble (Base + Reference Pipelines) ---
    print("Running Baseline: Original Ensemble (Base + Reference Pipelines)...")
    baseline_config = {
        'run_base_pipeline': True,
        'run_reference_pipeline': True,
        'ensemble_predictions': True,
        'ref_numerical_imputer_strategy': 'median' # Default for reference
    }
    baseline_rmse = run_experiment(X_train_full, X_val, y_train_full, y_val, baseline_config)
    results['Baseline Ensemble'] = baseline_rmse
    print(f"Baseline Ensemble RMSE: {baseline_rmse:.4f}\n")

    # --- Ablation 1: Only Base Pipeline (implicitly drops categorical features) ---
    print("Running Ablation 1: Only Base Pipeline (drops categorical features)...")
    ablation1_config = {
        'run_base_pipeline': True,
        'run_reference_pipeline': False,
        'ensemble_predictions': False,
        'ref_numerical_imputer_strategy': 'median' # Not used, but kept for config consistency
    }
    ablation1_rmse = run_experiment(X_train_full, X_val, y_train_full, y_val, ablation1_config)
    results['Only Base Pipeline'] = ablation1_rmse
    print(f"Only Base Pipeline RMSE: {ablation1_rmse:.4f}\n")

    # --- Ablation 2: Only Reference Pipeline (handles categorical features, median numerical imputation) ---
    print("Running Ablation 2: Only Reference Pipeline (handles categorical features)...")
    ablation2_config = {
        'run_base_pipeline': False,
        'run_reference_pipeline': True,
        'ensemble_predictions': False,
        'ref_numerical_imputer_strategy': 'median'
    }
    ablation2_rmse = run_experiment(X_train_full, X_val, y_train_full, y_val, ablation2_config)
    results['Only Reference Pipeline'] = ablation2_rmse
    print(f"Only Reference Pipeline RMSE: {ablation2_rmse:.4f}\n")

    # --- Ablation 3: Reference Pipeline with Numerical Imputer Strategy 'mean' instead of 'median' ---
    print("Running Ablation 3: Reference Pipeline (numerical imputer strategy='mean')...")
    ablation3_config = {
        'run_base_pipeline': False,
        'run_reference_pipeline': True,
        'ensemble_predictions': False,
        'ref_numerical_imputer_strategy': 'mean'
    }
    ablation3_rmse = run_experiment(X_train_full, X_val, y_train_full, y_val, ablation3_config)
    results['Reference Pipeline (mean imputer)'] = ablation3_rmse
    print(f"Reference Pipeline (mean imputer) RMSE: {ablation3_rmse:.4f}\n")

    print("\n--- Ablation Study Results ---")
    sorted_results = sorted(results.items(), key=lambda item: item[1])
    for name, rmse in sorted_results:
        print(f"{name}: RMSE = {rmse:.4f}")

    print("\n--- Contribution Analysis ---")
    baseline_rmse = results['Baseline Ensemble']

    # Identify the best performing configuration
    best_config_name = sorted_results[0][0]
    best_config_rmse = sorted_results[0][1]

    most_contributing_part_statement = ""

    if best_config_name == 'Baseline Ensemble':
        most_contributing_part_statement = (
            f"The ensemble of both the 'Base Pipeline' (numerical features only) and the "
            f"'Reference Pipeline' (numerical + categorical features) is the most significant "
            f"contributor to performance, achieving the lowest RMSE of {best_config_rmse:.4f}. "
            f"This suggests that combining the strengths of models trained with different feature "
            f"handling strategies is key."
        )
    elif best_config_name == 'Only Reference Pipeline':
        most_contributing_part_statement = (
            f"The 'Reference Pipeline' alone (which includes proper handling of categorical features "
            f"and numerical median imputation) is the most significant contributor to performance, "
            f"with an RMSE of {best_config_rmse:.4f}. This indicates that handling categorical features "
            f"is more impactful than the ensemble or simplifying to numerical-only features."
        )
    elif best_config_name == 'Only Base Pipeline':
        most_contributing_part_statement = (
            f"The 'Base Pipeline' alone (which focuses solely on numerical features with median imputation, "
            f"effectively dropping categorical features) is the most significant contributor to performance, "
            f"achieving an RMSE of {best_config_rmse:.4f}. This could suggest that categorical features "
            f"as processed by OneHotEncoder might introduce noise or are less predictive for this dataset."
        )
    elif best_config_name == 'Reference Pipeline (mean imputer)':
        most_contributing_part_statement = (
            f"The 'Reference Pipeline' with numerical features imputed by 'mean' is the most significant "
            f"contributor to performance, with an RMSE of {best_config_rmse:.4f}. This highlights that "
            f"the numerical imputation strategy ('mean' vs 'median') can be a critical factor for "
            f"this dataset."
        )
    else:
        most_contributing_part_statement = "An unexpected configuration was the best performer. Review the ablation results."

    print(most_contributing_part_statement)

