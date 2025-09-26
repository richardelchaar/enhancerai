
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')

def get_preprocessor(X_df):
    """
    Creates and returns a ColumnTransformer based on the columns present in X_df.
    It dynamically identifies numerical and categorical columns.
    Based on the original solution's `categorical_cols = []`, it's assumed
    that there are no categorical features in the input data.
    """
    # Dynamically identify categorical columns; if no 'object' columns exist, this will be empty
    categorical_cols = [col for col in X_df.columns if X_df[col].dtype == 'object']
    numerical_cols = [col for col in X_df.columns if col not in categorical_cols]

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    transformers = [
        ('num', numerical_transformer, numerical_cols)
    ]

    # Only add categorical transformer if there are identified categorical columns
    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformers.append(('cat', categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )
    return preprocessor

def apply_feature_engineering(df):
    """Applies feature engineering as defined in the original solution."""
    df_copy = df.copy()
    df_copy['rooms_per_household'] = df_copy['total_rooms'] / df_copy['households']
    df_copy['bedrooms_per_room'] = df_copy['total_bedrooms'] / df_copy['total_rooms']
    df_copy['population_per_household'] = df_copy['population'] / df_copy['households']
    return df_copy

def run_ablation_experiment(X_train, y_train, preprocessor, use_ensemble=True):
    """
    Runs K-Fold cross-validation with the given data, preprocessor, and ensembling strategy.
    Returns the average RMSE across folds.
    """
    # Model parameters as defined in the original solution
    model_A_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1,
                      'max_features': 0.8, 'max_depth': 15, 'min_samples_leaf': 5}
    model_B_params = {'n_estimators': 150, 'random_state': 43, 'n_jobs': -1,
                      'max_features': 'sqrt', 'max_depth': 20, 'min_samples_leaf': 3}

    model_A_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(**model_A_params))
    ])

    model_B_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(**model_B_params))
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Fit Model A and predict
        model_A_pipeline.fit(X_train_fold, y_train_fold)
        y_pred_A = model_A_pipeline.predict(X_val_fold)

        # Handle ensembling or single model prediction
        if use_ensemble:
            model_B_pipeline.fit(X_train_fold, y_train_fold)
            y_pred_B = model_B_pipeline.predict(X_val_fold)
            ensemble_pred_val = (y_pred_A + y_pred_B) / 2
        else:
            ensemble_pred_val = y_pred_A # Only use Model A's predictions

        fold_rmse = np.sqrt(mean_squared_error(y_val_fold, ensemble_pred_val))
        rmse_scores.append(fold_rmse)

    return np.mean(rmse_scores)

if __name__ == "__main__":
    train_file = "./input/train.csv" # Ensure this path is correct for your environment
    train_df = pd.read_csv(train_file)

    y_train_raw = train_df['median_house_value']
    X_train_base_features = train_df.drop('median_house_value', axis=1)

    results = {}

    # --- Base Case: Original Solution (with Feature Engineering and Ensembling) ---
    X_train_fe_base = apply_feature_engineering(X_train_base_features)
    preprocessor_fe_base = get_preprocessor(X_train_fe_base)
    rmse_base = run_ablation_experiment(X_train_fe_base, y_train_raw, preprocessor_fe_base, use_ensemble=True)
    results['Base Case (FE + Ensemble)'] = rmse_base
    print(f"Base Case (Feature Engineering + Ensembling) RMSE: {rmse_base:.4f}")

    # --- Ablation 1: No Feature Engineering (but with Ensembling) ---
    X_train_no_fe = X_train_base_features.copy()
    preprocessor_no_fe = get_preprocessor(X_train_no_fe)
    rmse_no_fe = run_ablation_experiment(X_train_no_fe, y_train_raw, preprocessor_no_fe, use_ensemble=True)
    results['Ablation: No Feature Engineering'] = rmse_no_fe
    print(f"Ablation (No Feature Engineering + Ensembling) RMSE: {rmse_no_fe:.4f}")

    # --- Ablation 2: No Ensembling (but with Feature Engineering, using only Model A) ---
    # Reusing the feature-engineered data and preprocessor from the base case
    rmse_no_ensemble = run_ablation_experiment(X_train_fe_base, y_train_raw, preprocessor_fe_base, use_ensemble=False)
    results['Ablation: No Ensembling'] = rmse_no_ensemble
    print(f"Ablation (Feature Engineering + No Ensembling) RMSE: {rmse_no_ensemble:.4f}")

    print("\n--- Ablation Study Summary ---")
    for name, rmse in results.items():
        print(f"{name}: RMSE = {rmse:.4f}")

    # --- Contribution Analysis ---
    base_rmse = results['Base Case (FE + Ensemble)']
    impact_no_fe = results['Ablation: No Feature Engineering'] - base_rmse
    impact_no_ensemble = results['Ablation: No Ensembling'] - base_rmse

    print(f"\nChange in RMSE when removing Feature Engineering: {impact_no_fe:.4f}")
    print(f"Change in RMSE when removing Ensembling: {impact_no_ensemble:.4f}")

    # A positive change in RMSE indicates performance degradation when the component is removed.
    # The component whose removal causes the largest positive change (or smallest negative change)
    # is considered to contribute the most.
    if impact_no_fe > impact_no_ensemble:
        print("\nBased on this ablation study, Feature Engineering contributes more to the overall performance.")
    elif impact_no_ensemble > impact_no_fe:
        print("\nBased on this ablation study, Ensembling contributes more to the overall performance.")
    else:
        print("\nBased on this ablation study, both Feature Engineering and Ensembling contribute similarly (or their removal had a similar impact) to the overall performance.")
