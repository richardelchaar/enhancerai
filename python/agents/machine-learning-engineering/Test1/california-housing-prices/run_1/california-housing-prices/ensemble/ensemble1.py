
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# --- Global Data Loading and Splitting ---
# Load datasets
train_df_global = pd.read_csv("./input/train.csv")
test_df_global = pd.read_csv("./input/test.csv")

# Separate target variable from features in the full training data
X_full_global = train_df_global.drop("median_house_value", axis=1)
y_full_global = train_df_global["median_house_value"]

# Split the full training data into training and validation sets ONCE
# This ensures that the validation set (y_val_global) is consistent for final evaluation.
X_train_global, X_val_global, y_train_global, y_val_global = train_test_split(
    X_full_global, y_full_global, test_size=0.2, random_state=42
)

# --- Solution 1 as a Function ---
def run_solution1(X_train_in, y_train_in, X_val_in, y_val_in, test_df_in):
    """
    Executes the logic of Solution 1, training models and producing
    ensembled validation and test predictions.
    """
    # Make copies to avoid modifying the global dataframes directly
    X_train_sol = X_train_in.copy()
    X_val_sol = X_val_in.copy()
    test_df_sol = test_df_in.copy()

    # Handle missing values for 'total_bedrooms'
    # Impute with the median from the training data (X_train_sol)
    median_total_bedrooms = X_train_sol['total_bedrooms'].median()
    X_train_sol['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)
    X_val_sol['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)
    test_df_sol['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

    # --- Model Training ---
    # 1. Initialize and Train LightGBM Regressor model
    lgbm_model = lgb.LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    lgbm_model.fit(X_train_sol, y_train_in)

    # 2. Initialize and Train XGBoost Regressor model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    xgb_model.fit(X_train_sol, y_train_in)

    # --- Prediction and Ensemble (Meta-learner) ---
    # Make predictions on the validation set
    y_pred_val_lgbm = lgbm_model.predict(X_val_sol)
    y_pred_val_xgb = xgb_model.predict(X_val_sol)

    # Stack the predictions for the meta-learner
    X_meta_val = np.column_stack((y_pred_val_lgbm, y_pred_val_xgb))

    # Initialize and train the meta-learner (Ridge Regressor)
    meta_model = Ridge(random_state=42)
    meta_model.fit(X_meta_val, y_val_in)

    # Use the trained meta-learner to make the final ensemble predictions on the validation set
    y_pred_val_ensemble_sol1 = meta_model.predict(X_meta_val)

    # Make predictions on the test set using the base models
    test_predictions_lgbm = lgbm_model.predict(test_df_sol)
    test_predictions_xgb = xgb_model.predict(test_df_sol)

    # Stack test predictions for the meta-learner
    X_meta_test = np.column_stack((test_predictions_lgbm, test_predictions_xgb))

    # Use the trained meta-learner to make final ensemble predictions on the test set
    test_predictions_ensemble_sol1 = meta_model.predict(X_meta_test)

    return y_pred_val_ensemble_sol1, test_predictions_ensemble_sol1


# --- Solution 2 as a Function ---
def run_solution2(X_train_in, y_train_in, X_val_in, y_val_in, X_full_in, y_full_in, test_df_in):
    """
    Executes the logic of Solution 2, training models and producing
    ensembled validation and test predictions.
    """
    # Make copies to avoid modifying the global dataframes directly
    X_train_sol = X_train_in.copy()
    X_val_sol = X_val_in.copy()
    X_full_sol = X_full_in.copy() # Need X_full for final test predictions as per original solution 2
    y_full_sol = y_full_in.copy()
    test_df_sol = test_df_in.copy()

    # Identify numerical and categorical features dynamically from the full training set
    numerical_features = X_full_sol.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_full_sol.select_dtypes(include='object').columns.tolist()

    # --- Preprocessing Pipelines for both base and reference approaches ---
    # Base Model Preprocessor (only numerical features, drops categorical)
    base_numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    base_preprocessor = ColumnTransformer(
        transformers=[
            ('num', base_numerical_transformer, numerical_features)
        ],
        remainder='drop'
    )

    # Reference Model Preprocessor (numerical and categorical handling)
    reference_numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
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

    # --- Define Models ---
    base_model_regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    reference_model_regressor = RandomForestRegressor(random_state=42, n_jobs=-1)

    # --- Create Full Pipelines ---
    base_pipeline = Pipeline(steps=[('preprocessor', base_preprocessor),
                                    ('regressor', base_model_regressor)])
    reference_pipeline = Pipeline(steps=[('preprocessor', reference_preprocessor),
                                         ('regressor', reference_model_regressor)])

    # --- Train models on the training split for validation predictions ---
    base_pipeline.fit(X_train_sol, y_train_in)
    reference_pipeline.fit(X_train_sol, y_train_in)

    # --- Make predictions on the validation set ---
    y_pred_base_val = base_pipeline.predict(X_val_sol)
    y_pred_ref_val = reference_pipeline.predict(X_val_sol)

    # --- Ensemble validation predictions (simple average) ---
    y_pred_ensemble_val_sol2 = (y_pred_base_val + y_pred_ref_val) / 2

    # --- Train both pipelines on the full training data for final test predictions ---
    base_pipeline.fit(X_full_sol, y_full_sol) # Train on full data for test predictions as per original solution 2
    reference_pipeline.fit(X_full_sol, y_full_sol)

    # --- Make predictions on the actual test set ---
    test_predictions_base = base_pipeline.predict(test_df_sol)
    test_predictions_ref = reference_pipeline.predict(test_df_sol)

    # --- Ensemble test predictions ---
    test_predictions_ensemble_sol2 = (test_predictions_base + test_predictions_ref) / 2

    return y_pred_ensemble_val_sol2, test_predictions_ensemble_sol2


# --- Execute both solutions and combine their outputs ---

# Run Solution 1
y_pred_val_ensemble_sol1, test_predictions_ensemble_sol1 = run_solution1(
    X_train_global, y_train_global, X_val_global, y_val_global, test_df_global
)

# Run Solution 2
y_pred_ensemble_val_sol2, test_predictions_ensemble_sol2 = run_solution2(
    X_train_global, y_train_global, X_val_global, y_val_global, X_full_global, y_full_global, test_df_global
)

# --- Combine Validation Predictions (Step 3 & 4 of Ensemble Plan) ---
final_validation_predictions = (y_pred_val_ensemble_sol1 + y_pred_ensemble_val_sol2) / 2
rmse_final_ensemble = np.sqrt(mean_squared_error(y_val_global, final_validation_predictions))

print(f"Final Validation Performance: {rmse_final_ensemble}")

# --- Combine Test Predictions for Submission (Step 5 & 6 of Ensemble Plan) ---
final_test_predictions = (test_predictions_ensemble_sol1 + test_predictions_ensemble_sol2) / 2

# Generate the Kaggle submission file
submission_df = pd.DataFrame({'median_house_value': final_test_predictions})
submission_df.to_csv('submission.csv', index=False)
