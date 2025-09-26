

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

# Define file paths
TRAIN_FILE = "./input/train.csv"
TEST_FILE = "./input/test.csv"
SUBMISSION_FILE = "median_house_value.csv"

def train_predict_and_evaluate():
    """
    Loads data, trains two RandomForestRegressor models (one based on base solution's feature handling,
    one based on reference solution's feature handling), ensembles their predictions,
    evaluates them, and generates predictions for the test set.
    """
    # 1. Load the datasets
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # Separate target variable from features
    X = train_df.drop("median_house_value", axis=1)
    y = train_df["median_house_value"]

    # Identify numerical and categorical features dynamically
    # These are used for defining preprocessing steps for both models
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    # 2. Define Preprocessing Pipelines for both base and reference approaches

    # --- Preprocessing for Base Model Approach ---
    # This pipeline mirrors the base solution's behavior by only handling numerical features
    # and implicitly dropping categorical features.
    base_numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    base_preprocessor = ColumnTransformer(
        transformers=[
            ('num', base_numerical_transformer, numerical_features)
        ],
        remainder='drop' # Explicitly drop categorical columns, matching base solution's effect
    )

    # --- Preprocessing for Reference Model Approach ---
    # This pipeline incorporates both numerical and categorical feature handling
    # as defined in the reference solution.
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

    # 3. Define Models
    # Base Model: RandomForestRegressor with n_estimators=100 and n_jobs=-1 from base solution
    base_model_regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # Reference Model: RandomForestRegressor with default n_estimators and n_jobs=-1 for performance
    reference_model_regressor = RandomForestRegressor(random_state=42, n_jobs=-1)

    # 4. Create Full Pipelines by combining preprocessor and model
    base_pipeline = Pipeline(steps=[('preprocessor', base_preprocessor),
                                    ('regressor', base_model_regressor)])

    reference_pipeline = Pipeline(steps=[('preprocessor', reference_preprocessor),
                                         ('regressor', reference_model_regressor)])

    # 5. Split the training data into training and validation sets for evaluation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Train both models on the training split
    base_pipeline.fit(X_train, y_train)
    reference_pipeline.fit(X_train, y_train)

    # 7. Make predictions on the validation set using both models
    y_pred_base_val = base_pipeline.predict(X_val)
    y_pred_ref_val = reference_pipeline.predict(X_val)

    # 8. Ensemble validation predictions (simple average)
    y_pred_ensemble_val = (y_pred_base_val + y_pred_ref_val) / 2

    # 9. Calculate Root Mean Squared Error (RMSE) on the ensembled validation set
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_ensemble_val))
    print(f"Final Validation Performance: {rmse_val}")

    # 10. Train both pipelines on the full training data for final predictions on the test set
    # This step is crucial for Kaggle competitions to utilize all available training data
    base_pipeline.fit(X, y)
    reference_pipeline.fit(X, y)

    # 11. Make predictions on the actual test set using both models
    test_predictions_base = base_pipeline.predict(test_df)
    test_predictions_ref = reference_pipeline.predict(test_df)

    # 12. Ensemble test predictions
    test_predictions_ensemble = (test_predictions_base + test_predictions_ref) / 2

    # 13. Create submission file
    submission_df = pd.DataFrame({'median_house_value': test_predictions_ensemble})
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission file '{SUBMISSION_FILE}' created successfully.")

if __name__ == "__main__":
    train_predict_and_evaluate()

