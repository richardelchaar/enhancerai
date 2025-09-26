

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


    # 2. Define Preprocessing Pipelines for the base approach

    # --- Preprocessing for Base Model Approach ---
    # This pipeline handles numerical features with median imputation.
    # Categorical features are implicitly dropped by 'remainder='drop''.
    base_numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    base_preprocessor = ColumnTransformer(
        transformers=[
            ('num', base_numerical_transformer, numerical_features)
        ],
        remainder='drop' # Explicitly drop categorical columns
    )

    # 3. Define Model
    # Base Model: RandomForestRegressor with n_estimators=100 and n_jobs=-1
    base_model_regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # 4. Create Full Pipeline by combining preprocessor and model
    base_pipeline = Pipeline(steps=[('preprocessor', base_preprocessor),
                                    ('regressor', base_model_regressor)])

    # 5. Split the training data into training and validation sets for evaluation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Train the base model on the training split
    base_pipeline.fit(X_train, y_train)

    # 7. Make predictions on the validation set using the base model
    y_pred_base_val = base_pipeline.predict(X_val)

    # 8. Calculate Root Mean Squared Error (RMSE) on the validation set
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_base_val))
    print(f"Final Validation Performance: {rmse_val}")

    # 9. Train the base pipeline on the full training data for final predictions on the test set
    # This step is crucial for Kaggle competitions to utilize all available training data
    base_pipeline.fit(X, y)

    # 10. Make predictions on the actual test set using the base model
    test_predictions_base = base_pipeline.predict(test_df)

    # 11. Create submission file
    submission_df = pd.DataFrame({'median_house_value': test_predictions_base})
    submission_df.to_csv(SUBMISSION_FILE, index=False)

    print(f"Submission file '{SUBMISSION_FILE}' created successfully.")

if __name__ == "__main__":
    train_predict_and_evaluate()

