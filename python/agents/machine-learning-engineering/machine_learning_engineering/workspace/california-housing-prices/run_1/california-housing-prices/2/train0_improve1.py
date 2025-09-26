
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # This import is not used in the corrected code, but was present in original
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import lightgbm as lgb # Import the lightgbm library

# Define file paths
TRAIN_FILE = "./input/train.csv"
TEST_FILE = "./input/test.csv"
SUBMISSION_FILE = "submission.csv" # Changed to a more standard name

def train_predict_and_evaluate():
    """
    Loads data, trains an LGBMRegressor model, evaluates it, and generates predictions for the test set.
    """
    # 1. Load the datasets
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # Separate target variable from features
    X = train_df.drop("median_house_value", axis=1)
    y = train_df["median_house_value"]

    # Identify numerical and categorical features dynamically
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    # 2. Define Preprocessing Pipeline
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

    # 3. Define Model
    # Reference Model: LGBMRegressor from the lightgbm library
    # Ensure lightgbm is installed: pip install lightgbm
    reference_model_regressor = lgb.LGBMRegressor(random_state=42, n_jobs=-1)

    # 4. Create Full Pipeline by combining preprocessor and model
    reference_pipeline = Pipeline(steps=[('preprocessor', reference_preprocessor),
                                         ('regressor', reference_model_regressor)])

    # 5. Split the training data into training and validation sets for evaluation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Train the refined model on the training split
    reference_pipeline.fit(X_train, y_train)

    # 7. Make predictions on the validation set using the refined model
    y_pred_val = reference_pipeline.predict(X_val)

    # 8. Calculate Root Mean Squared Error (RMSE) on the validation set
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    print(f"Final Validation Performance: {rmse_val}")

    # 9. Train the pipeline on the full training data for final predictions on the test set
    reference_pipeline.fit(X, y)

    # 10. Make predictions on the actual test set using the refined model
    test_predictions = reference_pipeline.predict(test_df)

    # 11. Create submission file
    submission_df = pd.DataFrame({'median_house_value': test_predictions})
    submission_df.to_csv(SUBMISSION_FILE, index=False)

    print(f"Submission file '{SUBMISSION_FILE}' created successfully.")

if __name__ == "__main__":
    train_predict_and_evaluate()
