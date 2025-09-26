
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import os

# Define file paths
TRAIN_FILE = "./input/train.csv"
TEST_FILE = "./input/test.csv"
SUBMISSION_FILE = "median_house_value.csv"

def train_predict_and_evaluate():
    """
    Loads data, trains a RandomForestRegressor model, evaluates it,
    and generates predictions for the test set.
    """
    # 1. Load the datasets
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
    except FileNotFoundError as e:
        print(f"Error: Dataset file not found. Please ensure '{TRAIN_FILE}' and '{TEST_FILE}' exist. {e}")
        return

    # Separate target variable from features
    X = train_df.drop("median_house_value", axis=1)
    y = train_df["median_house_value"]

    # Identify numerical columns for imputation
    numerical_cols = X.select_dtypes(include=np.number).columns

    # 2. Handle missing values
    # Impute missing values in 'total_bedrooms' and any other numerical columns using the median
    imputer = SimpleImputer(strategy='median')

    # Fit the imputer on the training data and transform both train and test features
    X_imputed = pd.DataFrame(imputer.fit_transform(X[numerical_cols]), columns=numerical_cols, index=X.index)
    test_imputed = pd.DataFrame(imputer.transform(test_df[numerical_cols]), columns=numerical_cols, index=test_df.index)

    # 3. Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # 4. Initialize and train a RandomForestRegressor model
    # Using default parameters with a fixed random_state for reproducibility
    # n_estimators can be tuned, but 100 is a reasonable starting point.
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 5. Make predictions on the validation set
    y_pred_val = model.predict(X_val)

    # 6. Calculate Root Mean Squared Error (RMSE) on the validation set
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    print(f"Final Validation Performance: {rmse_val}")

    # 7. Make predictions on the actual test set
    test_predictions = model.predict(test_imputed)

    # 8. Create submission file
    submission_df = pd.DataFrame({'median_house_value': test_predictions})
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission file '{SUBMISSION_FILE}' created successfully.")

if __name__ == "__main__":
    train_predict_and_evaluate()
