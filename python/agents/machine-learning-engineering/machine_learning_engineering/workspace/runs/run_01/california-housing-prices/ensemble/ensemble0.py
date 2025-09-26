
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import os

# Define file paths
TRAIN_FILE = "./input/train.csv"
TEST_FILE = "./input/test.csv"
SUBMISSION_FILE = "submission.csv"

def train_predict_and_evaluate():
    # Load data
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
    except FileNotFoundError:
        print(f"Error: Ensure '{TRAIN_FILE}' and '{TEST_FILE}' exist in the './input' directory.")
        return

    # Separate target variable
    X = train_df.drop("median_house_value", axis=1)
    y = train_df["median_house_value"]

    # Store test IDs if available, though not needed for this submission format
    # test_ids = test_df.index # Assuming index could serve as ID if explicit ID column was present

    # Combine for consistent preprocessing (feature engineering and imputation)
    # Using `pd.concat` for aligning columns and applying transformations uniformly
    all_data = pd.concat([X, test_df], ignore_index=True, sort=False)

    # Feature Engineering
    # These features often improve performance in housing datasets
    all_data['rooms_per_household'] = all_data['total_rooms'] / all_data['households']
    all_data['bedrooms_per_room'] = all_data['total_bedrooms'] / all_data['total_rooms']
    all_data['population_per_household'] = all_data['population'] / all_data['households']
    
    # Handle infinite values that might arise from division by zero in feature engineering
    # Replace inf with NaN, then imputation will handle them
    all_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Impute missing values (e.g., from 'total_bedrooms' or newly created features if households was 0)
    # Using 'median' strategy as it's robust to outliers
    imputer = SimpleImputer(strategy='median')
    
    # Fit imputer on the combined dataset to prevent data leakage and ensure consistency
    all_data_imputed = pd.DataFrame(imputer.fit_transform(all_data), columns=all_data.columns)

    # Split back into training and testing sets after preprocessing
    X_processed = all_data_imputed.iloc[:len(X)]
    test_processed = all_data_imputed.iloc[len(X):]

    # Model Training and Cross-Validation
    # Using RandomForestRegressor, a robust ensemble method
    # n_estimators: number of trees in the forest
    # random_state: for reproducibility
    # n_jobs: use all available cores for faster computation
    # max_features: fraction of features to consider at each split (often tuned)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_features=0.8)

    # K-Fold Cross-Validation for robust performance estimation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_processed, y)):
        X_train, X_val = X_processed.iloc[train_index], X_processed.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        rmse_scores.append(rmse)
        # print(f"Fold {fold+1} RMSE: {rmse}") # Uncomment for per-fold scores

    final_validation_score = np.mean(rmse_scores)
    print(f"Final Validation Performance: {final_validation_score}")

    # Make predictions on the preprocessed test set
    test_predictions = model.predict(test_processed)

    # Create submission file in the specified format
    submission_df = pd.DataFrame({'median_house_value': test_predictions})
    submission_df.to_csv(SUBMISSION_FILE, index=False, header=True) 
    
    print(f"Submission file '{SUBMISSION_FILE}' created successfully.")

if __name__ == "__main__":
    # Ensure the input directory exists for local testing/execution environments
    # In the actual competition environment, this directory is expected to be present
    # and contain the necessary CSV files.
    if not os.path.exists("./input"):
        print("Creating './input' directory. Please ensure 'train.csv' and 'test.csv' are placed inside.")
        os.makedirs("./input")
    
    train_predict_and_evaluate()
