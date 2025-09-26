
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define actual column names from the problem description
TARGET_COLUMN = 'median_house_value'
ALL_NUMERICAL_FEATURES = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income'
]

# --- IMPORTANT: ADDING DUMMY FILE CREATION (AS A FALLBACK) ---
# The problem statement implies these files will be available. However,
# to make the script self-contained and runnable without manual file creation,
# we add a fallback to create dummy files with the correct structure
# if they are missing. This resolves the KeyError issue from the original problem,
# where the dummy files created by the original code had different column names
# than those expected by the solutions.
if not os.path.exists('./input'):
    os.makedirs('./input')

if not os.path.exists('./input/train.csv'):
    print("Creating dummy 'train.csv' as it does not exist (for demonstration/testing).")
    # Using the structure from the problem description
    dummy_train_data = {
        'longitude': np.random.uniform(-125, -114, 100),
        'latitude': np.random.uniform(32, 42, 100),
        'housing_median_age': np.random.randint(1, 52, 100),
        'total_rooms': np.random.randint(10, 6000, 100),
        'total_bedrooms': np.random.randint(1, 1200, 100),
        'population': np.random.randint(3, 3600, 100),
        'households': np.random.randint(1, 1100, 100),
        'median_income': np.random.uniform(0.5, 15, 100),
        'median_house_value': np.random.uniform(50000, 500000, 100)
    }
    pd.DataFrame(dummy_train_data).to_csv('./input/train.csv', index=False)

if not os.path.exists('./input/test.csv'):
    print("Creating dummy 'test.csv' as it does not exist (for demonstration/testing).")
    # Using the structure from the problem description
    dummy_test_data = {
        'longitude': np.random.uniform(-125, -114, 50),
        'latitude': np.random.uniform(32, 42, 50),
        'housing_median_age': np.random.randint(1, 52, 50),
        'total_rooms': np.random.randint(10, 6000, 50),
        'total_bedrooms': np.random.randint(1, 1200, 50),
        'population': np.random.randint(3, 3600, 50),
        'households': np.random.randint(1, 1100, 50),
        'median_income': np.random.uniform(0.5, 15, 50)
    }
    pd.DataFrame(dummy_test_data).to_csv('./input/test.csv', index=False)
# ---------------------------------------------------------------------

# Helper function to calculate RMSE as required by the metric
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Python Solution 1
def run_solution1():
    """
    Executes Solution 1 to generate predictions and saves them to 'solution1_predictions.csv'.
    Also calculates and returns a validation score (RMSE) for Solution 1.
    """
    print("Running Solution 1...")
    train_df = pd.read_csv('./input/train.csv')
    test_df = pd.read_csv('./input/test.csv')

    # Basic Linear Regression model using a subset of features.
    # Features are chosen to reflect the intent of the original code's 'feature1', 'feature2'.
    features = ['longitude', 'latitude', 'median_income', 'total_rooms']
    X_train = train_df[features]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[features]

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Save predictions with a temporary ID (test_df.index) for merging later.
    # The actual test.csv does not contain an 'ID' column.
    submission_df = pd.DataFrame({
        'temp_id': test_df.index,
        'Prediction': predictions
    })
    submission_df.to_csv('solution1_predictions.csv', index=False)
    print("Solution 1 predictions saved.")

    # Calculate validation performance for Solution 1
    X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model_val = LinearRegression()
    model_val.fit(X_train_val, y_train_val)
    val_preds = model_val.predict(X_test_val)
    return rmse(y_test_val, val_preds)

# Python Solution 2
def run_solution2():
    """
    Executes Solution 2 to generate predictions and saves them to 'solution2_predictions.csv'.
    Also calculates and returns a validation score (RMSE) for Solution 2.
    """
    print("Running Solution 2...")
    train_df = pd.read_csv('./input/train.csv')
    test_df = pd.read_csv('./input/test.csv')

    # Linear Regression with a different subset of features and a slight modification (noise)
    features = ['median_income', 'population', 'households', 'housing_median_age'] # Different feature set
    X_train = train_df[features]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[features]

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test) + np.random.normal(0, 0.75, len(X_test)) # Introduce some variation

    submission_df = pd.DataFrame({
        'temp_id': test_df.index,
        'Prediction': predictions
    })
    submission_df.to_csv('solution2_predictions.csv', index=False)
    print("Solution 2 predictions saved.")

    # Calculate validation performance for Solution 2
    X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model_val = LinearRegression()
    model_val.fit(X_train_val, y_train_val)
    val_preds = model_val.predict(X_test_val) + np.random.normal(0, 0.75, len(y_test_val)) # Match noise for validation
    return rmse(y_test_val, val_preds)

# Python Solution 3
def run_solution3():
    """
    Executes Solution 3 to generate predictions and saves them to 'solution3_predictions.csv'.
    Also calculates and returns a validation score (RMSE) for Solution 3.
    """
    print("Running Solution 3...")
    train_df = pd.read_csv('./input/train.csv')
    test_df = pd.read_csv('./input/test.csv')

    # Linear Regression using only one feature and a scaling factor
    features = ['median_income'] # A single, strong predictor (analog to 'feature1' in the original code)
    X_train = train_df[features]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[features]

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test) * 1.05 # Apply a scaling factor

    submission_df = pd.DataFrame({
        'temp_id': test_df.index,
        'Prediction': predictions
    })
    submission_df.to_csv('solution3_predictions.csv', index=False)
    print("Solution 3 predictions saved.")

    # Calculate validation performance for Solution 3
    X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model_val = LinearRegression()
    model_val.fit(X_train_val, y_train_val)
    val_preds = model_val.predict(X_test_val) * 1.05 # Match scaling for validation
    return rmse(y_test_val, val_preds)


def main():
    # 1. Execute Each Solution Individually
    val_score_s1 = run_solution1()
    val_score_s2 = run_solution2()
    val_score_s3 = run_solution3()

    print(f"\nIndividual Solution Validation Scores (RMSE):")
    print(f"Solution 1 RMSE: {val_score_s1:.4f}")
    print(f"Solution 2 RMSE: {val_score_s2:.4f}")
    print(f"Solution 3 RMSE: {val_score_s3:.4f}")

    # 2. Load Predictions
    print("\nLoading individual predictions for ensembling...")
    df_s1 = pd.read_csv('solution1_predictions.csv')
    df_s2 = pd.read_csv('solution2_predictions.csv')
    df_s3 = pd.read_csv('solution3_predictions.csv')

    # Merge DataFrames based on 'temp_id' (which is the original index of the test set)
    ensemble_df = df_s1.rename(columns={'Prediction': 'Prediction_S1'})
    ensemble_df = pd.merge(ensemble_df, df_s2.rename(columns={'Prediction': 'Prediction_S2'}), on='temp_id', how='inner')
    ensemble_df = pd.merge(ensemble_df, df_s3.rename(columns={'Prediction': 'Prediction_S3'}), on='temp_id', how='inner')
    print("Predictions merged.")

    # 3. Perform Median Aggregation
    ensemble_df['Prediction'] = ensemble_df[['Prediction_S1', 'Prediction_S2', 'Prediction_S3']].median(axis=1)
    print("Median aggregation performed.")

    # 4. Generate Final Submission
    # Adhering strictly to the submission format: no header, only median_house_value predictions
    final_submission_df = ensemble_df[['Prediction']].rename(columns={'Prediction': TARGET_COLUMN})
    final_submission_df.to_csv('final_ensemble_median_submission.csv', index=False, header=False)
    print("Final ensemble median submission saved to 'final_ensemble_median_submission.csv'.")

    # Calculate final ensemble validation performance on a hold-out set
    print("\nCalculating final ensemble validation performance...")
    train_df_full = pd.read_csv('./input/train.csv')
    
    # Define features used by each model for validation, consistent with run_solution functions
    features_s1_val = ['longitude', 'latitude', 'median_income', 'total_rooms']
    features_s2_val = ['median_income', 'population', 'households', 'housing_median_age']
    features_s3_val = ['median_income']

    # Create a superset of all features used across models for the main validation split
    all_validation_features = list(set(features_s1_val + features_s2_val + features_s3_val))

    X_full_for_split = train_df_full[all_validation_features]
    y_full = train_df_full[TARGET_COLUMN]

    # Use a common validation split for ensemble evaluation
    X_train_ens_split, X_val_ens_split, y_train_ens, y_val_ens = train_test_split(
        X_full_for_split, y_full, test_size=0.2, random_state=42
    )
    
    # Get predictions from each model on the ensemble validation set
    model1_val = LinearRegression()
    model1_val.fit(X_train_ens_split[features_s1_val], y_train_ens)
    val_preds_s1_ens = model1_val.predict(X_val_ens_split[features_s1_val])

    model2_val = LinearRegression()
    model2_val.fit(X_train_ens_split[features_s2_val], y_train_ens)
    val_preds_s2_ens = model2_val.predict(X_val_ens_split[features_s2_val]) + np.random.normal(0, 0.75, len(X_val_ens_split))

    model3_val = LinearRegression()
    model3_val.fit(X_train_ens_split[features_s3_val], y_train_ens)
    val_preds_s3_ens = model3_val.predict(X_val_ens_split[features_s3_val]) * 1.05

    # Aggregate ensemble predictions for validation
    ensemble_val_predictions = np.median([val_preds_s1_ens, val_preds_s2_ens, val_preds_s3_ens], axis=0)

    # Calculate the final ensemble validation score (RMSE)
    final_ensemble_val_score = rmse(y_val_ens, ensemble_val_predictions)
    
    print(f'Final Validation Performance: {final_ensemble_val_score}')

if __name__ == '__main__':
    main()
