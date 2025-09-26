
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from catboost import CatBoostRegressor
import os

# Define the input directory
input_dir = "./input"

# Ensure the input directory exists
if not os.path.exists(input_dir):
    print(f"Error: The input directory '{input_dir}' was not found.")
    # In a real scenario, you might want to raise an exception or exit here if input files are mandatory.

# Load the training data
train_file_path = os.path.join(input_dir, "train.csv")
try:
    train_df = pd.read_csv(train_file_path)
    # Strip any leading/trailing whitespace from column names
    train_df.columns = train_df.columns.str.strip()
except FileNotFoundError:
    print(f"Error: train.csv not found at {train_file_path}. Please ensure it's in the '{input_dir}' directory.")
    # Create an empty DataFrame to prevent NameError if the file isn't found
    train_df = pd.DataFrame()
except Exception as e:
    print(f"Error loading train.csv: {e}")
    train_df = pd.DataFrame() # Ensure train_df is defined

# Define features (X) and target (y)
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income']
target = 'median_house_value'

# Check if train_df is not empty and has the required columns before proceeding
if not train_df.empty and all(col in train_df.columns for col in features + [target]):
    X = train_df[features]
    y = train_df[target]

    # Preprocessing: Impute missing values in 'total_bedrooms' with the mean
    # It's important to fit the imputer only on the training data to prevent data leakage.
    mean_total_bedrooms = X['total_bedrooms'].mean()
    X['total_bedrooms'].fillna(mean_total_bedrooms, inplace=True)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize CatBoostRegressor with specified parameters
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        early_stopping_rounds=100,
        verbose=False  # Set to True for detailed training output
    )

    # Train the model
    # Using X_val and y_val as the evaluation set for early stopping
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))

    # Make predictions on the validation set
    y_pred_val = cat_model.predict(X_val)

    # Calculate Root Mean Squared Error (RMSE)
    final_validation_score = np.sqrt(mean_squared_error(y_val, y_pred_val))

    # Print the final validation RMSE
    print(f"Final Validation Performance: {final_validation_score}")

    # --- Predictions for the test.csv for submission ---

    # Load the test data
    test_file_path = os.path.join(input_dir, "test.csv")
    try:
        test_df = pd.read_csv(test_file_path)
        # Strip any leading/trailing whitespace from column names
        test_df.columns = test_df.columns.str.strip()
    except FileNotFoundError:
        print(f"Error: test.csv not found at {test_file_path}. Please ensure it's in the '{input_dir}' directory.")
        test_df = pd.DataFrame() # Create an empty DataFrame
    except Exception as e:
        print(f"Error loading test.csv: {e}")
        test_df = pd.DataFrame() # Ensure test_df is defined

    # Check if test_df is not empty and has the required features before making predictions
    if not test_df.empty and all(col in test_df.columns for col in features):
        # Preprocessing for the test set: Impute missing values in 'total_bedrooms' using the mean from the training data
        test_df_processed = test_df[features].copy() # Ensure we're only using the relevant features and create a copy
        test_df_processed['total_bedrooms'].fillna(mean_total_bedrooms, inplace=True)

        # Make predictions on the preprocessed test set
        test_predictions = cat_model.predict(test_df_processed)

        # Create submission DataFrame
        submission_df = pd.DataFrame({'median_house_value': test_predictions})

        # Save the submission file
        submission_df.to_csv('submission.csv', index=False)
    else:
        print("Test data is empty or missing required features. Skipping submission file generation.")
        # Create an empty submission file if test data is problematic
        pd.DataFrame({'median_house_value': []}).to_csv('submission.csv', index=False)

else:
    print("Training data is empty or missing required features. Cannot train model or generate predictions.")
    # Create an empty submission file if training data is problematic
    pd.DataFrame({'median_house_value': []}).to_csv('submission.csv', index=False)

