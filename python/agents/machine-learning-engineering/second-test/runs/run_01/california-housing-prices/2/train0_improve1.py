
import pandas as pd
from sklearn.model_selection import KFold
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

    # K-Fold Cross-Validation setup
    n_splits = 5  # Number of folds for cross-validation. Adjust as needed.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Array to store Out-Of-Fold (OOF) predictions for the entire dataset X.
    # Each element of y will be predicted by a model that did not see it during training.
    oof_preds = np.zeros(len(y))

    # List to store Root Mean Squared Error (RMSE) for each fold.
    # This helps in understanding the model's performance stability across different data subsets.
    fold_rmses = []

    # List to store trained models for later prediction on the test set
    trained_cat_models = []

    # Iterate through each fold
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        # Split the data into training and validation sets for the current fold
        # Using .iloc ensures compatibility with pandas DataFrames/Series
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        # Initialize CatBoostRegressor with specified parameters.
        # A new model instance is created for each fold.
        cat_model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42, # Consistent random seed for reproducibility across folds
            early_stopping_rounds=100, # Stop training if validation metric doesn't improve
            verbose=False  # Set to True for detailed training output during each fold
        )

        # Train the model on the current fold's training data.
        # Early stopping is applied using the current fold's validation set.
        cat_model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold))
        trained_cat_models.append(cat_model)

        # Make predictions on the validation set of the current fold
        y_pred_val_fold = cat_model.predict(X_val_fold)

        # Store these predictions in the OOF array at the corresponding validation indices.
        oof_preds[val_index] = y_pred_val_fold

        # Calculate RMSE for the current fold and append it to the list.
        fold_rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_val_fold))
        fold_rmses.append(fold_rmse)

    # Calculate the final validation score using the overall Out-Of-Fold predictions.
    # This provides a more robust and unbiased estimate of the model's generalization performance
    # compared to a single train-validation split.
    final_validation_score = np.sqrt(mean_squared_error(y, oof_preds))

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

        # Make predictions on the preprocessed test set using the last trained model from the KFold loop
        # For a more robust approach, one would average predictions from all trained_cat_models.
        # However, to align with the original code's implicit intent of using a single model for submission
        # after the loop, we use the last one trained.
        if trained_cat_models:
            test_predictions = trained_cat_models[-1].predict(test_df_processed)
        else:
            # Fallback if no models were trained (e.g., due to data issues)
            test_predictions = np.zeros(len(test_df_processed))


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
