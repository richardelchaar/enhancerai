
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error
import numpy as np
from catboost import CatBoostRegressor
import os

# Define the input directory
input_dir = "./input"

# Ensure the input directory exists
if not os.path.exists(input_dir):
    print(f"Error: The input directory '{input_dir}' was not found.")

# Load the training data
train_file_path = os.path.join(input_dir, "train.csv")
try:
    train_df = pd.read_csv(train_file_path)
    # Strip any leading/trailing whitespace from column names
    train_df.columns = train_df.columns.str.strip()
except FileNotFoundError:
    print(f"Error: train.csv not found at {train_file_path}. Please ensure it's in the '{input_dir}' directory.")
    train_df = pd.DataFrame() # Create an empty DataFrame to prevent NameError
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
    # X_train, y_train will be used for hyperparameter tuning with CV.
    # X_val, y_val will be used as a final hold-out set for evaluation and early stopping of the best model.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize CatBoostRegressor with fixed parameters for the tuning process.
    # 'iterations' is set high; early stopping will be applied in the final training step.
    base_cat_model = CatBoostRegressor(
        iterations=2000, # A sufficiently large number of iterations for tuning
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=False,
        # 'early_stopping_rounds' is NOT included here, as RandomizedSearchCV does not directly support
        # passing eval_set to the estimator's fit method for each CV fold.
    )

    # Define the parameter distributions for RandomizedSearchCV.
    # The focus is on 'learning_rate', 'depth', and 'l2_leaf_reg' as per the improvement plan.
    param_distributions = {
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
        'depth': [4, 6, 8, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
    }

    # Set up K-fold cross-validation.
    # This will split X_train/y_train into 5 folds for internal validation during the search.
    cv_folds = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize RandomizedSearchCV.
    # 'scoring' is set to 'neg_root_mean_squared_error' because sklearn's search functions
    # maximize the score, and we want to minimize RMSE.
    # 'n_iter' controls the number of random parameter combinations to try.
    random_search = RandomizedSearchCV(
        estimator=base_cat_model,
        param_distributions=param_distributions,
        n_iter=30,  # Number of parameter settings to sample
        scoring='neg_root_mean_squared_error',
        cv=cv_folds,
        verbose=0, # Set to higher values (e.g., 1 or 2) for more detailed search output
        random_state=42,
        n_jobs=-1 # Use all available CPU cores for parallel processing
    )

    # Perform the randomized search on the training data (X_train, y_train).
    random_search.fit(X_train, y_train)

    # Train the final model with the best parameters found and re-introduce early stopping.
    # The 'eval_set' for early stopping will be the X_val, y_val split.
    final_cat_model = CatBoostRegressor(
        **random_search.best_params_, # Unpack the best parameters found by RandomizedSearchCV
        iterations=2000, # Max iterations for the final model, early stopping will determine actual iterations
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        early_stopping_rounds=100, # Re-introduce early stopping as per the plan
        verbose=False # Set to True for detailed training output
    )

    # Train the final model on the full X_train, y_train with X_val, y_val as the early stopping evaluation set.
    final_cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))

    # Make predictions on the validation set using the optimized and finally trained model.
    y_pred_val = final_cat_model.predict(X_val)

    # Calculate Root Mean Squared Error (RMSE) for the final validation score.
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

        # Make predictions on the preprocessed test set using the final_cat_model
        test_predictions = final_cat_model.predict(test_df_processed)

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
