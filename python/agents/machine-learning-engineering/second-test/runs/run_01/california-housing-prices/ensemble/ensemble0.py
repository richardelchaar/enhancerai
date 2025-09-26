
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
import os

# --- Python Solution 1 Start ---
# Load the datasets
try:
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
except FileNotFoundError:
    # If running locally without input/ directory, assume files are in current directory
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

# Identify features and target
TARGET_COL = 'median_house_value'
# FEATURES will be all numerical columns in train_df except TARGET_COL
FEATURES = [col for col in train_df.columns if col != TARGET_COL]

# Simple imputation for missing values in 'total_bedrooms'
# Calculate median from the training data to avoid data leakage
median_total_bedrooms_train = train_df['total_bedrooms'].median()
train_df['total_bedrooms'].fillna(median_total_bedrooms_train, inplace=True)
test_df['total_bedrooms'].fillna(median_total_bedrooms_train, inplace=True) # Use train median for test set

# Prepare data for models
X = train_df[FEATURES]
y = train_df[TARGET_COL]

# Split the training data into training and validation sets
# This helps evaluate the model's performance on unseen data before making final predictions
# Using random_state=42 ensures this split is deterministic and matches Solution 2's split for y_val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- LightGBM Model Training ---
# Initialize the LightGBM Regressor model
lgbm_model = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, n_jobs=-1)

# Train the LightGBM model
lgbm_model.fit(X_train, y_train)

# Make predictions on the validation set with LightGBM
y_val_pred_lgbm = lgbm_model.predict(X_val)

# Make predictions on the actual test set with LightGBM
test_predictions_lgbm = lgbm_model.predict(test_df[FEATURES])


# --- XGBoost Model Training ---
# Initialize the XGBoost Regressor model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_jobs=-1)

# Train the XGBoost model
xgb_model.fit(X_train, y_train)

# Make predictions on the validation set with XGBoost
y_val_pred_xgb = xgb_model.predict(X_val)

# Make predictions on the actual test set with XGBoost
test_predictions_xgb = xgb_model.predict(test_df[FEATURES])


# --- Ensembling Predictions ---
# Implement weighted average ensemble for validation predictions

best_weight_lgbm = 0.5  # Initialize with simple average weight
min_rmse_val_ensemble = float('inf')
y_val_pred_ensemble = None # Initialize to store the best ensemble predictions

# Grid search for optimal weight for LightGBM, with XGBoost weight being (1 - weight_lgbm)
# Searching in steps of 0.01 from 0.0 to 1.0 (inclusive)
# This small grid search determines the optimal blend
weights_to_try = np.arange(0.0, 1.01, 0.01)

for w_lgbm in weights_to_try:
    w_xgb = 1.0 - w_lgbm
    current_y_val_pred_ensemble = (w_lgbm * y_val_pred_lgbm) + (w_xgb * y_val_pred_xgb)
    rmse_current = np.sqrt(mean_squared_error(y_val, current_y_val_pred_ensemble))

    if rmse_current < min_rmse_val_ensemble:
        min_rmse_val_ensemble = rmse_current
        best_weight_lgbm = w_lgbm
        y_val_pred_ensemble = current_y_val_pred_ensemble # Store the predictions for the best weight

best_weight_xgb = 1.0 - best_weight_lgbm

# Evaluate the ensembled model using Root Mean Squared Error (RMSE) on the validation set
# The rmse_val_ensemble is already the minimum found during the grid search
rmse_val_ensemble = min_rmse_val_ensemble

# Print the validation performance of the ensemble with optimal weights
print(f"Solution 1 Validation Performance (Weighted Ensemble): {rmse_val_ensemble:.6f}")
print(f"Solution 1 Optimal Weights - LightGBM: {best_weight_lgbm:.2f}, XGBoost: {best_weight_xgb:.2f}")

# Apply the optimal weights to the test predictions
final_test_predictions = (best_weight_lgbm * test_predictions_lgbm) + \
                         (best_weight_xgb * test_predictions_xgb)

# Create the submission file (commented out as per ensemble plan to prevent premature submission)
# submission_df = pd.DataFrame({'median_house_value': final_test_predictions})
# submission_df.to_csv('submission.csv', index=False)

# --- Capture Solution 1 outputs for meta-ensembling ---
np.save('sol1_test_predictions.npy', final_test_predictions)
np.save('sol1_val_predictions.npy', y_val_pred_ensemble)
np.save('y_val_true.npy', y_val) # y_val needs to be captured only once
# --- Python Solution 1 End ---


# --- Python Solution 2 Start ---

# Define the input directory
input_dir = "./input" # Re-using input_dir, consistent with Solution 1

# Ensure the input directory exists
if not os.path.exists(input_dir):
    print(f"Error: The input directory '{input_dir}' was not found.")

# Load the training data
train_file_path = os.path.join(input_dir, "train.csv")
try:
    # Use original variable name; it will overwrite Solution 1's train_df, but that's fine
    # as Solution 1's data has already been processed and its outputs saved.
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
# These are the same numerical features as Solution 1
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income']
target = 'median_house_value' # Same target as Solution 1

# Check if train_df is not empty and has the required columns before proceeding
if not train_df.empty and all(col in train_df.columns for col in features + [target]):
    # These X, y will overwrite Solution 1's X, y. This is acceptable.
    X = train_df[features]
    y = train_df[target]

    # Preprocessing: Impute missing values in 'total_bedrooms' with the mean
    # Note: Solution 1 used median, Solution 2 uses mean. This difference in preprocessing
    # for a single column is acceptable as per the plan (no standardization of preprocessing required).
    mean_total_bedrooms = X['total_bedrooms'].mean()
    X['total_bedrooms'].fillna(mean_total_bedrooms, inplace=True)

    # Split the data into training and validation sets
    # These X_train, X_val, y_train, y_val will overwrite Solution 1's.
    # CRITICAL: Same random_state=42 and same base X, y (same features and target) means
    # X_val and y_val are identical to Solution 1's, which is required for the meta-ensemble.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize CatBoostRegressor with fixed parameters for the tuning process.
    base_cat_model = CatBoostRegressor(
        iterations=2000, # A sufficiently large number of iterations for tuning
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=False,
    )

    # Define the parameter distributions for RandomizedSearchCV.
    param_distributions = {
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
        'depth': [4, 6, 8, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
    }

    # Set up K-fold cross-validation.
    cv_folds = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize RandomizedSearchCV.
    random_search = RandomizedSearchCV(
        estimator=base_cat_model,
        param_distributions=param_distributions,
        n_iter=30,  # Number of parameter settings to sample
        scoring='neg_root_mean_squared_error',
        cv=cv_folds,
        verbose=0,
        random_state=42,
        n_jobs=-1
    )

    # Perform the randomized search on the training data (X_train, y_train).
    random_search.fit(X_train, y_train)

    # Train the final model with the best parameters found and re-introduce early stopping.
    final_cat_model = CatBoostRegressor(
        **random_search.best_params_,
        iterations=2000,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        early_stopping_rounds=100,
        verbose=False
    )

    # Train the final model on the full X_train, y_train with X_val, y_val as the early stopping evaluation set.
    final_cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))

    # Make predictions on the validation set using the optimized and finally trained model.
    y_pred_val = final_cat_model.predict(X_val)

    # Calculate Root Mean Squared Error (RMSE) for the final validation score.
    final_validation_score = np.sqrt(mean_squared_error(y_val, y_pred_val))

    # Print the final validation RMSE
    print(f"Solution 2 Final Validation Performance: {final_validation_score}")

    # --- Predictions for the test.csv for submission ---

    # Load the test data
    test_file_path = os.path.join(input_dir, "test.csv")
    try:
        # Use original variable name; it will overwrite Solution 1's test_df.
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
        # This will use the latest test_df (from Solution 2 load).
        test_df_processed = test_df[features].copy() # Ensure we're only using the relevant features and create a copy
        test_df_processed['total_bedrooms'].fillna(mean_total_bedrooms, inplace=True)

        # Make predictions on the preprocessed test set using the final_cat_model
        test_predictions = final_cat_model.predict(test_df_processed)

        # Create submission DataFrame (commented out as per ensemble plan to prevent premature submission)
        # submission_df = pd.DataFrame({'median_house_value': test_predictions})
        # submission_df.to_csv('submission.csv', index=False)

        # --- Capture Solution 2 outputs for meta-ensembling ---
        np.save('sol2_test_predictions.npy', test_predictions)
        np.save('sol2_val_predictions.npy', y_pred_val)
        # --- End Capture Solution 2 outputs ---

    else:
        print("Test data is empty or missing required features. Skipping submission file generation for Solution 2.")
        # Create dummy empty files if test data is problematic for consistency
        np.save('sol2_test_predictions.npy', np.array([]))
        np.save('sol2_val_predictions.npy', np.array([]))

else:
    print("Training data is empty or missing required features. Cannot train model or generate predictions for Solution 2.")
    # Create dummy empty files if training data is problematic for consistency
    np.save('sol2_test_predictions.npy', np.array([]))
    np.save('sol2_val_predictions.npy', np.array([]))
# --- Python Solution 2 End ---


# --- New Ensemble Script Start ---

# Load the saved prediction arrays
sol1_test_predictions = np.load('sol1_test_predictions.npy')
sol1_val_predictions = np.load('sol1_val_predictions.npy')
sol2_test_predictions = np.load('sol2_test_predictions.npy')
sol2_val_predictions = np.load('sol2_val_predictions.npy')
y_val_true = np.load('y_val_true.npy')

# Clean up temporary .npy files
try:
    os.remove('sol1_test_predictions.npy')
    os.remove('sol1_val_predictions.npy')
    os.remove('sol2_test_predictions.npy')
    os.remove('sol2_val_predictions.npy')
    os.remove('y_val_true.npy')
except OSError as e:
    # This might happen if previous steps failed to create the files
    print(f"Could not remove all temporary files: {e}")

# Ensure loaded arrays are not empty, especially if previous steps had issues
if len(sol1_test_predictions) == 0 or len(sol1_val_predictions) == 0 or \
   len(sol2_test_predictions) == 0 or len(sol2_val_predictions) == 0 or \
   len(y_val_true) == 0:
    print("Error: One or more prediction/true label arrays are empty. Cannot perform ensemble.")
    # Create an empty submission file if ensemble cannot be performed
    pd.DataFrame({'median_house_value': []}).to_csv('submission.csv', index=False)
else:
    # Optimize Meta-Ensemble Weights
    best_w_sol1 = 0.0
    min_rmse_meta_ensemble = float('inf')
    weights_to_try_meta = np.arange(0.0, 1.01, 0.01) # Iterate from 0.0 to 1.0 in 0.01 steps

    for w_sol1 in weights_to_try_meta:
        w_sol2 = 1.0 - w_sol1
        # Ensure array shapes are compatible, though they should be due to deterministic splitting
        if sol1_val_predictions.shape != y_val_true.shape or sol2_val_predictions.shape != y_val_true.shape:
             print("Warning: Validation prediction shapes do not match true labels. Skipping ensemble optimization.")
             min_rmse_meta_ensemble = np.nan # Indicate error
             break # Exit loop
        
        combined_val_preds = (w_sol1 * sol1_val_predictions) + (w_sol2 * sol2_val_predictions)
        rmse_current_meta = np.sqrt(mean_squared_error(y_val_true, combined_val_preds))

        if rmse_current_meta < min_rmse_meta_ensemble:
            min_rmse_meta_ensemble = rmse_current_meta
            best_w_sol1 = w_sol1

    best_w_sol2 = 1.0 - best_w_sol1

    # Generate Final Test Predictions
    if not np.isnan(min_rmse_meta_ensemble): # Only proceed if optimization was successful
        # Ensure test prediction shapes are compatible before combining
        if sol1_test_predictions.shape != sol2_test_predictions.shape:
            print("Warning: Test prediction shapes do not match. Cannot generate final ensemble submission.")
            pd.DataFrame({'median_house_value': []}).to_csv('submission.csv', index=False) # Empty submission
        else:
            final_ensemble_predictions = (best_w_sol1 * sol1_test_predictions) + \
                                         (best_w_sol2 * sol2_test_predictions)

            # Save Final Submission
            submission_df_final = pd.DataFrame({'median_house_value': final_ensemble_predictions})
            submission_df_final.to_csv('submission.csv', index=False)

            # Print the final validation performance of the meta-ensemble
            print(f"Final Validation Performance: {min_rmse_meta_ensemble:.6f}")
            print(f"Optimal Meta-Ensemble Weights - Solution 1: {best_w_sol1:.2f}, Solution 2: {best_w_sol2:.2f}")
    else:
        print("Meta-ensemble optimization failed. No final submission generated.")
        pd.DataFrame({'median_house_value': []}).to_csv('submission.csv', index=False) # Empty submission

# --- New Ensemble Script End ---
