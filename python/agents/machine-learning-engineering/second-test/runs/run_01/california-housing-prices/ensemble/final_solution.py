

import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression # For meta-learner
import os

# --- Helper function for loading data safely ---
def load_data(filename, input_dir="./input"):
    """
    Loads a CSV file from the specified input directory or current directory.
    Strips whitespace from column names.
    """
    # Attempt to load from input_dir first
    file_path_input = os.path.join(input_dir, filename)
    if os.path.exists(file_path_input):
        df = pd.read_csv(file_path_input)
        df.columns = df.columns.str.strip()
        return df
    # Fallback to current directory
    elif os.path.exists(filename):
        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip()
        return df
    else:
        # If neither path works, return an empty DataFrame and print an error
        print(f"Error: {filename} not found in '{input_dir}' or current directory.")
        return pd.DataFrame()

# Create a directory to store intermediate predictions for meta-learning
PREDS_DIR = "ensemble_preds"
os.makedirs(PREDS_DIR, exist_ok=True)


# --- Solution 1 Logic ---
print("--- Running Solution 1 ---")
train_df_s1 = load_data("train.csv")
test_df_s1 = load_data("test.csv")

# Initialize placeholders for Solution 1 outputs in case of data loading issues
y_val_s1 = np.array([])
y_val_pred_lgbm_s1 = np.array([])
y_val_pred_xgb_s1 = np.array([])
test_predictions_lgbm_s1 = np.array([])
test_predictions_xgb_s1 = np.array([])

if train_df_s1.empty or test_df_s1.empty:
    print("Skipping Solution 1 due to data loading issues.")
else:
    TARGET_COL_S1 = 'median_house_value'
    # Ensure all columns required by Solution 1 are present
    required_cols_s1 = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                        'total_bedrooms', 'population', 'households', 'median_income', TARGET_COL_S1]
    if not all(col in train_df_s1.columns for col in required_cols_s1):
        print("Missing required columns for Solution 1 in train.csv. Skipping Solution 1.")
    else:
        FEATURES_S1 = [col for col in train_df_s1.columns if col != TARGET_COL_S1]

        # Simple imputation for missing values in 'total_bedrooms' (Solution 1's logic)
        median_total_bedrooms_train_s1 = train_df_s1['total_bedrooms'].median()
        train_df_s1['total_bedrooms'].fillna(median_total_bedrooms_train_s1, inplace=True)
        test_df_s1['total_bedrooms'].fillna(median_total_bedrooms_train_s1, inplace=True)

        X_s1 = train_df_s1[FEATURES_S1]
        y_s1 = train_df_s1[TARGET_COL_S1]

        # Split the training data into training and validation sets
        X_train_s1, X_val_s1, y_train_s1, y_val_s1 = train_test_split(X_s1, y_s1, test_size=0.2, random_state=42)

        # --- LightGBM Model Training ---
        lgbm_model_s1 = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, n_jobs=-1)
        lgbm_model_s1.fit(X_train_s1, y_train_s1)
        y_val_pred_lgbm_s1 = lgbm_model_s1.predict(X_val_s1)
        test_predictions_lgbm_s1 = lgbm_model_s1.predict(test_df_s1[FEATURES_S1])

        # --- XGBoost Model Training ---
        xgb_model_s1 = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_jobs=-1)
        xgb_model_s1.fit(X_train_s1, y_train_s1)
        y_val_pred_xgb_s1 = xgb_model_s1.predict(X_val_s1)
        test_predictions_xgb_s1 = xgb_model_s1.predict(test_df_s1[FEATURES_S1])

        # Save Solution 1 predictions for meta-learning
        np.save(os.path.join(PREDS_DIR, 'y_val_s1.npy'), y_val_s1)
        np.save(os.path.join(PREDS_DIR, 'y_val_pred_lgbm_s1.npy'), y_val_pred_lgbm_s1)
        np.save(os.path.join(PREDS_DIR, 'y_val_pred_xgb_s1.npy'), y_val_pred_xgb_s1)
        np.save(os.path.join(PREDS_DIR, 'test_predictions_lgbm_s1.npy'), test_predictions_lgbm_s1)
        np.save(os.path.join(PREDS_DIR, 'test_predictions_xgb_s1.npy'), test_predictions_xgb_s1)
        print("Solution 1 predictions saved.")

        # The original submission generation part of Solution 1 is removed as per the ensemble plan.


# --- Solution 2 Logic ---
print("\n--- Running Solution 2 ---")
train_df_s2 = load_data("train.csv")
test_df_s2 = load_data("test.csv")

features_s2 = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income']
target_s2 = 'median_house_value'

# Initialize placeholders for Solution 2 outputs in case of errors
y_pred_val_catboost_s2 = np.array([])
test_predictions_catboost_s2 = np.array([])


if not train_df_s2.empty and all(col in train_df_s2.columns for col in features_s2 + [target_s2]):
    X_s2 = train_df_s2[features_s2]
    y_s2 = train_df_s2[target_s2]

    # Preprocessing: Impute missing values in 'total_bedrooms' with the mean (Solution 2's logic)
    mean_total_bedrooms_s2 = X_s2['total_bedrooms'].mean()
    X_s2['total_bedrooms'].fillna(mean_total_bedrooms_s2, inplace=True)

    # Split the data into training and validation sets
    X_train_s2, X_val_s2, y_train_s2, y_val_s2 = train_test_split(X_s2, y_s2, test_size=0.2, random_state=42)

    base_cat_model_s2 = CatBoostRegressor(
        iterations=2000,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=False,
    )

    param_distributions_s2 = {
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
        'depth': [4, 6, 8, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
    }

    cv_folds_s2 = KFold(n_splits=5, shuffle=True, random_state=42)

    random_search_s2 = RandomizedSearchCV(
        estimator=base_cat_model_s2,
        param_distributions=param_distributions_s2,
        n_iter=30,
        scoring='neg_root_mean_squared_error',
        cv=cv_folds_s2,
        verbose=0,
        random_state=42,
        n_jobs=-1
    )

    random_search_s2.fit(X_train_s2, y_train_s2)

    final_cat_model_s2 = CatBoostRegressor(
        **random_search_s2.best_params_,
        iterations=2000,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        early_stopping_rounds=100,
        verbose=False
    )

    final_cat_model_s2.fit(X_train_s2, y_train_s2, eval_set=(X_val_s2, y_val_s2))

    y_pred_val_catboost_s2 = final_cat_model_s2.predict(X_val_s2)

    # Process test_df for CatBoost
    if not test_df_s2.empty and all(col in test_df_s2.columns for col in features_s2):
        test_df_processed_s2 = test_df_s2[features_s2].copy()
        test_df_processed_s2['total_bedrooms'].fillna(mean_total_bedrooms_s2, inplace=True)
        test_predictions_catboost_s2 = final_cat_model_s2.predict(test_df_processed_s2)

        # Save Solution 2 predictions for meta-learning
        np.save(os.path.join(PREDS_DIR, 'y_pred_val_catboost_s2.npy'), y_pred_val_catboost_s2)
        np.save(os.path.join(PREDS_DIR, 'test_predictions_catboost_s2.npy'), test_predictions_catboost_s2)
        print("Solution 2 predictions saved.")
    else:
        print("Test data for Solution 2 is empty or missing required features. Skipping CatBoost test prediction saving.")
else:
    print("Training data for Solution 2 is empty or missing required features. Skipping CatBoost training and prediction saving.")


# --- Meta-Learning Stage ---
print("\n--- Running Meta-Learning ---")

# Load predictions
try:
    y_val_meta = np.load(os.path.join(PREDS_DIR, 'y_val_s1.npy'))
    y_val_pred_lgbm_meta = np.load(os.path.join(PREDS_DIR, 'y_val_pred_lgbm_s1.npy'))
    y_val_pred_xgb_meta = np.load(os.path.join(PREDS_DIR, 'y_val_pred_xgb_s1.npy'))
    y_pred_val_catboost_meta = np.load(os.path.join(PREDS_DIR, 'y_pred_val_catboost_s2.npy'))

    test_predictions_lgbm_meta = np.load(os.path.join(PREDS_DIR, 'test_predictions_lgbm_s1.npy'))
    test_predictions_xgb_meta = np.load(os.path.join(PREDS_DIR, 'test_predictions_xgb_s1.npy'))
    test_predictions_catboost_meta = np.load(os.path.join(PREDS_DIR, 'test_predictions_catboost_s2.npy'))
except FileNotFoundError as e:
    print(f"Error loading intermediate prediction files: {e}. Ensure all base models ran successfully and saved predictions.")
    # In a real competition, you might raise an exception or exit here. For this scenario, we'll try to proceed.
    raise RuntimeError("Failed to load necessary prediction files for meta-learning.") from e


# Ensure all prediction arrays have the same length for validation and test sets
if not (len(y_val_meta) > 0 and 
        len(y_val_meta) == len(y_val_pred_lgbm_meta) == len(y_val_pred_xgb_meta) == len(y_pred_val_catboost_meta)):
    raise ValueError(
        f"Validation prediction arrays have inconsistent or zero lengths:\n"
        f"y_val_meta: {len(y_val_meta)}, y_val_pred_lgbm_meta: {len(y_val_pred_lgbm_meta)}, "
        f"y_val_pred_xgb_meta: {len(y_val_pred_xgb_meta)}, y_pred_val_catboost_meta: {len(y_pred_val_catboost_meta)}"
    )
if not (len(test_predictions_lgbm_meta) > 0 and 
        len(test_predictions_lgbm_meta) == len(test_predictions_xgb_meta) == len(test_predictions_catboost_meta)):
    raise ValueError(
        f"Test prediction arrays have inconsistent or zero lengths:\n"
        f"test_predictions_lgbm_meta: {len(test_predictions_lgbm_meta)}, test_predictions_xgb_meta: {len(test_predictions_xgb_meta)}, "
        f"test_predictions_catboost_meta: {len(test_predictions_catboost_meta)}"
    )

# Construct meta-training and meta-test datasets
X_meta_train = np.c_[y_val_pred_lgbm_meta, y_val_pred_xgb_meta, y_pred_val_catboost_meta]
y_meta_train = y_val_meta

X_meta_test = np.c_[test_predictions_lgbm_meta, test_predictions_xgb_meta, test_predictions_catboost_meta]

# Train a LinearRegression meta-learner
meta_learner = LinearRegression()
meta_learner.fit(X_meta_train, y_meta_train)

# Generate final ensembled predictions for submission
final_ensembled_predictions = meta_learner.predict(X_meta_test)

# Evaluate meta-learner on the meta-training set to get a validation score
y_meta_pred_val = meta_learner.predict(X_meta_train)
final_validation_score = np.sqrt(mean_squared_error(y_meta_train, y_meta_pred_val))
print(f"Final Validation Performance: {final_validation_score}")

# Create the final submission directory if it doesn't exist
FINAL_DIR = "./final"
os.makedirs(FINAL_DIR, exist_ok=True)

# Create submission file
submission_df = pd.DataFrame({'median_house_value': final_ensembled_predictions})
submission_df.to_csv(os.path.join(FINAL_DIR, 'submission.csv'), index=False)
print("Submission file created successfully at ./final/submission.csv.")

