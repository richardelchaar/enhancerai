

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import subprocess
import sys

# Install catboost if not already installed
try:
    from catboost import CatBoostRegressor
except ImportError:
    print("CatBoost not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
    from catboost import CatBoostRegressor

# Install lightgbm if not already installed
try:
    import lightgbm as lgb
except ImportError:
    print("LightGBM not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
    import lightgbm as lgb

# --- Configuration ---
TRAIN_FILE = './input/train.csv'
TEST_FILE = './input/test.csv'
TARGET_COLUMN = 'median_house_value'
RANDOM_SEED = 42 # Using RANDOM_SEED from base solution for consistency
VALIDATION_SPLIT_RATIO = 0.2

# --- 1. Load Data ---
# Strict adherence to "Do not use try: and except: or if else to ignore unintended behavior."
# assumes files will always be present as per task description "All the provided data is already prepared".
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# Separate target variable from training features
X = train_df.drop(TARGET_COLUMN, axis=1)
y = train_df[TARGET_COLUMN]


# --- 2. Handle Missing Values ---
# The task description implies all features are numerical.
# 'total_bedrooms' is known to sometimes have missing values in this dataset.

# Targeted imputation for 'total_bedrooms' using 'total_rooms'
# Calculate median bedrooms_per_room ratio from training data where both are present
ratio_data = X[['total_bedrooms', 'total_rooms']].dropna()
if not ratio_data.empty and 'total_rooms' in ratio_data.columns and 'total_bedrooms' in ratio_data.columns:
    # Avoid division by zero: filter out rows where total_rooms is zero before calculating ratio
    # Assuming total_rooms should always be positive for meaningful ratio
    valid_ratio_data = ratio_data[ratio_data['total_rooms'] > 0]
    if not valid_ratio_data.empty:
        median_bedrooms_per_room = (valid_ratio_data['total_bedrooms'] / valid_ratio_data['total_rooms']).median()
    else:
        # Fallback if no valid ratios can be calculated (e.g., all total_rooms are 0 or NaN)
        # In such cases, we might fall back to simple median imputation for total_bedrooms later
        median_bedrooms_per_room = None
else:
    median_bedrooms_per_room = None # No valid data to calculate ratio

if median_bedrooms_per_room is not None:
    # Impute missing 'total_bedrooms' in X
    missing_bedrooms_X_mask = X['total_bedrooms'].isnull()
    # Ensure 'total_rooms' is not NaN for rows being imputed, if so, that specific imputation will be NaN
    # and handled by the generic imputer later.
    imputed_values_X = X.loc[missing_bedrooms_X_mask, 'total_rooms'] * median_bedrooms_per_room
    X.loc[missing_bedrooms_X_mask, 'total_bedrooms'] = imputed_values_X

    # Impute missing 'total_bedrooms' in test_df
    missing_bedrooms_test_mask = test_df['total_bedrooms'].isnull()
    # Ensure 'total_rooms' is not NaN for rows being imputed
    imputed_values_test = test_df.loc[missing_bedrooms_test_mask, 'total_rooms'] * median_bedrooms_per_room
    test_df.loc[missing_bedrooms_test_mask, 'total_bedrooms'] = imputed_values_test

# Use SimpleImputer to fill any remaining numerical missing values (including any total_bedrooms
# that couldn't be imputed by the ratio due to missing total_rooms, or other columns).
imputer = SimpleImputer(strategy='median')

# Fit imputer on training data (X) and transform both training features and test features
X_imputed = imputer.fit_transform(X)
test_imputed = imputer.transform(test_df)

# Convert back to DataFrame, preserving column names
X = pd.DataFrame(X_imputed, columns=X.columns)
test_df = pd.DataFrame(test_imputed, columns=test_df.columns)


# --- 3. Split Data for Validation ---
# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VALIDATION_SPLIT_RATIO, random_state=RANDOM_SEED
)

# --- 4. Train Models and Ensemble ---

# --- CatBoost Regressor (from Base Solution) ---
# Initialize CatBoost Regressor
model_cb = CatBoostRegressor(
    loss_function='RMSE',
    random_seed=RANDOM_SEED,
    verbose=False,
    iterations=1000,
    learning_rate=0.05,
    early_stopping_rounds=50
)

# Train the CatBoost model
model_cb.fit(X_train, y_train, eval_set=(X_val, y_val))

# Get predictions from CatBoost model
cb_val_predictions = model_cb.predict(X_val)
cb_test_predictions = model_cb.predict(test_df)


# --- LightGBM Model (from Reference Solution) ---
# Define LightGBM parameters, using RANDOM_SEED for consistency
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': RANDOM_SEED, # Use RANDOM_SEED instead of RANDOM_STATE
    'boosting_type': 'gbdt',
}

model_lgb = lgb.LGBMRegressor(**lgb_params)

# Train the LightGBM model with early stopping
model_lgb.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])

# Get predictions from LightGBM model
lgb_val_predictions = model_lgb.predict(X_val)
lgb_test_predictions = model_lgb.predict(test_df)


# --- Ensembling (Simple Averaging) ---
# Average the predictions from both models for validation set
ensemble_val_predictions = (cb_val_predictions + lgb_val_predictions) / 2

# Average the predictions from both models for test set
ensemble_test_predictions = (cb_test_predictions + lgb_test_predictions) / 2


# --- 5. Evaluate the ensemble model on the validation set ---
final_validation_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_predictions))

# --- 6. Print Final Validation Performance ---
print(f'Final Validation Performance: {final_validation_rmse}')

# --- 7. Make Predictions on the Test Set (Ensemble Predictions) ---
# The ensemble_test_predictions are already calculated above.

# --- 8. Format the Submission ---
# The submission format requires printing the column name first, then each prediction on a new line.
print(TARGET_COLUMN)
for pred_value in ensemble_test_predictions:
    print(f"{pred_value}")

