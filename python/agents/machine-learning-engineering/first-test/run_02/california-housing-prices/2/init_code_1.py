
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import os

# --- Configuration ---
TRAIN_FILE = './input/train.csv'
TEST_FILE = './input/test.csv'
TARGET_COLUMN = 'median_house_value'
RANDOM_STATE = 42
VALIDATION_SPLIT_RATIO = 0.2

# --- 1. Load Data ---
try:
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
except FileNotFoundError as e:
    # If files are not found, raise an error to stop execution gracefully
    raise FileNotFoundError(f"Error loading data: {e}. Make sure '{os.path.dirname(TRAIN_FILE)}' directory exists and contains '{os.path.basename(TRAIN_FILE)}' and '{os.path.basename(TEST_FILE)}'.")

# Separate target variable
X = train_df.drop(TARGET_COLUMN, axis=1)
y = train_df[TARGET_COLUMN]

# --- 2. Handle Missing Values ---
# Identify numerical columns (all columns in this dataset are numerical)
numerical_cols_X = X.select_dtypes(include=np.number).columns
numerical_cols_test = test_df.select_dtypes(include=np.number).columns

# Use SimpleImputer to fill missing values with the median
imputer = SimpleImputer(strategy='median')

# Fit imputer on training data and transform both training and test data
X[numerical_cols_X] = imputer.fit_transform(X[numerical_cols_X])
# Ensure test_df also has its columns imputed based on the training data's medians
test_df[numerical_cols_test] = imputer.transform(test_df[numerical_cols_test])

# --- 3. Split Data for Validation ---
# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALIDATION_SPLIT_RATIO, random_state=RANDOM_STATE)

# --- 4. Train a LightGBM Model ---
# Define LightGBM parameters
lgb_params = {
    'objective': 'regression',  # Default is regression_l2 (MSE), which optimizes for RMSE
    'metric': 'rmse',
    'n_estimators': 2000,       # Number of boosting rounds
    'learning_rate': 0.01,
    'feature_fraction': 0.8,    # Subsample ratio of columns when constructing each tree
    'bagging_fraction': 0.8,    # Subsample ratio of the training instance
    'bagging_freq': 1,          # Frequency for bagging
    'lambda_l1': 0.1,           # L1 regularization
    'lambda_l2': 0.1,           # L2 regularization
    'num_leaves': 31,           # Max number of leaves in one tree
    'verbose': -1,              # Suppress verbose output
    'n_jobs': -1,               # Use all available cores
    'seed': RANDOM_STATE,
    'boosting_type': 'gbdt',    # Traditional Gradient Boosting Decision Tree
}

model = lgb.LGBMRegressor(**lgb_params)

# Train the model with early stopping
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='rmse',
          callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])

# --- 5. Evaluate the model on the validation set ---
val_predictions = model.predict(X_val)
final_validation_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))

# --- 6. Print Final Validation Performance ---
print(f'Final Validation Performance: {final_validation_rmse}')

# --- 7. Make Predictions on the Test Set ---
test_predictions = model.predict(test_df)

# --- 8. Format the Submission ---
# The submission format requires printing the column name first, then each prediction on a new line.
print(TARGET_COLUMN)
for pred_value in test_predictions:
    print(f"{pred_value}")
