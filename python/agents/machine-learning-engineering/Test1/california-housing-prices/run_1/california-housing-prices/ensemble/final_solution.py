

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Create the final directory if it doesn't exist
os.makedirs('./final', exist_ok=True)

# --- 1. Standardize Data Loading and Splitting ---

# Define file paths
TRAIN_FILE = "./input/train.csv"
TEST_FILE = "./input/test.csv"
SUBMISSION_FILE = "./final/submission.csv" # Standardizing submission file name for consistency and placing it in the final directory

# Load datasets
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# Separate target variable from features in the training data
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Perform a single train_test_split for consistency across all models
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Make a copy of the test_df for independent preprocessing for different pipelines
test_df_copy_lgbm_xgb = test_df.copy()
test_df_copy_rf_base = test_df.copy()
test_df_copy_rf_ref = test_df.copy()

# --- 2. Generate Out-of-Fold (OOF) and Test Predictions for All Base Models ---

# --- Solution 1's Models (LightGBM and XGBoost) ---

# Handle missing values for 'total_bedrooms' for Solution 1's models
# Impute with the median from X_train to prevent data leakage.
median_total_bedrooms = X_train['total_bedrooms'].median()

# Apply imputation to X_train, X_val, and test_df for LGBM/XGB
X_train_lgbm_xgb = X_train.copy()
X_val_lgbm_xgb = X_val.copy()

X_train_lgbm_xgb['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)
X_val_lgbm_xgb['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)
test_df_copy_lgbm_xgb['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# 1. Initialize and Train LightGBM Regressor model
lgbm_model = lgb.LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
lgbm_model.fit(X_train_lgbm_xgb, y_train)
val_preds_lgbm = lgbm_model.predict(X_val_lgbm_xgb)
test_preds_lgbm = lgbm_model.predict(test_df_copy_lgbm_xgb)

# 2. Initialize and Train XGBoost Regressor model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
xgb_model.fit(X_train_lgbm_xgb, y_train)
val_preds_xgb = xgb_model.predict(X_val_lgbm_xgb)
test_preds_xgb = xgb_model.predict(test_df_copy_lgbm_xgb)


# --- Solution 2's Models (RandomForest with different preprocessing) ---

# Identify numerical and categorical features dynamically from the full dataset (X)
# This is done once to define the ColumnTransformers
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Define Preprocessing Pipelines for both base and reference approaches

# --- Preprocessing for Base Model Approach (numerical only) ---
base_numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

base_preprocessor = ColumnTransformer(
    transformers=[
        ('num', base_numerical_transformer, numerical_features)
    ],
    remainder='drop' # Explicitly drop categorical columns, matching base solution's effect
)

# --- Preprocessing for Reference Model Approach (numerical and categorical) ---
reference_numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

reference_categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

reference_preprocessor = ColumnTransformer(
    transformers=[
        ('num', reference_numerical_transformer, numerical_features),
        ('cat', reference_categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# Define Models
# Base Model: RandomForestRegressor with n_estimators=100 from base solution
base_model_regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Reference Model: RandomForestRegressor with default n_estimators from reference solution
reference_model_regressor = RandomForestRegressor(random_state=42, n_jobs=-1)

# Create Full Pipelines by combining preprocessor and model
base_pipeline = Pipeline(steps=[('preprocessor', base_preprocessor),
                                ('regressor', base_model_regressor)])

reference_pipeline = Pipeline(steps=[('preprocessor', reference_preprocessor),
                                     ('regressor', reference_model_regressor)])

# Train both pipelines on X_train and y_train
base_pipeline.fit(X_train, y_train)
reference_pipeline.fit(X_train, y_train)

# Make predictions on the validation set using both models
val_preds_base_rf = base_pipeline.predict(X_val)
val_preds_ref_rf = reference_pipeline.predict(X_val)

# Make predictions on the test set using both models
test_preds_base_rf = base_pipeline.predict(test_df)
test_preds_ref_rf = reference_pipeline.predict(test_df)

# --- 3. Prepare Meta-Learner Training Data ---

# Create a new feature matrix for the meta-learner
X_meta_train = np.column_stack([val_preds_lgbm, val_preds_xgb, val_preds_base_rf, val_preds_ref_rf])

# The target variable for the meta-learner will be y_val
y_meta_train = y_val

# --- 4. Train the Meta-Learner ---

# Initialize and train the meta-learner (Ridge Regressor)
meta_model = Ridge(random_state=42)
meta_model.fit(X_meta_train, y_meta_train)

# --- 5. Generate Final Ensemble Test Predictions ---

# Create the feature matrix for the meta-learner for the test set
X_meta_test = np.column_stack([test_preds_lgbm, test_preds_xgb, test_preds_base_rf, test_preds_ref_rf])

# Use the trained meta-learner to predict on X_meta_test
final_test_predictions = meta_model.predict(X_meta_test)

# Evaluate the ensembled model on the validation set for performance metric
final_val_predictions = meta_model.predict(X_meta_train)
rmse_val = np.sqrt(mean_squared_error(y_val, final_val_predictions))

# Print the final validation performance
print(f"Final Validation Performance: {rmse_val}")

# Create submission file
submission_df = pd.DataFrame({'median_house_value': final_test_predictions})
submission_df.to_csv(SUBMISSION_FILE, index=False)
