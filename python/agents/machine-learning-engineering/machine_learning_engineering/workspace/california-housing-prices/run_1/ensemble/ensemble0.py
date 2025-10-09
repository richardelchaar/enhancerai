
# Suppress verbose model output to prevent token explosion
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
# Suppress LightGBM verbosity
os.environ['LIGHTGBM_VERBOSITY'] = '-1'
# Suppress XGBoost verbosity  
os.environ['XGBOOST_VERBOSITY'] = '0'
# Suppress sklearn warnings
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import re

# Load datasets
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate features and target from the training data
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Store the number of training samples to separate them later after preprocessing
num_train_samples = X.shape[0]

# Concatenate training features and test features for consistent preprocessing
combined_df = pd.concat([X, test_df], ignore_index=True)

# Identify numerical and categorical columns
# Assuming 'ocean_proximity' is the only categorical column based on typical datasets
numerical_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
categorical_cols = [] # Initialize as empty

# Check if 'ocean_proximity' column exists in combined_df
if 'ocean_proximity' in combined_df.columns:
    categorical_cols.append('ocean_proximity')
else:
    # If 'ocean_proximity' is not present, ensure it's not in categorical_cols.
    # This ensures robustness if the dataset schema changes or is different from assumption.
    # We should still handle the case where the problem statement implies its existence
    # but the provided CSVs don't have it. For this specific task, if it's missing,
    # the original code would have implicitly handled it by having an empty list,
    # but the context strongly suggests it's usually there.
    # For now, let's assume if it's not in combined_df, then it won't be one-hot encoded.
    pass

# --- Preprocessing: Handle missing values and categorical features ---

# Handle missing numerical values (e.g., in 'total_bedrooms') and infinite values
for col in numerical_cols:
    if col in combined_df.columns:
        # Replace infinities with NaN first
        combined_df[col] = combined_df[col].replace([np.inf, -np.inf], np.nan)
        # Calculate median only from the training data part to prevent data leakage
        median_val = combined_df.loc[:num_train_samples-1, col].median()
        combined_df[col].fillna(median_val, inplace=True)

# Handle categorical features using one-hot encoding
combined_df = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=False)

# Sanitize column names for XGBoost compatibility
# XGBoost can have issues with special characters like [, ], <, > in feature names.
def sanitize_features_for_xgb(df):
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = re.sub(r'[^A-Za-z0-9_]+', '', col) # Remove all non-alphanumeric/underscore characters
        # If the start of the column name is a digit, prepend 'col_' to make it a valid identifier
        if new_col and new_col[0].isdigit():
            new_col = 'col_' + new_col
        new_cols.append(new_col)
    df.columns = new_cols
    return df

combined_df = sanitize_features_for_xgb(combined_df)

# Separate back into processed training features and test features
X_processed = combined_df.iloc[:num_train_samples].copy()
test_processed = combined_df.iloc[num_train_samples:].copy()

# Split the processed training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# --- LightGBM Model ---
lgbm = lgb.LGBMRegressor(objective='regression',
                         metric='rmse',
                         random_state=42,
                         n_jobs=-1,
                         verbose=-1) # Suppress verbose output
lgbm.fit(X_train, y_train)
y_pred_lgbm_val = lgbm.predict(X_val)

# --- CatBoost Model ---
catboost = CatBoostRegressor(iterations=100,
                             learning_rate=0.1,
                             depth=6,
                             loss_function='RMSE',
                             random_seed=42,
                             verbose=False, # Suppress verbose output
                             allow_writing_files=False)
catboost.fit(X_train, y_train)
y_pred_catboost_val = catboost.predict(X_val)

# --- XGBoost Model (tuned) ---
xgb_model_base = xgb.XGBRegressor(objective='reg:squarederror',
                                  eval_metric='rmse',
                                  random_state=42,
                                  verbosity=0) # Suppress verbose output

param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 5, 10]
}

xgb_random_search = RandomizedSearchCV(estimator=xgb_model_base,
                                       param_distributions=param_distributions,
                                       n_iter=50, # Number of parameter settings that are sampled
                                       scoring='neg_mean_squared_error',
                                       cv=3,
                                       verbose=0, # Suppress verbose output during search
                                       random_state=42,
                                       n_jobs=-1)
xgb_random_search.fit(X_train, y_train)
xgb_model = xgb_random_search.best_estimator_
y_pred_xgb_val = xgb_model.predict(X_val)

# --- Ensemble Prediction ---
# Simple average ensemble for validation predictions
y_pred_ensemble_val = (y_pred_lgbm_val + y_pred_catboost_val + y_pred_xgb_val) / 3
final_validation_score = np.sqrt(mean_squared_error(y_val, y_pred_ensemble_val))

print(f'Final Validation Performance: {final_validation_score}')

# --- Generate predictions for test.csv ---
# Predict with each model on the preprocessed test data
y_pred_lgbm_test = lgbm.predict(test_processed)
y_pred_catboost_test = catboost.predict(test_processed)
y_pred_xgb_test = xgb_model.predict(test_processed)

# Simple average ensemble for test set predictions
y_pred_ensemble_test = (y_pred_lgbm_test + y_pred_catboost_test + y_pred_xgb_test) / 3

# Create submission file
submission_df = pd.DataFrame({'median_house_value': y_pred_ensemble_test})

# Print the submission in the required format
print("median_house_value")
for val in submission_df['median_house_value']:
    print(val)