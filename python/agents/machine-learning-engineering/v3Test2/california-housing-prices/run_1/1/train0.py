import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load datasets
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Prepare the data
# Separate target variable from features
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Identify features for the test set
X_test_submission = test_df.copy()

# Handle missing values (e.g., in 'total_bedrooms') and infinite values
# Identify numerical columns for imputation based on both base and reference solutions
numerical_cols_to_impute = ['total_rooms', 'total_bedrooms', 'population', 'households']

for col in numerical_cols_to_impute:
    if col in X.columns:
        # Replace infinities with NaN first
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        # Calculate median only from the training features to prevent data leakage
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        # Apply the same median to the test set
        if col in X_test_submission.columns:
            X_test_submission[col] = X_test_submission[col].replace([np.inf, -np.inf], np.nan)
            X_test_submission[col].fillna(median_val, inplace=True)

# The 'ocean_proximity' column is not present in the provided dataset schema
# and attempting to use pd.get_dummies on it causes a KeyError.
# Therefore, the lines related to 'ocean_proximity' are commented out.
# X = pd.get_dummies(X, columns=['ocean_proximity'], drop_first=True)
# X_test_submission = pd.get_dummies(X_test_submission, columns=['ocean_proximity'], drop_first=True)

# Align columns - crucial for consistent feature sets between training and test after one-hot encoding
# This ensures that if a category is missing in one set but present in another, columns are aligned.
train_cols = X.columns
test_cols = X_test_submission.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test_submission[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X[c] = 0

# Ensure the order of columns is the same for both datasets
X_test_submission = X_test_submission[train_cols]
X = X[train_cols]

# Split the training data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training and Prediction ---

# 1. Initialize and train the LightGBM Regressor model
lgbm = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1)
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_val)

# 2. Initialize and train the CatBoost Regressor model
catboost = CatBoostRegressor(iterations=100,  # Number of boosting iterations (trees)
                             learning_rate=0.1, # Step size shrinkage
                             depth=6,         # Depth of trees
                             loss_function='RMSE', # Loss function to optimize
                             random_seed=42,    # For reproducibility
                             verbose=False,     # Suppress training output
                             allow_writing_files=False) # Prevent CatBoost from writing diagnostic files
catboost.fit(X_train, y_train)
y_pred_catboost = catboost.predict(X_val)

# --- Ensembling ---
# Simple average ensembling of predictions from LightGBM and CatBoost
y_pred_ensemble = (y_pred_lgbm + y_pred_catboost) / 2

# Evaluate the ensembled model using Root Mean Squared Error (RMSE)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))

# Print the final validation performance
print(f'Final Validation Performance: {rmse_val}')

# Make predictions on the actual test set for submission
y_pred_lgbm_test = lgbm.predict(X_test_submission)
y_pred_catboost_test = catboost.predict(X_test_submission)

# Ensembled predictions for the test set
final_predictions = (y_pred_lgbm_test + y_pred_catboost_test) / 2

# Create submission file
submission_df = pd.DataFrame(final_predictions, columns=['median_house_value'])
submission_df.to_csv('submission.csv', index=False)