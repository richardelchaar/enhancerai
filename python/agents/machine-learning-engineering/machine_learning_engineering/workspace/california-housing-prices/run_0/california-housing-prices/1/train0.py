

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the datasets
try:
    train_df = pd.read_csv('./input/train.csv')
    test_df = pd.read_csv('./input/test.csv')
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the './input/' directory.")
    # Create dummy dataframes for execution outside Kaggle environment if files not found
    data = {
        'longitude': np.random.rand(100) * -20 - 110,
        'latitude': np.random.rand(100) * 10 + 30,
        'housing_median_age': np.random.randint(1, 50, 100),
        'total_rooms': np.random.randint(100, 6000, 100),
        'total_bedrooms': np.random.randint(50, 1200, 100),
        'population': np.random.randint(100, 5000, 100),
        'households': np.random.randint(50, 1000, 100),
        'median_income': np.random.rand(100) * 10,
        'median_house_value': np.random.rand(100) * 500000
    }
    train_df = pd.DataFrame(data)
    test_data = {
        'longitude': np.random.rand(50) * -20 - 110,
        'latitude': np.random.rand(50) * 10 + 30,
        'housing_median_age': np.random.randint(1, 50, 50),
        'total_rooms': np.random.randint(100, 6000, 50),
        'total_bedrooms': np.random.randint(50, 1200, 50),
        'population': np.random.randint(100, 5000, 50),
        'households': np.random.randint(50, 1000, 50),
        'median_income': np.random.rand(50) * 10
    }
    test_df = pd.DataFrame(test_data)
    # Introduce some NaN values in total_bedrooms for testing imputation
    train_df.loc[train_df.sample(frac=0.01, random_state=42).index, 'total_bedrooms'] = np.nan
    test_df.loc[test_df.sample(frac=0.01, random_state=42).index, 'total_bedrooms'] = np.nan


# Separate target variable
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Preprocessing: Handle missing values
# Impute missing 'total_bedrooms' with the median
# Using median as it's more robust to outliers
median_total_bedrooms = X['total_bedrooms'].median()
X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)
test_df['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Split the training data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---

# 1. Initialize and Train LightGBM Regressor (from base solution)
lgbm_model = lgb.LGBMRegressor(random_state=42)
lgbm_model.fit(X_train, y_train)

# 2. Initialize and Train XGBoost Regressor (from reference solution)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train)

# --- Prediction and Ensembling ---

# Make predictions on the validation set for each model
y_pred_lgbm = lgbm_model.predict(X_val)
y_pred_xgb = xgb_model.predict(X_val)

# Ensemble the predictions by averaging
y_pred_ensemble = (y_pred_lgbm + y_pred_xgb) / 2

# --- Evaluation ---

# Evaluate the ensembled model using RMSE
rmse_val_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))

# Print the final validation performance
print(f'Final Validation Performance: {rmse_val_ensemble}')

# --- Optional: Generate Test Predictions for Submission (commented out as not explicitly required for metric output) ---
# test_predictions_lgbm = lgbm_model.predict(test_df)
# test_predictions_xgb = xgb_model.predict(test_df)
# test_predictions_ensemble = (test_predictions_lgbm + test_predictions_xgb) / 2
#
# # Create submission file
# submission_df = pd.DataFrame({'median_house_value': test_predictions_ensemble})
# submission_df.to_csv('submission.csv', index=False)

