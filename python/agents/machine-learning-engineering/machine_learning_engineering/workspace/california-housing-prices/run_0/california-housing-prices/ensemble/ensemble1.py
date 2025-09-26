
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import os

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

# Preprocessing: Handle missing values using SimpleImputer from Solution 2
imputer = SimpleImputer(strategy='median')

# Fit imputer on training data and transform total_bedrooms for both X and test_df
X['total_bedrooms'] = imputer.fit_transform(X[['total_bedrooms']])
test_df['total_bedrooms'] = imputer.transform(test_df[['total_bedrooms']]) # Apply to test_df

# Split the training data into training and validation sets
# Using a fixed random_state for reproducibility as in Solution 1 and the plan
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---

# 1. Initialize and Train LightGBM Regressor (default parameters as in Solution 1)
lgbm_model = lgb.LGBMRegressor(random_state=42)
lgbm_model.fit(X_train, y_train)

# 2. Initialize and Train XGBoost Regressor (using GridSearchCV from Solution 2)
# Define the parameter grid for light hyperparameter tuning
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Initialize the XGBoost Regressor for GridSearch
xgb_base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, use_label_encoder=False, eval_metric='rmse')

# Perform GridSearchCV for hyperparameter tuning
grid_search_xgb = GridSearchCV(
    estimator=xgb_base_model,
    param_grid=param_grid_xgb,
    scoring='neg_root_mean_squared_error',
    cv=3,
    n_jobs=-1,
    verbose=0
)

grid_search_xgb.fit(X_train, y_train)

# Get the best XGBoost model found by GridSearchCV
xgb_model = grid_search_xgb.best_estimator_

# --- Prediction and Ensembling ---

# Make predictions on the validation set for each model
y_pred_lgbm = lgbm_model.predict(X_val)
y_pred_xgb = xgb_model.predict(X_val)

# Combine validation predictions using a simple unweighted average (as per ensemble plan)
y_pred_ensemble = (y_pred_lgbm + y_pred_xgb) / 2

# --- Evaluation ---

# Evaluate the ensembled model using RMSE
rmse_val_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))

# Print the final validation performance as required
print(f'Final Validation Performance: {rmse_val_ensemble}')

# --- Generate Test Predictions for Submission ---
# Make predictions on the preprocessed test set with both models
test_predictions_lgbm = lgbm_model.predict(test_df)
test_predictions_xgb = xgb_model.predict(test_df)

# Ensemble test predictions using the same unweighted average
test_predictions_ensemble = (test_predictions_lgbm + test_predictions_xgb) / 2

# Create submission file (commented out as per instructions not to modify original submission part)
# submission_df = pd.DataFrame({'median_house_value': test_predictions_ensemble})
# submission_df.to_csv('submission.csv', index=False)
# print("Submission file created: submission.csv")
