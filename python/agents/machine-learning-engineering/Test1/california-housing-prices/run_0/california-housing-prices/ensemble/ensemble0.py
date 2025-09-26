

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import os

# --- 1. Load Data ---
# Adopt Solution 1's robust loading with dummy data fallback
try:
    train_df = pd.read_csv('./input/train.csv')
    test_df = pd.read_csv('./input/test.csv')
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the './input/' directory. Using dummy data.")
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

# --- 2. Preprocessing: Handle missing values ---
# Adopt Solution 2's SimpleImputer for total_bedrooms
imputer = SimpleImputer(strategy='median')

# Fit imputer on training data and transform total_bedrooms for both X and test_df
X['total_bedrooms'] = imputer.fit_transform(X[['total_bedrooms']])
test_df['total_bedrooms'] = imputer.transform(test_df[['total_bedrooms']]) # Apply to test_df

# --- 3. Data Splitting ---
# Maintain train_test_split with random_state=42 as used in both solutions
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Model Training - LightGBM ---
# Train LightGBM with default parameters as in both solutions
lgbm_model = lgb.LGBMRegressor(objective='regression_l2', random_state=42) # objective added from solution 2 for clarity
lgbm_model.fit(X_train, y_train)

# --- 5. Model Training - XGBoost with Hyperparameter Tuning ---
# Adopt Solution 2's GridSearchCV for XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

xgb_base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, use_label_encoder=False, eval_metric='rmse')

grid_search_xgb = GridSearchCV(
    estimator=xgb_base_model,
    param_grid=param_grid_xgb,
    scoring='neg_root_mean_squared_error', # GridSearchCV maximizes scores, so neg_rmse to find best (min) RMSE
    cv=3,
    n_jobs=-1,
    verbose=0
)

grid_search_xgb.fit(X_train, y_train)
xgb_model = grid_search_xgb.best_estimator_

# --- 6. Prediction on Validation Set ---
y_pred_lgbm = lgbm_model.predict(X_val)
y_pred_xgb = xgb_model.predict(X_val)

# --- 7. Ensembling - Dynamic Weighting ---
# Calculate RMSE for each model on the validation set
rmse_lgbm = np.sqrt(mean_squared_error(y_val, y_pred_lgbm))
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))

# Implement Solution 2's dynamic weighting scheme
epsilon = 1e-6 # To avoid division by zero
weight_lgbm = 1 / (rmse_lgbm + epsilon)
weight_xgb = 1 / (rmse_xgb + epsilon)

# Normalize weights so they sum to 1
total_weight = weight_lgbm + weight_xgb
normalized_weight_lgbm = weight_lgbm / total_weight
normalized_weight_xgb = weight_xgb / total_weight

# Combine validation predictions using dynamic weights
y_pred_ensemble = (normalized_weight_lgbm * y_pred_lgbm) + (normalized_weight_xgb * y_pred_xgb)

# --- 8. Evaluation ---
rmse_val_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))
print(f'Final Validation Performance: {rmse_val_ensemble}')

# --- 9. Test Set Predictions for Submission ---
# Apply the same two trained models (default LGBM and tuned XGBoost) to the test dataset
test_predictions_lgbm = lgbm_model.predict(test_df)
test_predictions_xgb = xgb_model.predict(test_df)

# Combine their test predictions using the same dynamic weights derived from the validation set
test_predictions_ensemble = (normalized_weight_lgbm * test_predictions_lgbm) + (normalized_weight_xgb * test_predictions_xgb)

# Create submission file (commented out as per instructions, similar to original solutions)
# submission_df = pd.DataFrame({'median_house_value': test_predictions_ensemble})
# submission_df.to_csv('submission.csv', index=False)
# print("Submission file created: submission.csv")

