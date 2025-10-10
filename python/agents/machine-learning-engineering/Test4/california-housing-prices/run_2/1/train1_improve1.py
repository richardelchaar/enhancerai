
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

# Suppress verbose model output to prevent token explosion
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
# Suppress LightGBM verbosity
os.environ['LIGHTGBM_VERBOSITY'] = '-1'
# Suppress XGBoost verbosity
os.environ['XGBOOST_VERBOSITY'] = '0'
# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)


# Load the datasets
# As per instructions, assume files are in './input' and do not use try/except.
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate target variable from features
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# --- Preprocessing ---
# Handle missing values
# Impute missing values with the median of the column from the training data.
# This ensures consistency between training and test set imputation.
for col in X.columns:
    if X[col].isnull().any():
        median_val = X[col].median()  # Calculate median from training data
        X[col].fillna(median_val, inplace=True)
        # Apply the same imputation to the test set using the training set's median
        if col in test_df.columns:
            test_df[col].fillna(median_val, inplace=True)

# --- Geographic Feature Engineering ---
# (1) KMeans clustering for latitude and longitude
for k in [5, 10, 20]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')

# Fit KMeans on training data's latitude and longitude
    X_coords = X[['latitude', 'longitude']]
    kmeans.fit(X_coords)

# Predict cluster IDs for both training and test data
    X[f'geo_cluster_{k}'] = kmeans.predict(X_coords)
    test_df[f'geo_cluster_{k}'] = kmeans.predict(test_df[['latitude', 'longitude']])

# Calculate Euclidean distance to the assigned cluster centroid for both training and test data
    # For X
    distances_X = euclidean_distances(X_coords, kmeans.cluster_centers_)
    X[f'distance_to_cluster_{k}'] = distances_X[np.arange(len(X_coords)), X[f'geo_cluster_{k}']]

# For test_df
    distances_test = euclidean_distances(test_df[['latitude', 'longitude']], kmeans.cluster_centers_)
    test_df[f'distance_to_cluster_{k}'] = distances_test[np.arange(len(test_df)), test_df[f'geo_cluster_{k}']]

# (2) Grid-based binning by rounding latitude and longitude
X['lat_bin'] = X['latitude'].round(1)
X['lon_bin'] = X['longitude'].round(1)
test_df['lat_bin'] = test_df['latitude'].round(1)
test_df['lon_bin'] = test_df['longitude'].round(1)

# Convert new categorical features to 'category' dtype
categorical_features = []  # List to store names of categorical features for LightGBM
for k in [5, 10, 20]:
    X[f'geo_cluster_{k}'] = X[f'geo_cluster_{k}'].astype('category')
    test_df[f'geo_cluster_{k}'] = test_df[f'geo_cluster_{k}'].astype('category')
    categorical_features.append(f'geo_cluster_{k}')

X['lat_bin'] = X['lat_bin'].astype('category')
X['lon_bin'] = X['lon_bin'].astype('category')
test_df['lat_bin'] = test_df['lat_bin'].astype('category')
test_df['lon_bin'] = test_df['lon_bin'].astype('category')
categorical_features.extend(['lat_bin', 'lon_bin'])

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training and Prediction ---

# Define parameter grids for RandomizedSearchCV
lgbm_param_grid = {
    'n_estimators': [500, 1000, 1500, 2000, 3000],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'num_leaves': [15, 31, 50, 100],
    'max_depth': [3, 5, 10, 15, 20],
    'min_child_samples': [5, 10, 20, 50],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0, 0.1, 0.5, 1.0]
}

xgb_param_grid = {
    'n_estimators': [500, 1000, 1500, 2000, 3000],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'max_depth': [3, 5, 7, 10],
    'min_child_weight': [1, 3, 5, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3, 0.5],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0, 0.1, 0.5, 1.0]
}

# 1. LightGBM Model Tuning
lgbm = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1, n_jobs=-1)
lgbm_random_search = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=lgbm_param_grid,
    n_iter=40,
    cv=3,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    verbose=1,
    n_jobs=-1  # Use all available cores
)
# Fit RandomizedSearchCV on training data, explicitly passing categorical features
lgbm_random_search.fit(X_train, y_train, categorical_feature=categorical_features)
lgbm_model = lgbm_random_search.best_estimator_
print(f"LightGBM Best Parameters: {lgbm_random_search.best_params_}")
print(f"LightGBM Best RMSE (CV): {-lgbm_random_search.best_score_}")

# Make predictions on the validation set with LightGBM
y_pred_lgbm = lgbm_model.predict(X_val)

# 2. XGBoost Model Tuning
xgb_base = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, verbosity=0, enable_categorical=True, n_jobs=-1)
xgb_random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=xgb_param_grid,
    n_iter=40,
    cv=3,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    verbose=1,
    n_jobs=-1  # Use all available cores
)
# Train the XGBoost model
xgb_random_search.fit(X_train, y_train)
xgb_model = xgb_random_search.best_estimator_
print(f"XGBoost Best Parameters: {xgb_random_search.best_params_}")
print(f"XGBoost Best RMSE (CV): {-xgb_random_search.best_score_}")

# Make predictions on the validation set with XGBoost
y_pred_xgb = xgb_model.predict(X_val)

# --- Ensembling ---
# Simple average ensemble of the two models' predictions
y_pred_ensemble = (y_pred_lgbm + y_pred_xgb) / 2

# --- Evaluation ---
# Calculate RMSE on the validation set for the ensembled predictions
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))

# Print the final validation performance
print(f"Final Validation Performance: {rmse_val}")

# Optional: Prepare for submission
# Make predictions on the test set using both models
y_pred_lgbm_test = lgbm_model.predict(test_df)
y_pred_xgb_test = xgb_model.predict(test_df)
y_pred_ensemble_test = (y_pred_lgbm_test + y_pred_xgb_test) / 2

# Create submission file (example structure)
submission_df = pd.DataFrame({'median_house_value': y_pred_ensemble_test})
submission_df.to_csv('submission.csv', index=False)
