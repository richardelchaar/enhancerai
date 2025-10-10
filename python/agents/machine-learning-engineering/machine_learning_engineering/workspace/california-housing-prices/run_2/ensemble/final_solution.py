
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
from sklearn.model_selection import train_test_split
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
for k in [5, 10, 20]:
    X[f'geo_cluster_{k}'] = X[f'geo_cluster_{k}'].astype('category')
    test_df[f'geo_cluster_{k}'] = test_df[f'geo_cluster_{k}'].astype('category')

X['lat_bin'] = X['lat_bin'].astype('category')
X['lon_bin'] = X['lon_bin'].astype('category')
test_df['lat_bin'] = test_df['lat_bin'].astype('category')
test_df['lon_bin'] = test_df['lon_bin'].astype('category')

# --- Model Training and Prediction ---
# Train models on the full training dataset (X, y)

# 1. LightGBM Model
lgbm_model = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1)
# Train the LightGBM model on the full training data
lgbm_model.fit(X, y)

# 2. XGBoost Model
# Fix: Added enable_categorical=True to handle categorical features.
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, verbosity=0, n_estimators=1000, enable_categorical=True)
# Train the XGBoost model on the full training data
xgb_model.fit(X, y)

# Make predictions on the test set using both models
y_pred_lgbm_test = lgbm_model.predict(test_df)
y_pred_xgb_test = xgb_model.predict(test_df)

# --- Ensembling ---
# Simple average ensemble of the two models' predictions for the test set
y_pred_ensemble_test = (y_pred_lgbm_test + y_pred_xgb_test) / 2

# Create the './final' directory if it doesn't exist
os.makedirs('./final', exist_ok=True)

# Create submission file
submission_df = pd.DataFrame({'median_house_value': y_pred_ensemble_test})
submission_df.to_csv('./final/submission.csv', index=False)
