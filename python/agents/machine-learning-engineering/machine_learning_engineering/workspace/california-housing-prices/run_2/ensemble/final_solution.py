
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
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import uniform, randint
import os

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

# --- Feature Engineering (from Run 1) ---
# Combine X and test_df for consistent feature engineering
# Store original shapes to split back later
X_shape = X.shape[0]
combined_df = pd.concat([X, test_df], ignore_index=True)

# 1. Geographic Clustering (KMeans)
# Use MiniBatchKMeans for efficiency
kmeans = MiniBatchKMeans(n_clusters=10, random_state=42, n_init=10, batch_size=256)
combined_df['geo_cluster_k'] = kmeans.fit_predict(combined_df[['latitude', 'longitude']])

# Calculate distance to cluster centroids
for i in range(kmeans.n_clusters):
    combined_df[f'distance_to_cluster_{i}'] = np.sqrt(
        (combined_df['latitude'] - kmeans.cluster_centers_[i, 0])**2 +
        (combined_df['longitude'] - kmeans.cluster_centers_[i, 1])**2
    )

# 2. Latitude and Longitude Binning
# Define number of bins
num_lat_bins = 10
num_lon_bins = 10

# Create bins for latitude and longitude
combined_df['lat_bin'] = pd.cut(combined_df['latitude'], bins=num_lat_bins, labels=False, include_lowest=True)
combined_df['lon_bin'] = pd.cut(combined_df['longitude'], bins=num_lon_bins, labels=False, include_lowest=True)

# Convert new categorical features to 'category' dtype for LightGBM
categorical_features = ['geo_cluster_k', 'lat_bin', 'lon_bin']
for col in categorical_features:
    combined_df[col] = combined_df[col].astype('category')

# Split back into X and test_df
X = combined_df.iloc[:X_shape].copy()
test_df = combined_df.iloc[X_shape:].copy()

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training and Prediction with Hyperparameter Tuning ---

# Define parameter distributions for RandomizedSearchCV

# LightGBM Parameter Grid
lgbm_param_dist = {
    'n_estimators': randint(500, 3001),  # Max 3000
    'learning_rate': uniform(0.01, 0.1 - 0.01),  # From 0.01 to 0.1
    'num_leaves': randint(15, 101),  # From 15 to 100
    'max_depth': randint(3, 21),  # From 3 to 20
    'min_child_samples': randint(5, 51),  # From 5 to 50
    'subsample': uniform(0.6, 0.4),  # From 0.6 to 1.0
    'colsample_bytree': uniform(0.6, 0.4),  # From 0.6 to 1.0
    'reg_alpha': uniform(0, 1.0),  # From 0 to 1.0
    'reg_lambda': uniform(0, 1.0)  # From 0 to 1.0
}

# XGBoost Parameter Grid
xgb_param_dist = {
    'n_estimators': randint(500, 3001),  # Max 3000
    'learning_rate': uniform(0.01, 0.1 - 0.01),  # From 0.01 to 0.1
    'max_depth': randint(3, 11),  # From 3 to 10
    'min_child_weight': randint(1, 11),  # From 1 to 10
    'subsample': uniform(0.6, 0.4),  # From 0.6 to 1.0
    'colsample_bytree': uniform(0.6, 0.4),  # From 0.6 to 1.0
    'gamma': uniform(0, 0.5),  # From 0 to 0.5
    'reg_alpha': uniform(0, 1.0),  # From 0 to 1.0
    'reg_lambda': uniform(0, 1.0)  # From 0 to 1.0
}

# 1. LightGBM Model Tuning
lgbm_model_base = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1, n_jobs=-1)
lgbm_search = RandomizedSearchCV(
    estimator=lgbm_model_base,
    param_distributions=lgbm_param_dist,
    n_iter=40,
    scoring='neg_root_mean_squared_error',
    cv=3,
    random_state=42,
    verbose=1,
    n_jobs=-1
)
# Fit RandomizedSearchCV on X_train, y_train, explicitly passing categorical features
lgbm_search.fit(X_train, y_train, categorical_feature=[col for col in categorical_features if col in X_train.columns])
lgbm_best_model = lgbm_search.best_estimator_

# Make predictions on the validation set with LightGBM
y_pred_lgbm = lgbm_best_model.predict(X_val)

# 2. XGBoost Model Tuning
xgb_model_base = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, verbosity=0, enable_categorical=True, n_jobs=-1)
xgb_search = RandomizedSearchCV(
    estimator=xgb_model_base,
    param_distributions=xgb_param_dist,
    n_iter=40,
    scoring='neg_root_mean_squared_error',
    cv=3,
    random_state=42,
    verbose=1,
    n_jobs=-1
)
# Fit RandomizedSearchCV on X_train, y_train
xgb_search.fit(X_train, y_train)
xgb_best_model = xgb_search.best_estimator_

# Make predictions on the validation set with XGBoost
y_pred_xgb = xgb_best_model.predict(X_val)

# --- Ensembling ---
# Simple average ensemble of the two models' predictions
y_pred_ensemble = (y_pred_lgbm + y_pred_xgb) / 2

# --- Evaluation ---
# Calculate RMSE on the validation set for the ensembled predictions
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))

# Print the final validation performance
print(f"Final Validation Performance: {rmse_val}")

# --- Final Model Training and Submission ---

# Retrain the best LightGBM model on the full training data (X, y)
lgbm_final_model = lgbm_search.best_estimator_
lgbm_final_model.fit(X, y, categorical_feature=[col for col in categorical_features if col in X.columns])

# Retrain the best XGBoost model on the full training data (X, y)
xgb_final_model = xgb_search.best_estimator_
xgb_final_model.fit(X, y)

# Make predictions on the test set using the retrained models
y_pred_lgbm_test = lgbm_final_model.predict(test_df)
y_pred_xgb_test = xgb_final_model.predict(test_df)

# Simple average ensemble of the two models' predictions for the test set
y_pred_ensemble_test = (y_pred_lgbm_test + y_pred_xgb_test) / 2

# Create the submission directory if it doesn't exist
os.makedirs('./final', exist_ok=True)

# Create submission file
submission_df = pd.DataFrame({'median_house_value': y_pred_ensemble_test})
submission_df.to_csv('./final/submission.csv', index=False)
