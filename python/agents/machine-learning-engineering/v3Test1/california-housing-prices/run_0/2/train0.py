
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the datasets
# Assuming train.csv and test.csv are located in the ./input directory
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate target variable from features
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Handle missing values
# The 'total_bedrooms' column is known to have a small number of missing values
# in this dataset. We impute them with the median value calculated from the training data.
median_total_bedrooms_train = X['total_bedrooms'].median()
X['total_bedrooms'] = X['total_bedrooms'].fillna(median_total_bedrooms_train)
test_df['total_bedrooms'] = test_df['total_bedrooms'].fillna(median_total_bedrooms_train)

# Split the training data into a training set and a hold-out validation set
# This ensures we evaluate the model on unseen data before making final predictions.
# A fixed random_state is used for reproducibility.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---

# 1. Initialize and Train LightGBM Regressor (from Base Solution)
# We use 'regression' objective which minimizes L2 loss (MSE),
# and track 'rmse' as the evaluation metric during training.
# 'random_state' ensures reproducibility.
# 'verbose=-1' suppresses all LightGBM training output.
# 'n_jobs=-1' utilizes all available CPU cores for faster training.
lgbm_model = lgb.LGBMRegressor(objective='regression',
                               metric='rmse',
                               random_state=42,
                               verbose=-1,
                               n_jobs=-1)

print("Training LightGBM model...")
lgbm_model.fit(X_train, y_train)
print("LightGBM model training complete.")

# 2. Initialize and Train CatBoost Regressor (from Reference Solution)
# CatBoost natively supports 'RMSE' as a loss function for regression tasks.
# 'random_seed' ensures reproducibility.
# 'verbose=0' suppresses all training output.
# 'iterations' and 'learning_rate' are set to reasonable defaults.
catboost_model = CatBoostRegressor(loss_function='RMSE',
                                   random_seed=42,
                                   verbose=0,  # Suppress training output
                                   iterations=100,  # Number of boosting iterations
                                   learning_rate=0.1) # Step size shrinkage

print("Training CatBoost model...")
catboost_model.fit(X_train, y_train)
print("CatBoost model training complete.")

# --- Make Predictions ---

# Make predictions on the hold-out validation set using both models
y_pred_lgbm = lgbm_model.predict(X_val)
y_pred_catboost = catboost_model.predict(X_val)

# --- Ensemble the Models ---

# Simple averaging ensemble
# This combines the predictions from both LightGBM and CatBoost models.
y_pred_ensemble = (y_pred_lgbm + y_pred_catboost) / 2

# Calculate the Root Mean Squared Error (RMSE) on the ensembled validation predictions
rmse_ensemble_val = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))

# Print the final validation performance in the specified format
print(f"Final Validation Performance: {rmse_ensemble_val}")
