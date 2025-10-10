import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load the training and test data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate features and target from training data
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Identify numerical features for imputation
numerical_features = X.select_dtypes(include=np.number).columns

# Impute missing values using median strategy
# Fit the imputer on the training features, then transform both training features and test data
imputer = SimpleImputer(strategy='median')
X[numerical_features] = imputer.fit_transform(X[numerical_features])
test_df[numerical_features] = imputer.transform(test_df[numerical_features])

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model 1: LightGBM ---
# Initialize and train the LGBMRegressor model
model_lgbm = lgb.LGBMRegressor(objective='regression_l2', metric='rmse', random_state=42, verbose=-1, n_jobs=-1)
model_lgbm.fit(X_train, y_train)

# Make predictions on the validation set with LGBM
y_pred_lgbm_val = model_lgbm.predict(X_val)

# --- Model 2: XGBoost ---
# Initialize and train the XGBRegressor model
model_xgb = xgb.XGBRegressor(objective='reg:squarederror',
                             eval_metric='rmse',
                             random_state=42,
                             n_jobs=-1,
                             verbosity=0)  # Suppress verbose output
model_xgb.fit(X_train, y_train)

# Make predictions on the validation set with XGBoost
y_pred_xgb_val = model_xgb.predict(X_val)

# --- Ensemble the predictions ---
# Simple averaging ensemble
y_pred_ensemble_val = (y_pred_lgbm_val + y_pred_xgb_val) / 2

# Calculate RMSE on the validation set for the ensembled predictions
rmse_val_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_ensemble_val))

# Print the final validation performance
print(f"Final Validation Performance: {rmse_val_ensemble}")