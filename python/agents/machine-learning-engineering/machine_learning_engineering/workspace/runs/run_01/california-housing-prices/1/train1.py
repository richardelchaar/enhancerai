
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb  # Import for XGBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load the datasets
# As per the task description, data is available in the './input' directory.
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate target variable from features in the training data
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Identify numerical features for imputation.
# In this dataset, all features (longitude, latitude, housing_median_age, etc.) are numerical.
numerical_cols = X.select_dtypes(include=np.number).columns

# Initialize SimpleImputer to handle missing values.
# We use the median strategy as it's robust to outliers and suitable for a simple solution.
# It's crucial to fit the imputer on the training data ONLY to prevent data leakage.
imputer = SimpleImputer(strategy='median')

# Fit the imputer on the training features and transform them.
X_imputed_array = imputer.fit_transform(X[numerical_cols])
X = pd.DataFrame(X_imputed_array, columns=numerical_cols, index=X.index)

# Apply the same imputer (already fitted on training data) to the test data.
# Note: Use 'transform' method, not 'fit_transform', for the test set.
test_numerical_cols = test_df.select_dtypes(include=np.number).columns
X_test_imputed_array = imputer.transform(test_df[test_numerical_cols])
X_test = pd.DataFrame(X_test_imputed_array, columns=test_numerical_cols, index=test_df.index)

# Split the processed training data into training and validation sets.
# This allows us to evaluate the model's performance on unseen data.
# A 80/20 split is a common practice, and random_state ensures reproducibility.
X_train_split, X_val, y_train_split, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---

# Initialize and train the LightGBM Regressor model
lgbm_model = lgb.LGBMRegressor(objective='regression',
                               metric='rmse',
                               n_estimators=100,
                               learning_rate=0.1,
                               num_leaves=31,
                               random_state=42,
                               n_jobs=-1)

# Train the LightGBM model on the training split.
lgbm_model.fit(X_train_split, y_train_split)

# Make predictions on the hold-out validation set using LightGBM.
y_pred_val_lgbm = lgbm_model.predict(X_val)

# Initialize and train the XGBoost Regressor model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', # 'reg:squarederror' is standard for RMSE
                             n_estimators=100,
                             learning_rate=0.1,
                             max_depth=5,
                             random_state=42,
                             n_jobs=-1) # Use all available CPU cores for faster training

# Train the XGBoost model on the training split.
xgb_model.fit(X_train_split, y_train_split)

# Make predictions on the hold-out validation set using XGBoost.
y_pred_val_xgb = xgb_model.predict(X_val)

# --- Ensemble Predictions ---
# Simple averaging ensemble of LightGBM and XGBoost predictions.
y_pred_val_ensemble = (y_pred_val_lgbm + y_pred_val_xgb) / 2

# Evaluate the ensembled model's performance using Root Mean Squared Error (RMSE).
rmse_val_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble))

# Print the final validation performance in the required format.
print(f"Final Validation Performance: {rmse_val_ensemble}")
