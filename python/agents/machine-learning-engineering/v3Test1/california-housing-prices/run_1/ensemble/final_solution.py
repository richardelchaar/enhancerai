
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Load the training data from the specified path
df_train = pd.read_csv('./input/train.csv')

# Handle missing values: Impute 'total_bedrooms' with its median.
# This is a robust and simple strategy for missing numerical data.
# Calculate median from training data to avoid data leakage from test set.
median_total_bedrooms = df_train['total_bedrooms'].median()
df_train['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Define features (X) and target (y)
# The target variable is 'median_house_value'.
X = df_train.drop('median_house_value', axis=1)
y = df_train['median_house_value']

# Split the data into training and validation sets.
# A 80/20 split is used, with a fixed random_state for reproducibility.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- LightGBM Model ---
# Initialize the LightGBM Regressor model.
# 'objective=regression' and 'metric=rmse' are appropriate for this task.
# 'random_state' ensures reproducibility.
# 'verbose=-1' is used to suppress all verbose output during training, as required.
lgbm_model = lgb.LGBMRegressor(objective='regression',
                               metric='rmse',
                               random_state=42,
                               verbose=-1)

# Train the LightGBM model on the training data.
lgbm_model.fit(X_train, y_train)

# Make predictions on the validation set.
y_pred_lgbm = lgbm_model.predict(X_val)

# --- XGBoost Model ---
# Initialize the XGBoost Regressor model.
# 'objective='reg:squarederror'' is the standard objective for regression tasks.
# 'eval_metric='rmse'' is specified for evaluation during training (though we'll use sklearn's for final score).
# 'random_state' ensures reproducibility.
# 'verbosity=0' is used to suppress all verbose output during training, as required.
xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                             eval_metric='rmse',
                             random_state=42,
                             verbosity=0) # suppress all verbose output

# Train the XGBoost model on the training data.
xgb_model.fit(X_train, y_train)

# Make predictions on the validation set.
y_pred_xgb = xgb_model.predict(X_val)

# --- Ensembling Predictions ---
# Simple averaging of predictions from LightGBM and XGBoost models.
y_pred_ensembled = (y_pred_lgbm + y_pred_xgb) / 2

# Calculate the Root Mean Squared Error (RMSE) on the ensembled predictions.
# RMSE is the specified metric for this competition.
rmse = np.sqrt(mean_squared_error(y_val, y_pred_ensembled))

# Print the final validation performance in the required format.
print(f"Final Validation Performance: {rmse}")

# --- Submission Preparation ---
# Load the test data
df_test = pd.read_csv('./input/test.csv')

# Handle missing values in test data: Impute 'total_bedrooms' with the median from training data.
# It's crucial to use the median calculated from the training data to avoid data leakage.
df_test['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Retrain models on the full training dataset for final predictions
# This is a common practice to utilize all available data for the most robust model.

# LightGBM model on full training data
lgbm_model_final = lgb.LGBMRegressor(objective='regression',
                                     metric='rmse',
                                     random_state=42,
                                     verbose=-1)
lgbm_model_final.fit(X, y) # Train on full X and y

# XGBoost model on full training data
xgb_model_final = xgb.XGBRegressor(objective='reg:squarederror',
                                   eval_metric='rmse',
                                   random_state=42,
                                   verbosity=0)
xgb_model_final.fit(X, y) # Train on full X and y

# Make predictions on the test data using the models trained on the full dataset
y_pred_lgbm_test = lgbm_model_final.predict(df_test)
y_pred_xgb_test = xgb_model_final.predict(df_test)

# Ensembled predictions for the test set
y_pred_ensembled_test = (y_pred_lgbm_test + y_pred_xgb_test) / 2

# Create the submission DataFrame
submission_df = pd.DataFrame({'median_house_value': y_pred_ensembled_test})

# Ensure the output directory exists
os.makedirs('./final', exist_ok=True)

# Save the submission file to the specified path
submission_df.to_csv('./final/submission.csv', index=False)

print("Submission file created successfully at ./final/submission.csv")

