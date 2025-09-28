
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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define paths
TRAIN_PATH = './input/train.csv'
TEST_PATH = './input/test.csv'
SUBMISSION_PATH = 'submission.csv'

# 1. Load data
try:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Make sure 'train.csv' and 'test.csv' are in the './input/' directory.")
    raise

# Make copies to avoid SettingWithCopyWarning in pandas
train_df_copy = train_df.copy()
test_df_copy = test_df.copy()

# Drop 'id' column if it exists, as it's not a feature
if 'id' in train_df_copy.columns:
    train_df_copy = train_df_copy.drop('id', axis=1)
if 'id' in test_df_copy.columns:
    test_df_copy = test_df_copy.drop('id', axis=1)

# Separate target variable from training features
y = train_df_copy['median_house_value']
X = train_df_copy.drop('median_house_value', axis=1)
X_test = test_df_copy.copy() # X_test will hold features for the test data

# 2. Preprocessing and Feature Engineering

# Impute missing values for 'total_bedrooms' using the median strategy, fitted only on training data.
imputer = SimpleImputer(strategy='median')
X['total_bedrooms'] = imputer.fit_transform(X[['total_bedrooms']])
# Transform the test set using the imputer fitted on the training data to prevent data leakage.
X_test['total_bedrooms'] = imputer.transform(X_test[['total_bedrooms']])

# Create new engineered numerical features
def create_engineered_features(df):
    """
    Creates new features based on existing numerical columns.
    """
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    return df

X = create_engineered_features(X)
X_test = create_engineered_features(X_test)

# The original error was due to 'ocean_proximity' not being found in the DataFrame.
# Based on the provided dataset schema (longitude, latitude, housing_median_age, etc.),
# 'ocean_proximity' is not present in the input CSV files.
# Therefore, the lines attempting to one-hot encode it and align columns based on its categories
# are removed as they are not applicable to this dataset.

# 3. Split training data for validation
# Splitting data into training and testing sets is a common practice for evaluating models.
# random_state ensures reproducibility of results.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train RandomForestRegressor model (Base Solution Model)
# RandomForestRegressor is an ensemble machine learning algorithm that uses multiple decision trees.
# It averages their outputs to make a final prediction, which helps to reduce variance and overfitting.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=0)
rf_model.fit(X_train, y_train)

# 5. Train LightGBM Regressor model (Reference Solution Model)
# LightGBM is a gradient boosting framework that uses tree-based learning algorithms.
# It is designed to be distributed and efficient.
# verbose=-1 is used to suppress LightGBM's verbose output.
lgbm_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
lgbm_model.fit(X_train, y_train)

# 6. Evaluate on validation set and Ensemble
y_pred_rf_val = rf_model.predict(X_val)
y_pred_lgbm_val = lgbm_model.predict(X_val)

# Simple Averaging Ensemble of predictions from both models
y_pred_val_ensemble = (y_pred_rf_val + y_pred_lgbm_val) / 2

# Root Mean Squared Error (RMSE) is a common metric for regression problems.
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble))

# Print the final validation performance as required by the task.
print(f"Final Validation Performance: {rmse_val}")

# 7. Predict on the test set
predictions_rf_test = rf_model.predict(X_test)
predictions_lgbm_test = lgbm_model.predict(X_test)

# Simple Averaging Ensemble for final test predictions
predictions_ensemble_test = (predictions_rf_test + predictions_lgbm_test) / 2

# 8. Generate submission file
submission_df = pd.DataFrame({'median_house_value': predictions_ensemble_test})
submission_df.to_csv(SUBMISSION_PATH, index=False)
