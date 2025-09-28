
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
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings('ignore')

# Define paths
TRAIN_PATH = './input/train.csv'
TEST_PATH = './input/test.csv'

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
X_test_final = test_df_copy.copy() # X_test_final will hold features for the final test data

# 2. Preprocessing and Feature Engineering

# Impute missing values for 'total_bedrooms' using the median strategy, fitted only on training data.
imputer = SimpleImputer(strategy='median')
X['total_bedrooms'] = imputer.fit_transform(X[['total_bedrooms']])
# Transform the test set using the imputer fitted on the training data to prevent data leakage.
X_test_final['total_bedrooms'] = imputer.transform(X_test_final[['total_bedrooms']])

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
X_test_final = create_engineered_features(X_test_final)

# 3. Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train RandomForestRegressor model (from Solution 1)
print("Training RandomForest model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=0)
rf_model.fit(X_train, y_train)
print("RandomForest model training complete.")

# 5. Train LightGBM Regressor model (using parameters from Solution 2)
print("Training LightGBM model...")
lgbm_model = lgb.LGBMRegressor(objective='regression',
                               metric='rmse',
                               random_state=42,
                               verbose=-1,
                               n_jobs=-1)
lgbm_model.fit(X_train, y_train)
print("LightGBM model training complete.")

# 6. Train CatBoost Regressor model (from Solution 2)
print("Training CatBoost model...")
catboost_model = CatBoostRegressor(loss_function='RMSE',
                                   random_seed=42,
                                   verbose=0,
                                   iterations=100,
                                   learning_rate=0.1)
catboost_model.fit(X_train, y_train)
print("CatBoost model training complete.")
