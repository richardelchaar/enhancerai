
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training data from the specified path
df_train = pd.read_csv('./input/train.csv')

# Handle missing values: Impute 'total_bedrooms' with its median.
# This is a robust and simple strategy for missing numerical data.
df_train['total_bedrooms'].fillna(df_train['total_bedrooms'].median(), inplace=True)

# Define features (X) and target (y)
# The target variable is 'median_house_value'.
X = df_train.drop('median_house_value', axis=1)
y = df_train['median_house_value']

# Split the data into training and validation sets.
# A 80/20 split is used, with a fixed random_state for reproducibility.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Make predictions on the validation set to evaluate performance.
y_pred_val = lgbm_model.predict(X_val)

# Calculate the Root Mean Squared Error (RMSE) on the validation set.
# RMSE is the specified metric for this competition.
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance in the required format.
print(f"Final Validation Performance: {rmse}")
