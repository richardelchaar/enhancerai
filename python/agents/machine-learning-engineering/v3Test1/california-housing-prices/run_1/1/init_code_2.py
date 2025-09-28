
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
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training data from the specified path
df_train = pd.read_csv('./input/train.csv')

# Handle missing values: Impute 'total_bedrooms' with its median.
# This is a common and robust strategy for numerical features with missing data.
# The original example did not specify how to handle missing values,
# but it's crucial for most models. Checking for missing values:
# df_train.isnull().sum() would show total_bedrooms has missing values.
if 'total_bedrooms' in df_train.columns:
    df_train['total_bedrooms'].fillna(df_train['total_bedrooms'].median(), inplace=True)

# Define features (X) and target (y)
# The target variable is 'median_house_value'.
X = df_train.drop('median_house_value', axis=1)
y = df_train['median_house_value']

# Split the data into training and validation sets.
# A 80/20 split is used, with a fixed random_state for reproducibility.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Make predictions on the validation set to evaluate performance.
y_pred_val = xgb_model.predict(X_val)

# Calculate the Root Mean Squared Error (RMSE) on the validation set.
# RMSE is the specified metric for this competition.
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance in the required format.
print(f"Final Validation Performance: {rmse}")
