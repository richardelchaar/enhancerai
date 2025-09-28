
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
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the datasets
# Assuming train.csv is located in the ./input directory
train_df = pd.read_csv("./input/train.csv")
# The test.csv is not needed for validation performance calculation,
# but loading it here for completeness if predictions were needed later.
# test_df = pd.read_csv("./input/test.csv")

# Separate target variable from features
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Handle missing values
# The 'total_bedrooms' column is known to have a small number of missing values
# in this dataset. We impute them with the median value calculated from the training data.
median_total_bedrooms_train = X['total_bedrooms'].median()
X['total_bedrooms'] = X['total_bedrooms'].fillna(median_total_bedrooms_train)
# If using test_df for predictions later, its missing values would also be filled here:
# test_df['total_bedrooms'] = test_df['total_bedrooms'].fillna(median_total_bedrooms_train)

# Split the training data into a training set and a hold-out validation set
# This ensures we evaluate the model on unseen data before making final predictions.
# A fixed random_state is used for reproducibility.
# The test_size of 0.2 means 20% of the data will be used for validation.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoost Regressor with RMSE as the loss function
# CatBoost natively supports 'RMSE' as a loss function for regression tasks.
# 'random_seed' ensures reproducibility.
# 'verbose=0' suppresses all training output to meet the requirement.
# 'iterations' and 'learning_rate' are set to reasonable defaults as per the example.
model = CatBoostRegressor(loss_function='RMSE', 
                          random_seed=42, 
                          verbose=0,  # Suppress training output
                          iterations=100,  # Number of boosting iterations
                          learning_rate=0.1 # Step size shrinkage
                         )

# Train the CatBoost model on the training data
model.fit(X_train, y_train)

# Make predictions on the hold-out validation set
y_pred_val = model.predict(X_val)

# Calculate the Root Mean Squared Error (RMSE) on the validation set
# This is the evaluation metric for the task.
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance in the specified format
print(f"Final Validation Performance: {rmse_val}")
