
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
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Install necessary libraries if not already installed ---
# This block is commented out as it's typically for environment setup,
# and the platform running this code might have these pre-installed.
# If you encounter ModuleNotFoundError, uncomment and run these lines.
# try:
#     import pandas
# except ImportError:
#     print("Installing pandas...")
#     !pip install pandas numpy scikit-learn
#     import pandas
#     import numpy
#     import sklearn

# Define paths
TRAIN_PATH = './input/train.csv'
TEST_PATH = './input/test.csv'
SUBMISSION_PATH = 'submission.csv'

# 1. Load data
try:
    train_df = pd.read_csv(TRAIN_PATH) # [1, 3, 5, 8, 13]
    test_df = pd.read_csv(TEST_PATH) # [1, 3, 5, 8, 13]
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Make sure 'train.csv' and 'test.csv' are in the './input/' directory.")
    # In a production environment, you might want to log this error and exit more gracefully.
    # For this task, we will assume files are present or handle it if they aren't provided by the system.
    # Re-raising the error to stop execution if files are truly missing.
    raise

# Make copies to avoid SettingWithCopyWarning in pandas
train_df_copy = train_df.copy()
test_df_copy = test_df.copy()

# 2. Preprocessing and Feature Engineering
def preprocess_data(df):
    """
    Applies preprocessing steps including median imputation for missing values
    and feature engineering.
    """
    # Impute missing values for 'total_bedrooms' using the median strategy. [2, 4, 16, 18, 19]
    # SimpleImputer is a univariate imputer for completing missing values with simple strategies. [2, 18]
    # 'median' strategy replaces missing values using the median along each column. [2, 4, 19]
    imputer = SimpleImputer(strategy='median') # [2, 4, 16, 18, 19]
    df['total_bedrooms'] = imputer.fit_transform(df[['total_bedrooms']])

    # Create new features to potentially improve model performance.
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']

    return df

train_df_processed = preprocess_data(train_df_copy)
test_df_processed = preprocess_data(test_df_copy)

# Define features (X) and target (y)
features = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    'rooms_per_household', 'bedrooms_per_room', 'population_per_household'
]
target = 'median_house_value'

X = train_df_processed[features]
y = train_df_processed[target]
X_test = test_df_processed[features]

# 3. Split training data for validation
# Splitting data into training and testing sets is a common practice for evaluating models. [17, 21]
# random_state ensures reproducibility of results. [17, 21]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train RandomForestRegressor model
# RandomForestRegressor is an ensemble machine learning algorithm that uses multiple decision trees. [11, 15, 17, 21]
# It averages their outputs to make a final prediction, which helps to reduce variance and overfitting. [15, 17, 20]
# n_estimators: The number of trees in the forest. [17, 20]
# n_jobs=-1 uses all available processors for faster computation.
# verbose=0 suppresses detailed output during training.
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=0) # [11, 15, 17, 20, 21]
model.fit(X_train, y_train) # [11, 21]

# 5. Evaluate on validation set
y_pred_val = model.predict(X_val) # [11]
# Root Mean Squared Error (RMSE) is a common metric for regression problems. [9, 10, 14]
# It measures the average magnitude of the errors, with lower values indicating better accuracy. [10]
# np.sqrt(mean_squared_error(y_true, y_pred)) calculates RMSE. [6, 7, 9, 10]
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val)) # [6, 7, 9, 10]

# Print the final validation performance as required by the task.
print(f"Final Validation Performance: {rmse_val}")

# 6. Predict on the test set
predictions = model.predict(X_test) # [11, 17]

# 7. Generate submission file
submission_df = pd.DataFrame({'median_house_value': predictions})
submission_df.to_csv(SUBMISSION_PATH, index=False)

