
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
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# Separate features and target
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Handle missing values using SimpleImputer (median strategy) for all numerical columns
# It's crucial to fit the imputer on the training data only and then transform both train and test.
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Split the data into training and validation sets for consistent evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a RandomForestRegressor model
# Suppress verbose output by setting verbose=0
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=0)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Calculate Root Mean Squared Error (RMSE)
# The 'squared' parameter is not supported in older scikit-learn versions.
# Calculate MSE first, then take the square root to get RMSE.
mse = mean_squared_error(y_val, y_pred_val)
final_validation_score = np.sqrt(mse)

# Print the final validation performance
print(f"Final Validation Performance: {final_validation_score}")

# Load the test data
test_df = pd.read_csv("./input/test.csv")

# Apply the *same* imputer fitted on the training data to the test data.
# This prevents data leakage from the test set.
test_imputed = imputer.transform(test_df)
test_df_processed = pd.DataFrame(test_imputed, columns=test_df.columns)

# Make predictions on the processed test data
test_predictions = model.predict(test_df_processed)

# Create the submission file
submission_df = pd.DataFrame({'median_house_value': test_predictions})
submission_df.to_csv('submission.csv', index=False)
