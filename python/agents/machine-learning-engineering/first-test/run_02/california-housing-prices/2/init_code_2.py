
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import subprocess
import sys

# Install catboost if not already installed
try:
    from catboost import CatBoostRegressor
except ImportError:
    print("CatBoost not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
    from catboost import CatBoostRegressor

# --- Configuration ---
TRAIN_FILE = './input/train.csv'
TEST_FILE = './input/test.csv'
TARGET_COLUMN = 'median_house_value'
RANDOM_SEED = 42
VALIDATION_SPLIT_RATIO = 0.2

# --- 1. Load Data ---
# Strict adherence to "Do not use try: and except: or if else to ignore unintended behavior."
# assumes files will always be present as per task description "All the provided data is already prepared".
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# Separate target variable from training features
X = train_df.drop(TARGET_COLUMN, axis=1)
y = train_df[TARGET_COLUMN]

# --- 2. Handle Missing Values ---
# The task description implies all features are numerical.
# 'total_bedrooms' is known to sometimes have missing values in this dataset.
# Use SimpleImputer to fill missing numerical values with the median.
imputer = SimpleImputer(strategy='median')

# Fit imputer on training data (X) and transform both training features and test features
X_imputed = imputer.fit_transform(X)
test_imputed = imputer.transform(test_df)

# Convert back to DataFrame, preserving column names
X = pd.DataFrame(X_imputed, columns=X.columns)
test_df = pd.DataFrame(test_imputed, columns=test_df.columns)

# --- 3. Split Data for Validation ---
# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VALIDATION_SPLIT_RATIO, random_state=RANDOM_SEED
)

# --- 4. Train a CatBoost Regressor Model ---
# Initialize CatBoost Regressor as specified in the model description.
# 'loss_function': 'RMSE' directly optimizes for RMSE.
# verbose=False to suppress training output.
# random_seed for reproducibility.
model = CatBoostRegressor(
    loss_function='RMSE',
    random_seed=RANDOM_SEED,
    verbose=False,
    # Additional simple parameters can be added for slightly better performance
    # without going into full HPO, e.g., increasing iterations.
    iterations=1000, # A reasonable number of boosting iterations
    learning_rate=0.05, # A common learning rate
    early_stopping_rounds=50 # Enable early stopping for robustness
)

# Train the model
# CatBoost's fit method can take eval_set for early stopping
model.fit(X_train, y_train, eval_set=(X_val, y_val))

# --- 5. Evaluate the model on the validation set ---
val_predictions = model.predict(X_val)
final_validation_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))

# --- 6. Print Final Validation Performance ---
print(f'Final Validation Performance: {final_validation_rmse}')

# --- 7. Make Predictions on the Test Set ---
test_predictions = model.predict(test_df)

# --- 8. Format the Submission ---
# The submission format requires printing the column name first, then each prediction on a new line.
print(TARGET_COLUMN)
for pred_value in test_predictions:
    print(f"{pred_value}")
