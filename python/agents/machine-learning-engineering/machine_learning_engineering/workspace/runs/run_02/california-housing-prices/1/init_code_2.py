
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load the datasets
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Define features (X) and target (y)
TARGET = 'median_house_value'
# All columns except the target are features
FEATURES = [col for col in train_df.columns if col != TARGET]

X = train_df[FEATURES]
y = train_df[TARGET]

# Prepare features for the final test set predictions (test.csv)
X_test_final = test_df[FEATURES]

# Handle missing values
# The 'total_bedrooms' column is known to have missing values in this dataset.
# We'll use SimpleImputer with a 'median' strategy.
imputer = SimpleImputer(strategy='median')

# Fit the imputer on the training features and transform them
X[FEATURES] = imputer.fit_transform(X[FEATURES])

# Transform the final test set features using the same imputer
X_test_final[FEATURES] = imputer.transform(X_test_final[FEATURES])

# Split the training data into training and validation sets
# A fixed random_state is used for reproducibility.
# The validation set will be 20% of the training data.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the CatBoost Regressor
# Parameters are chosen to be reasonable defaults for a simple, fast training.
# 'iterations' is the number of boosting rounds.
# 'loss_function' is set to 'RMSE' as per the task's metric.
# 'verbose=0' suppresses the training output for cleaner execution.
# 'random_seed' ensures reproducibility.
model = CatBoostRegressor(
    iterations=200,          # A reasonable number of boosting iterations
    learning_rate=0.1,       # Default learning rate
    depth=6,                 # Default tree depth
    loss_function='RMSE',    # Objective function for regression
    random_seed=42,          # For reproducibility
    verbose=0                # Suppress training output
)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Evaluate the model using Root Mean Squared Error (RMSE)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance in the required format
print(f"Final Validation Performance: {rmse_val}")

# To generate predictions for the submission file (optional, but good practice for Kaggle):
# y_pred_test = model.predict(X_test_final)
# submission_df = pd.DataFrame({'median_house_value': y_pred_test})
# submission_df.to_csv('submission.csv', index=False)
