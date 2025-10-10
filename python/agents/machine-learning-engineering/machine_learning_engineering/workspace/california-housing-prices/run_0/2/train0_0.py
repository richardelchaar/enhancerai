import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the datasets
try:
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
except FileNotFoundError:
    # Fallback for Kaggle environment where files might be in ../input/
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")

# Separate target variable from features
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Identify categorical features (none explicitly mentioned, assuming all are numerical for now)
# For this dataset, all features are numerical.

# Handle missing values
# The 'total_bedrooms' column often has missing values in this dataset.
# Impute missing values with the median of the column.
for col in X.columns:
    if X[col].isnull().any():
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        # Apply the same imputation to the test set using the training set's median
        if col in test_df.columns:
            test_df[col].fillna(median_val, inplace=True)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LightGBM Regressor model
# 'objective': 'regression' is the default for regression tasks.
# 'metric': 'rmse' is specified in the problem description.
# 'verbose': -1 suppresses verbose output.
model = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Calculate RMSE on the validation set
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance
print(f"Final Validation Performance: {rmse_val}")

# Prepare for submission (optional, but good practice)
# Make predictions on the test set
# y_pred_test = model.predict(test_df)

# Create submission file (example structure)
# submission_df = pd.DataFrame({'median_house_value': y_pred_test})
# submission_df.to_csv('submission.csv', index=False)