
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load the datasets
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
except FileNotFoundError:
    # Adjust path for Kaggle environment if needed, or local execution with ./input
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")

# Separate target variable from features
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Split the training data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical features for imputation
numerical_features = X_train.select_dtypes(include=np.number).columns

# Impute missing values (if any)
# Use SimpleImputer for numerical features, strategy='median' is robust to outliers
imputer = SimpleImputer(strategy='median')

# Fit imputer on training data and transform all splits
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train[numerical_features]), columns=numerical_features, index=X_train.index)
X_val_imputed = pd.DataFrame(imputer.transform(X_val[numerical_features]), columns=numerical_features, index=X_val.index)
test_df_imputed = pd.DataFrame(imputer.transform(test_df[numerical_features]), columns=numerical_features, index=test_df.index)

# Ensure all columns are in the same order if there were non-numerical columns initially (though not in this dataset)
# For this specific dataset, all features are numerical, so simply using the imputed dataframes is fine.
# If there were categorical features, they would need separate handling (e.g., one-hot encoding)
# and then concatenation with the imputed numerical features.

# Initialize and train the XGBoost Regressor model
# objective='reg:squarederror' is recommended for regression tasks.
# n_estimators: number of boosting rounds/trees - kept simple for initial solution
# learning_rate: step size shrinkage to prevent overfitting - kept simple
# random_state for reproducibility
model_xgboost = XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1)
model_xgboost.fit(X_train_imputed, y_train)

# Make predictions on the validation set
y_val_pred = model_xgboost.predict(X_val_imputed)

# Calculate RMSE on the validation set
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Print the validation performance
print(f"Final Validation Performance: {rmse_val}")

# Generate predictions for the test dataset for submission
test_predictions = model_xgboost.predict(test_df_imputed)

# Create submission file (if this were a full Kaggle submission script)
submission_df = pd.DataFrame({'median_house_value': test_predictions})
# submission_df.to_csv('submission.csv', index=False)
# print("Submission file created successfully!")
