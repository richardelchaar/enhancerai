
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load the datasets
# Assuming the files are in the './input/' directory as per common Kaggle setup
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate target variable from features
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Split the training data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical features for imputation (all features except target are numerical here)
numerical_features = X_train.select_dtypes(include=np.number).columns

# Impute missing values
# The 'total_bedrooms' column is known to have missing values.
# Using SimpleImputer with a median strategy is robust to outliers.
imputer = SimpleImputer(strategy='median')

# Fit imputer on training data and transform all splits
# It's crucial to fit only on training data to prevent data leakage.
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train[numerical_features]), 
                               columns=numerical_features, index=X_train.index)
X_val_imputed = pd.DataFrame(imputer.transform(X_val[numerical_features]), 
                             columns=numerical_features, index=X_val.index)
test_df_imputed = pd.DataFrame(imputer.transform(test_df[numerical_features]), 
                               columns=numerical_features, index=test_df.index)

# Initialize and train the LightGBM Regressor model
# objective='regression' specifies the task is regression
# n_estimators: number of boosting rounds (trees)
# learning_rate: step size shrinkage to prevent overfitting
# random_state for reproducibility
model_lightgbm = lgb.LGBMRegressor(objective='regression', n_estimators=500, 
                                   learning_rate=0.05, random_state=42, n_jobs=-1)

# Fit the model on the imputed training data
model_lightgbm.fit(X_train_imputed, y_train)

# Make predictions on the validation set
y_val_pred = model_lightgbm.predict(X_val_imputed)

# Calculate Root Mean Squared Error (RMSE) on the validation set
# This is the specified evaluation metric for the task.
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Print the validation performance as required
print(f"Final Validation Performance: {rmse_val}")

# Generate predictions for the test dataset for submission
test_predictions = model_lightgbm.predict(test_df_imputed)

# Create submission file (optional, but good practice for Kaggle)
# submission_df = pd.DataFrame({'median_house_value': test_predictions})
# submission_df.to_csv('submission.csv', index=False)
