
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Load datasets
# The data is located in the './input' directory as per the instructions.
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate target variable from features in the training data
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Handle missing values for 'total_bedrooms'
# Impute with the median from the training data to prevent data leakage.
# This median will be used for both the training features and the test set.
median_total_bedrooms = X['total_bedrooms'].median()

X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)
test_df['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Split the training data into training and validation sets
# A fixed random_state is used for reproducibility, as required for simple solutions.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LightGBM Regressor model
# As per the model description, we use 'objective='regression''.
# n_estimators and learning_rate are set to the example values.
# random_state ensures reproducibility. n_jobs=-1 utilizes all available CPU cores.
lgbm_model = lgb.LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)

# Train the model
lgbm_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = lgbm_model.predict(X_val)

# Evaluate the model using Root Mean Squared Error (RMSE)
# RMSE is the specified metric for this task.
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance in the specified format
print(f"Final Validation Performance: {rmse_val}")

# Optionally, for generating a submission file, one would predict on the test_df:
# test_predictions = lgbm_model.predict(test_df)
# submission_df = pd.DataFrame({'median_house_value': test_predictions})
# submission_df.to_csv('submission.csv', index=False)
