
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the datasets
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Identify features and target
TARGET_COL = 'median_house_value'
FEATURES = [col for col in train_df.columns if col != TARGET_COL]

# Simple imputation for missing values in 'total_bedrooms'
# Calculate median from the training data to avoid data leakage
median_total_bedrooms_train = train_df['total_bedrooms'].median()
train_df['total_bedrooms'].fillna(median_total_bedrooms_train, inplace=True)
test_df['total_bedrooms'].fillna(median_total_bedrooms_train, inplace=True) # Use train median for test set

# Prepare data for XGBoost
X = train_df[FEATURES]
y = train_df[TARGET_COL]

# Split the training data into training and validation sets
# This helps evaluate the model's performance on unseen data before making final predictions
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost Regressor model
# objective='reg:squarederror' for regression tasks
# eval_metric='rmse' to optimize for Root Mean Squared Error
# random_state for reproducibility
model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_jobs=-1)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = model.predict(X_val)

# Evaluate the model using Root Mean Squared Error (RMSE) on the validation set
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Print the validation performance
print(f"Final Validation Performance: {rmse_val}")

# Make predictions on the actual test set for submission
test_predictions = model.predict(test_df[FEATURES])

# Create the submission file
submission_df = pd.DataFrame({'median_house_value': test_predictions})

# Save the submission file
# submission_df.to_csv('submission.csv', index=False)
