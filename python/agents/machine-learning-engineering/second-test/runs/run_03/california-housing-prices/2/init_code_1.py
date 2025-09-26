
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training dataset from the specified input directory
train_df = pd.read_csv("./input/train.csv")

# Separate features (X) and the target variable (y)
# The target variable is 'median_house_value'
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# --- Preprocessing for Missing Values ---
# The 'total_bedrooms' column is known to have missing values in this dataset.
# We impute these missing values with the median of the column.
# Using the median from the training data prevents data leakage from the validation set.
median_total_bedrooms = X['total_bedrooms'].median()
X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# --- Data Splitting ---
# Split the processed training data into training and a hold-out validation set.
# This allows us to evaluate the model's performance on unseen data.
# A test_size of 0.2 means 20% of the data will be used for validation.
# random_state ensures reproducibility of the split.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Initialization and Training ---
# Initialize the XGBoost Regressor model as described in the model description.
# 'objective='reg:squarederror'' specifies a regression task with squared error.
# 'n_estimators' is the number of boosting rounds (decision trees).
# 'learning_rate' controls the step size shrinkage.
# 'random_state' ensures reproducibility of the model training.
# 'n_jobs=-1' utilizes all available CPU cores for faster training.
model_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

# Train the XGBoost model on the training subset
model_xgb.fit(X_train, y_train)

# --- Model Evaluation ---
# Make predictions on the hold-out validation set
y_pred_val = model_xgb.predict(X_val)

# Calculate the Root Mean Squared Error (RMSE) for the validation set.
# This is the specified metric for the task.
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance in the required format
print(f"Final Validation Performance: {rmse_val}")
