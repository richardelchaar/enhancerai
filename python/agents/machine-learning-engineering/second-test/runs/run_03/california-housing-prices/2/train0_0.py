
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training dataset
train_df = pd.read_csv("./input/train.csv")

# Separate features (X) and the target variable (y)
# The target variable is 'median_house_value'
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# --- Preprocessing for Missing Values ---
# The 'total_bedrooms' column is known to have missing values in this dataset.
# Impute these missing values with the median of the column to prevent data leakage
# from the validation set, the median is calculated only from the training features.
median_total_bedrooms = X['total_bedrooms'].median()
X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# --- Data Splitting ---
# Split the processed training data into training and a hold-out validation set.
# This allows us to evaluate the model's performance on unseen data.
# A test_size of 0.2 means 20% of the data will be used for validation.
# random_state ensures reproducibility of the split.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Initialization and Training ---
# Initialize the LightGBM Regressor model as described in the model description.
# 'objective='regression'' specifies a regression task.
# 'n_estimators' is the number of boosting rounds (trees).
# 'learning_rate' controls the step size shrinkage.
# 'random_state' ensures reproducibility of the model training.
# 'n_jobs=-1' utilizes all available CPU cores for faster training.
model_lgb = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1 # Use all available cores
)

# Train the LightGBM model on the training subset
model_lgb.fit(X_train, y_train)

# --- Model Evaluation ---
# Make predictions on the hold-out validation set
y_pred_val = model_lgb.predict(X_val)

# Calculate the Root Mean Squared Error (RMSE) for the validation set.
# This is the specified metric for the task.
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance in the required format
print(f"Final Validation Performance: {rmse_val}")
