
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training data
try:
    df_train = pd.read_csv("./input/train.csv")
except FileNotFoundError:
    df_train = pd.read_csv("train.csv")


# Separate features (X) and target (y)
X = df_train.drop("median_house_value", axis=1)
y = df_train["median_house_value"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values in 'total_bedrooms' with the mean from the training set
# This prevents data leakage from the validation set
mean_total_bedrooms = X_train['total_bedrooms'].mean()
X_train['total_bedrooms'].fillna(mean_total_bedrooms, inplace=True)
X_val['total_bedrooms'].fillna(mean_total_bedrooms, inplace=True) # Apply the same mean to the validation set

# Initialize the LightGBM Regressor model
lgbm = lgb.LGBMRegressor(random_state=42)

# Train the model
lgbm.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = lgbm.predict(X_val)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

# Print the final validation performance
print(f"Final Validation Performance: {rmse}")
