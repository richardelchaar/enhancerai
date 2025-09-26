
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training data
try:
    train_df = pd.read_csv('./input/train.csv')
except FileNotFoundError:
    # Fallback for Kaggle environment where data might be in ../input/
    train_df = pd.read_csv('../input/train.csv')


# Define features (X) and target (y)
TARGET = 'median_house_value'
features = [col for col in train_df.columns if col != TARGET]
X = train_df[features]
y = train_df[TARGET]

# Handle missing values: Impute 'total_bedrooms' with the median
# The problem description implies 'total_bedrooms' is the only column that might have NaNs.
# Calculate median from the training features to prevent data leakage.
median_total_bedrooms = X['total_bedrooms'].median()
X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Split the data into training and validation sets
# A reasonable split ratio like 80/20 is often used for validation.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LightGBM Regressor model
# Using objective='regression' and metric='rmse' as suggested by the model description.
# A random_state is set for reproducibility.
lgbm = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42)

# Train the model
lgbm.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = lgbm.predict(X_val)

# Calculate the Root Mean Squared Error (RMSE) on the validation set
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance
print(f"Final Validation Performance: {rmse}")

