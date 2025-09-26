
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

# 1. Load Data
# Assuming train.csv and test.csv are available in the ./input/ directory
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# 2. Preprocessing
# Separate features (X) and target (y)
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Impute missing 'total_bedrooms' with the median from the training data to prevent data leakage.
# The 'total_bedrooms' column is the only one identified with potential missing values in this dataset context.
if 'total_bedrooms' in X.columns:
    median_total_bedrooms_train = X['total_bedrooms'].median()
    X['total_bedrooms'].fillna(median_total_bedrooms_train, inplace=True)
    # Also apply the same imputation to the test set using the median from the training data
    if 'total_bedrooms' in test_df.columns:
        test_df['total_bedrooms'].fillna(median_total_bedrooms_train, inplace=True)

# 3. Split data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize the LightGBM Regressor model
# 'objective=regression' is appropriate for this task as it's a regression problem.
# Using default n_estimators and learning_rate as per instructions for a simple solution.
model_lightgbm = LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1, random_state=42)

# 5. Train the model
model_lightgbm.fit(X_train, y_train)

# 6. Make predictions on the validation set
y_pred_val = model_lightgbm.predict(X_val)

# 7. Evaluate the model using Root Mean Squared Error (RMSE)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the validation performance in the required format
print(f"Final Validation Performance: {rmse_val}")
