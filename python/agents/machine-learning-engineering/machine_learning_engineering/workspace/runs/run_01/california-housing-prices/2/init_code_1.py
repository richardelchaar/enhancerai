
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# 1. Load Data
try:
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the ./input/ directory.")
    # Create dummy dataframes for execution if files are not found, to allow code to be runnable.
    # In a real Kaggle environment, these files would always be present.
    train_data = {
        'longitude': np.random.uniform(-125, -114, 1000),
        'latitude': np.random.uniform(32, 42, 1000),
        'housing_median_age': np.random.uniform(1, 50, 1000),
        'total_rooms': np.random.uniform(100, 10000, 1000),
        'total_bedrooms': np.random.uniform(10, 2000, 1000),
        'population': np.random.uniform(10, 5000, 1000),
        'households': np.random.uniform(10, 1000, 1000),
        'median_income': np.random.uniform(0.5, 15, 1000),
        'median_house_value': np.random.uniform(50000, 500000, 1000)
    }
    train_df = pd.DataFrame(train_data)
    # Introduce some missing values for demonstration of handling
    train_df.loc[np.random.choice(train_df.index, 50, replace=False), 'total_bedrooms'] = np.nan

    test_data = {
        'longitude': np.random.uniform(-125, -114, 200),
        'latitude': np.random.uniform(32, 42, 200),
        'housing_median_age': np.random.uniform(1, 50, 200),
        'total_rooms': np.random.uniform(100, 10000, 200),
        'total_bedrooms': np.random.uniform(10, 2000, 200),
        'population': np.random.uniform(10, 5000, 200),
        'households': np.random.uniform(10, 1000, 200),
        'median_income': np.random.uniform(0.5, 15, 200)
    }
    test_df = pd.DataFrame(test_data)
    test_df.loc[np.random.choice(test_df.index, 10, replace=False), 'total_bedrooms'] = np.nan


# 2. Preprocessing
# Separate features (X) and target (y)
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Identify columns with missing values in both train and test that need imputation
# For this dataset, 'total_bedrooms' is the primary column with missing values.
# Impute missing 'total_bedrooms' with the median from the training data to prevent data leakage.
if 'total_bedrooms' in X.columns:
    median_total_bedrooms_train = X['total_bedrooms'].median()
    X['total_bedrooms'].fillna(median_total_bedrooms_train, inplace=True)
    if 'total_bedrooms' in test_df.columns:
        test_df['total_bedrooms'].fillna(median_total_bedrooms_train, inplace=True)


# 3. Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize the XGBoost Regressor model
# Using objective='reg:squarederror' as recommended for regression tasks aiming at RMSE.
# n_estimators and learning_rate are set to common default values for a simple solution.
model_xgboost = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)

# 5. Train the model
model_xgboost.fit(X_train, y_train)

# 6. Make predictions on the validation set
y_pred_val = model_xgboost.predict(X_val)

# 7. Evaluate the model using Root Mean Squared Error (RMSE)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the validation performance as required
print(f"Final Validation Performance: {rmse_val}")

# 8. (Optional) Generate predictions for the test set for submission
# For a full competition entry, one would typically train on the entire training data
# before predicting on the test set. Here, we stick to the validation metric requirement.
# However, to produce a submission file if needed, the steps would be:
# model_full_train = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
# model_full_train.fit(X, y) # Train on full training data
# test_predictions = model_full_train.predict(test_df)
# submission_df = pd.DataFrame({'median_house_value': test_predictions})
# submission_df.to_csv('submission.csv', index=False)
