
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load the datasets
# As per the task description, data is available in the './input' directory.
train_df = pd.read_csv("./input/train.csv")
# test_df is not used for validation performance, but for consistency in preprocessing it's good to load it.
# However, for the final output (validation performance), we only need train_df.
# test_df = pd.read_csv("./input/test.csv")

# Separate target variable from features in the training data
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Identify numerical features for imputation.
# In this dataset, all features are numerical.
numerical_cols = X.select_dtypes(include=np.number).columns

# Initialize SimpleImputer to handle missing values.
# We use the median strategy as it's robust to outliers and suitable for a simple solution.
# It's crucial to fit the imputer on the training data ONLY to prevent data leakage.
imputer = SimpleImputer(strategy='median')

# Fit the imputer on the training features and transform them.
X_imputed_array = imputer.fit_transform(X[numerical_cols])
X_imputed = pd.DataFrame(X_imputed_array, columns=numerical_cols, index=X.index)

# Split the processed training data into training and validation sets.
# This allows us to evaluate the model's performance on unseen data.
# A 80/20 split is a common practice, and random_state ensures reproducibility.
X_train_split, X_val, y_train_split, y_val = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize the XGBoost Regressor model as described in the model description.
# Common parameters for regression: objective='reg:squarederror' for RMSE.
# Parameters like n_estimators, learning_rate, max_depth are taken from the example code.
# random_state ensures reproducibility.
xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                             n_estimators=100,
                             learning_rate=0.1,
                             max_depth=5,
                             random_state=42,
                             n_jobs=-1) # Use all available CPU cores for faster training

# Train the XGBoost model on the training split.
xgb_model.fit(X_train_split, y_train_split)

# Make predictions on the hold-out validation set.
y_pred_val = xgb_model.predict(X_val)

# Evaluate the model's performance using Root Mean Squared Error (RMSE),
# which is the specified metric for this task.
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance in the required format.
print(f"Final Validation Performance: {rmse_val}")
