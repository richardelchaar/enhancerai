
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Load the training data
train_df = pd.read_csv('./input/train.csv')

# Separate target variable from features
# The target variable is 'median_house_value'
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Handle missing values
# For simplicity and robustness in this initial solution,
# we impute missing 'total_bedrooms' values with the median of that column.
# This is a common and effective strategy for handling sporadic missing numerical data.
if 'total_bedrooms' in X.columns and X['total_bedrooms'].isnull().any():
    median_total_bedrooms = X['total_bedrooms'].median()
    X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

# Split the training data into training and validation sets
# A 80/20 split is used, and a fixed random_state ensures reproducibility of the split.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LightGBM Regressor
# We use the 'regression' objective which minimizes L2 loss (MSE).
# The 'metric' parameter is set to 'rmse' for consistency with the evaluation metric.
# random_state is set for reproducibility of the model training.
# verbose=-1 is used to suppress all verbose output during training, as required.
model = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Evaluate the Model using Root Mean Squared Error (RMSE) on the validation set
# First, calculate Mean Squared Error, then take its square root.
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance in the specified format
print(f"Final Validation Performance: {rmse_val}")
