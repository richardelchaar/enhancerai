
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load datasets
# All data is assumed to be in the './input' directory as per task description.
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Preprocessing: Handle missing values
# The 'total_bedrooms' column often contains a few missing values in this dataset.
# Filling with the median is a robust strategy for tree-based models and simple approach.
train_df['total_bedrooms'].fillna(train_df['total_bedrooms'].median(), inplace=True)
test_df['total_bedrooms'].fillna(test_df['total_bedrooms'].median(), inplace=True)

# Define features (X) and target (y)
# Features are all columns except 'median_house_value' in the training set.
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income']
target = 'median_house_value'

X = train_df[features]
y = train_df[target]

# Split training data into a training set and a hold-out validation set
# A 80/20 split is commonly used. random_state ensures reproducibility.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LightGBM Regressor model
# objective='regression' specifies the task type.
# metric='rmse' sets the evaluation metric for training monitoring (though we calculate explicitly).
# random_state for reproducibility of the model's internal randomness.
lgbm_model = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42)

# Train the model on the training data
lgbm_model.fit(X_train, y_train)

# Make predictions on the hold-out validation set
y_pred_val = lgbm_model.predict(X_val)

# Evaluate the model on the validation set using Root Mean Squared Error (RMSE)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

# Print the final validation performance as required
print(f"Final Validation Performance: {rmse_val}")

# Prepare test data for final predictions
X_test = test_df[features]

# Generate predictions on the actual test set
test_predictions = lgbm_model.predict(X_test)

# Create the submission file in the specified format
submission_df = pd.DataFrame({'median_house_value': test_predictions})
submission_df.to_csv('submission.csv', index=False)
