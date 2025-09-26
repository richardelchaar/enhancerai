
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Preprocessing
# Fill missing total_bedrooms with the median
train_df['total_bedrooms'].fillna(train_df['total_bedrooms'].median(), inplace=True)
test_df['total_bedrooms'].fillna(test_df['total_bedrooms'].median(), inplace=True)

# Define features and target
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income']
target = 'median_house_value'

X = train_df[features]
y = train_df[target]
X_test = test_df[features]

# Split training data for validation (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Regressor
# objective='reg:squarederror' for regression tasks (minimizes squared error)
# eval_metric='rmse' for evaluating root mean squared error during training if applicable
# For a simple solution, we use default hyperparameters and a fixed random_state for reproducibility.
model = xgb.XGBRegressor(objective='reg:squarederror',
                         eval_metric='rmse',
                         n_estimators=1000, # A reasonable number of estimators
                         learning_rate=0.05, # Default is 0.3, reducing it might help
                         random_state=42,
                         n_jobs=-1) # Use all available cores

# Train the model
# For a simple solution, we don't use early stopping here as it involves tuning
model.fit(X_train, y_train,
          # For evaluation during training, uncomment the following line.
          # eval_set=[(X_val, y_val)], verbose=False
         )

# Make predictions on the validation set
y_val_pred = model.predict(X_val)

# Calculate RMSE on the validation set
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Print the final validation performance
print(f'Final Validation Performance: {rmse_val}')

# Make predictions on the actual test set for submission
test_predictions = model.predict(X_test)

# Ensure predictions are non-negative, as median_house_value cannot be negative
test_predictions[test_predictions < 0] = 0

# Create submission file
submission_df = pd.DataFrame({'median_house_value': test_predictions})
# The problem asks for no header in the submission file
submission_df.to_csv("submission.csv", index=False, header=False)
