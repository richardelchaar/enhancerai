
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
try:
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
except FileNotFoundError:
    # Fallback for local execution if 'input' directory isn't present
    print("'/input/' directory not found. Attempting to load from current directory.")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")


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

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the LightGBM Regressor
lgb_params = {
    'objective': 'regression',  # L2 loss for RMSE optimization
    'metric': 'rmse',
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,    # Feature subsampling
    'bagging_fraction': 0.8,    # Data subsampling (row-wise)
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,              # Suppress verbose output during training
    'n_jobs': -1,               # Use all available cores
    'seed': 42,
    'boosting_type': 'gbdt',
}

model = lgb.LGBMRegressor(**lgb_params)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='rmse',
          callbacks=[lgb.early_stopping(100, verbose=False)]) # Early stopping to prevent overfitting

# Make predictions on the validation set
y_val_pred = model.predict(X_val)

# Calculate RMSE on the validation set
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f'Final Validation Performance: {rmse_val}')

# Make predictions on the test set
test_predictions = model.predict(X_test)

# Ensure predictions are non-negative, as median_house_value cannot be negative
test_predictions[test_predictions < 0] = 0

# Create submission file
submission_df = pd.DataFrame({'median_house_value': test_predictions})
submission_df.to_csv("submission.csv", index=False, header=False)

print("Submission file 'submission.csv' created successfully.")
