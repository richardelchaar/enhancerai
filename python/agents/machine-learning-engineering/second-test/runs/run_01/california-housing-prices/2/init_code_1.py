
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Ensure the input directory exists (for local testing purposes)
if not os.path.exists("./input"):
    os.makedirs("./input")

# Create dummy train.csv and test.csv if they don't exist for demonstration
# In a real Kaggle environment, these files would already be present.
if not os.path.exists("./input/train.csv"):
    train_data = {
        'longitude': np.random.uniform(-125, -114, 1000),
        'latitude': np.random.uniform(32, 42, 1000),
        'housing_median_age': np.random.randint(1, 52, 1000),
        'total_rooms': np.random.randint(100, 10000, 1000),
        'total_bedrooms': np.random.randint(50, 2000, 1000),
        'population': np.random.randint(50, 5000, 1000),
        'households': np.random.randint(20, 1500, 1000),
        'median_income': np.random.uniform(0.5, 15, 1000),
        'median_house_value': np.random.uniform(50000, 500000, 1000)
    }
    # Introduce some NaNs in total_bedrooms for testing imputation
    nan_indices = np.random.choice(len(train_data['total_bedrooms']), 20, replace=False)
    for idx in nan_indices:
        train_data['total_bedrooms'][idx] = np.nan
    pd.DataFrame(train_data).to_csv("./input/train.csv", index=False)

if not os.path.exists("./input/test.csv"):
    test_data = {
        'longitude': np.random.uniform(-125, -114, 200),
        'latitude': np.random.uniform(32, 42, 200),
        'housing_median_age': np.random.randint(1, 52, 200),
        'total_rooms': np.random.randint(100, 10000, 200),
        'total_bedrooms': np.random.randint(50, 2000, 200),
        'population': np.random.randint(50, 5000, 200),
        'households': np.random.randint(20, 1500, 200),
        'median_income': np.random.uniform(0.5, 15, 200),
    }
    # Introduce some NaNs in total_bedrooms for testing imputation
    nan_indices = np.random.choice(len(test_data['total_bedrooms']), 5, replace=False)
    for idx in nan_indices:
        test_data['total_bedrooms'][idx] = np.nan
    pd.DataFrame(test_data).to_csv("./input/test.csv", index=False)


# Load datasets
try:
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
except FileNotFoundError:
    print("Ensure 'train.csv' and 'test.csv' are in the './input' directory.")
    # Exit or handle error appropriately for a robust system, here we will create dummy data
    # and proceed for demonstration purposes as per the above dummy data creation.
    # In a real environment, this would likely be an unrecoverable error.
    pass


# Define features and target
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income']
target = 'median_house_value'

X = train_df[features]
y = train_df[target]
X_test = test_df[features]

# Impute missing values (using mean imputation)
# Calculate mean from the training data only
for col in ['total_bedrooms']: # Only total_bedrooms is expected to have missing values based on typical datasets
    if X[col].isnull().any():
        mean_val = X[col].mean()
        X[col] = X[col].fillna(mean_val)
        X_test[col] = X_test[col].fillna(mean_val) # Apply the same mean to test set

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM Model Training
lgb_params = {
    'objective': 'regression_l1', # MAE
    'metric': 'rmse',
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}

model = lgb.LGBMRegressor(**lgb_params)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='rmse',
          callbacks=[lgb.early_stopping(100, verbose=False)]) # Early stopping after 100 rounds without improvement

# Predict on validation set
val_predictions = model.predict(X_val)
final_validation_score = np.sqrt(mean_squared_error(y_val, val_predictions))

# Print final validation performance as required
print(f'Final Validation Performance: {final_validation_score}')

# Predict on the test set
test_predictions = model.predict(X_test)

# Create submission file
submission_df = pd.DataFrame({'median_house_value': test_predictions})

# Ensure the output directory exists
if not os.path.exists("./submission"):
    os.makedirs("./submission")

submission_df.to_csv('submission.csv', index=False, header=True)

print("Submission file 'submission.csv' created successfully.")
