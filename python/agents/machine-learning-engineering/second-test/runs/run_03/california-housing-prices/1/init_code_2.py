
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Load the datasets
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Identify features (X) and target (y)
# The original error indicated 'median_house_value' was not found in train_df.
# This code ensures that 'median_house_value' is correctly extracted from the loaded train_df.
X_train_full = train_df.drop('median_house_value', axis=1)
y_train_full = train_df['median_house_value']

# Align columns between training features and test features
# This ensures that the test set has the exact same feature columns in the same order as the training set.
train_cols = X_train_full.columns
X_test = test_df[train_cols]

# Handle missing values: Impute with the median
# It's good practice to fit the imputer on the training data and apply it to both
# training and test sets to prevent data leakage.
for col in X_train_full.columns:
    if X_train_full[col].isnull().any():
        median_val = X_train_full[col].median()
        X_train_full[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)

# Create a validation split for evaluating model performance
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Model Training
# Using RandomForestRegressor as a robust choice for this regression task.
# n_jobs=-1 utilizes all available CPU cores for faster training.
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Train the model on the training portion of the data
model.fit(X_train, y_train)

# Make predictions on the validation set to calculate performance
val_preds = model.predict(X_val)
final_validation_score = np.sqrt(mean_squared_error(y_val, val_preds))

# Print the final validation performance as required
print(f'Final Validation Performance: {final_validation_score}')

# Retrain the model on the full training data (optional, but often done for final predictions)
# This ensures the model learns from all available training examples before predicting on the unseen test set.
full_data_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
full_data_model.fit(X_train_full, y_train_full)

# Make predictions on the actual test set
predictions = full_data_model.predict(X_test)

# Format the submission file
submission_df = pd.DataFrame({'median_house_value': predictions})
submission_df.to_csv('submission.csv', index=False)
