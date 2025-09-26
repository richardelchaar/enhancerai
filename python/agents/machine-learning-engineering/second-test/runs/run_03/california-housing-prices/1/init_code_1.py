
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import warnings

# Suppress all warnings for cleaner output, specifically related to division by zero producing inf
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate target variable for training data
y_train_full = train_df['median_house_value']
X_train_full = train_df.drop('median_house_value', axis=1)

# Combine for consistent preprocessing (training and test features)
combined_features = pd.concat([X_train_full, test_df], ignore_index=True)

# Feature Engineering - addressing potential ZeroDivisionError
# Division by zero can result in np.inf. We convert these to np.nan
# so that the SimpleImputer can handle them effectively.

# rooms_per_household
# If 'households' is zero, this division would result in inf.
combined_features['rooms_per_household'] = combined_features['total_rooms'] / combined_features['households']
combined_features['rooms_per_household'].replace([np.inf, -np.inf], np.nan, inplace=True)

# bedrooms_per_room
# If 'total_rooms' is zero, this division would result in inf.
combined_features['bedrooms_per_room'] = combined_features['total_bedrooms'] / combined_features['total_rooms']
combined_features['bedrooms_per_room'].replace([np.inf, -np.inf], np.nan, inplace=True)

# population_per_household
# If 'households' is zero, this division would result in inf.
combined_features['population_per_household'] = combined_features['population'] / combined_features['households']
combined_features['population_per_household'].replace([np.inf, -np.inf], np.nan, inplace=True)

# Impute missing values for all numerical columns.
# This handles original NaNs (e.g., in 'total_bedrooms') and any NaNs created
# during feature engineering from division by zero (converted from inf).
imputer = SimpleImputer(strategy='median')
numerical_cols = combined_features.select_dtypes(include=np.number).columns
combined_features[numerical_cols] = imputer.fit_transform(combined_features[numerical_cols])

# Split back into training and testing sets
X_train_processed = combined_features.iloc[:len(X_train_full)]
X_test_processed = combined_features.iloc[len(X_train_full):]

# Validation split for model evaluation
X_train, X_val, y_train, y_val = train_test_split(X_train_processed, y_train_full, test_size=0.2, random_state=42)

# Model Training
# Using RandomForestRegressor for robust performance, n_jobs=-1 utilizes all available cores.
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Validation
val_predictions = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f'Final Validation Performance: {rmse}')

# Make predictions on the test set
test_predictions = model.predict(X_test_processed)

# Format the predictions for submission
submission_df = pd.DataFrame({'median_house_value': test_predictions})

# Print the predictions to standard output in the specified format
# The to_string method with index=False will produce the desired output format,
# including the header by default.
print(submission_df.to_string(index=False))
