
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# Load datasets
# Based on the problem description: "All the provided input data is stored in "./input" directory."
# The FileNotFoundError indicates the previous attempt did not find the files at the specified path.
# We will use the explicit path `input/train.csv` and `input/test.csv` as per the problem statement.
# If a FileNotFoundError persists, it indicates an issue with the execution environment's file system setup,
# not a bug in the code's path interpretation of the provided instructions.
train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')

# Separate target variable
X_train = train_df.drop('median_house_value', axis=1)
y_train = train_df['median_house_value']
X_test = test_df.copy()

# Feature Engineering (apply to both train and test)
def create_features(df):
    # Create new features to potentially improve model performance
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    # Handle potential division by zero if 'households' column contains zeros
    df['rooms_per_household'] = df['rooms_per_household'].replace([np.inf, -np.inf], np.nan)

    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    # Handle potential division by zero if 'total_rooms' column contains zeros
    df['bedrooms_per_room'] = df['bedrooms_per_room'].replace([np.inf, -np.inf], np.nan)

    df['population_per_household'] = df['population'] / df['households']
    # Handle potential division by zero if 'households' column contains zeros
    df['population_per_household'] = df['population_per_household'].replace([np.inf, -np.inf], np.nan)
    return df

X_train = create_features(X_train)
X_test = create_features(X_test)

# Handle missing values using SimpleImputer (median strategy)
# It's crucial to fit the imputer ONLY on the training data and then transform both train and test data.
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Convert the imputed NumPy arrays back to Pandas DataFrames, preserving column names.
# This ensures consistency for model training and prediction.
X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns)
X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)

# Model Training: Random Forest Regressor
# Using n_estimators=100 as a reasonable default. random_state for reproducibility.
# n_jobs=-1 utilizes all available CPU cores for faster training and prediction.
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Cross-validation for robust performance estimation using Root Mean Squared Error (RMSE)
# The scoring 'neg_mean_squared_error' is used because scikit-learn's cross_val_score
# expects a score to maximize. Negative MSE is equivalent to minimizing MSE.
# We then take the square root of the negative score to get RMSE.
cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                            scoring='neg_mean_squared_error', n_jobs=-1)
rmse_scores = np.sqrt(-cv_scores)
final_validation_score = rmse_scores.mean()

# Train the model on the entire training dataset before making final predictions
model.fit(X_train, y_train)

# Make predictions on the test dataset
predictions = model.predict(X_test)

# Create submission file in the specified format
submission_df = pd.DataFrame({'median_house_value': predictions})

# Save the submission file without index and with the correct header
submission_df.to_csv('submission.csv', index=False, header=True)

# Print the final validation performance as required
print(f'Final Validation Performance: {final_validation_score}')
