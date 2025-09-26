
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Load the datasets
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate target variable
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Identify numerical and categorical features dynamically
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Define preprocessing steps
# Numerical features pipeline: impute missing values with the median
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Categorical features pipeline: impute missing values with the most frequent value, then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer to apply different transformations
# to numerical and categorical features. This addresses the bug by not dropping categorical features.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Drop any columns not specified
)

# Define the model
# RandomForestRegressor is chosen for its robustness and good performance.
model = RandomForestRegressor(random_state=42) # Added random_state for reproducibility

# Create the full pipeline
# This pipeline first preprocesses the data and then applies the regression model.
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', model)])

# Train the model on the full training data for final predictions
pipeline.fit(X, y)

# --- Validation Performance Calculation ---
# Split the training data to evaluate the model's performance on unseen data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a fresh pipeline instance for validation or re-fit the existing one on the training split
# For cleaner separation, we can re-create or simply re-fit the `pipeline` instance.
# Re-fitting is fine as it resets its internal state.
validation_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('regressor', model)])
validation_pipeline.fit(X_train, y_train)
val_predictions = validation_pipeline.predict(X_val)

# Calculate Root Mean Squared Error (RMSE) for validation
final_validation_score = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f'Final Validation Performance: {final_validation_score}')

# --- Generate Predictions for Submission ---
# Make predictions on the provided test data
test_predictions = pipeline.predict(test_df)

# Format predictions for submission as specified
print("median_house_value")
for pred in test_predictions:
    print(pred)

