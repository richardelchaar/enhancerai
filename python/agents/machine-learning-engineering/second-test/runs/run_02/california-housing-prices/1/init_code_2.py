
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import os

def train_and_predict():
    # Load the datasets
    # The input directory is "./input" as specified in the problem description.
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")

    # Separate target variable from features
    X = train_df.drop("median_house_value", axis=1)
    y = train_df["median_house_value"]

    # Identify numerical and categorical columns
    # Based on the error analysis provided, the most likely bug is unhandled categorical features.
    # We will set up a robust pipeline that handles both, even if the provided snippets
    # show only numerical features. This ensures the code works for datasets that might
    # include categorical columns (like 'ocean_proximity' in California Housing datasets).
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=np.number).columns

    # Create preprocessing pipelines for numerical and categorical features
    # Numerical pipeline: Impute missing values with the mean, then scale features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Impute missing values with the most frequent value, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Handles potential NaNs in categorical columns
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # Handles new categories in test set
    ])

    # Combine preprocessing steps using ColumnTransformer
    # This applies different transformers to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough' # Pass through any columns not explicitly selected (e.g., if there are unexpected types)
    )

    # Create the full machine learning pipeline
    # It first applies the preprocessor and then trains a RandomForestRegressor.
    # A random_state is added for reproducibility.
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(random_state=42))])

    # Split the training data to create a validation set
    # This allows for evaluating the model's performance on unseen data during development.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model using the training data
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_val_pred = model.predict(X_val)

    # Calculate and print the Root Mean Squared Error (RMSE) on the validation set
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    print(f'Final Validation Performance: {rmse_val}')

    # Make predictions on the actual test set
    test_predictions = model.predict(test_df)

    # Format the predictions into the specified submission format
    output_df = pd.DataFrame({'median_house_value': test_predictions})
    # Save the predictions to a CSV file without the header and index
    output_df.to_csv("submission.csv", index=False, header=False)

# Entry point for the script execution
if __name__ == "__main__":
    train_and_predict()

