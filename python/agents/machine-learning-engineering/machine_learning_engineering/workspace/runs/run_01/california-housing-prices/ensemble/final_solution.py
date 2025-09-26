
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import os

def main():
    # Define paths
    input_dir = "./input"
    train_file = os.path.join(input_dir, "train.csv")
    test_file = os.path.join(input_dir, "test.csv")
    submission_file = "submission.csv"

    # Load data
    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure 'train.csv' and 'test.csv' are in the '{input_dir}' directory.")
        return # Exit if files are not found

    # Separate target variable
    X = train_df.drop("median_house_value", axis=1)
    y = train_df["median_house_value"]

    # Identify numerical features (all are in this dataset)
    numerical_features = X.select_dtypes(include=np.number).columns

    # Impute missing values for numerical features
    # Use median strategy for robustness against outliers
    imputer = SimpleImputer(strategy="median")

    # Fit imputer on training data and transform both train and test data
    X_imputed = pd.DataFrame(imputer.fit_transform(X[numerical_features]), columns=numerical_features, index=X.index)
    test_imputed = pd.DataFrame(imputer.transform(test_df[numerical_features]), columns=numerical_features, index=test_df.index)

    # Train-validation split for performance evaluation
    X_train_val, X_test_val, y_train_val, y_val = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Model training
    # Using RandomForestRegressor as it's robust and generally performs well
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_val, y_train_val)

    # Evaluate model on the validation set
    val_predictions = model.predict(X_test_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))

    print(f"Final Validation Performance: {rmse}")

    # Retrain model on the full training data for final predictions
    model_final = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_final.fit(X_imputed, y)

    # Make predictions on the actual test set
    final_predictions = model_final.predict(test_imputed)

    # Create submission file
    submission_df = pd.DataFrame({'median_house_value': final_predictions})
    submission_df.to_csv(submission_file, index=False, header=True) # Ensure header is written
    print(f"Submission file created successfully: {submission_file}")

if __name__ == "__main__":
    main()
