
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(train_path, test_path):
    """
    Loads training and testing data, applies feature engineering, handles missing values,
    and scales numerical features.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Separate target variable from training data
    y_train = train_df['median_house_value']
    X_train = train_df.drop('median_house_value', axis=1)

    # Combine for consistent preprocessing
    # Keeping track of original indices to split later
    X_test_original_index = test_df.index
    
    # Concatenate train and test data for unified preprocessing, keeping track of origin
    # Create a 'source' column to differentiate between train and test
    X_train['source'] = 'train'
    test_df['source'] = 'test'

    combined_df = pd.concat([X_train, test_df], ignore_index=True)

    # --- Feature Engineering ---
    # Create new features as per common practice for this dataset
    combined_df['rooms_per_household'] = combined_df['total_rooms'] / combined_df['households']
    combined_df['bedrooms_per_room'] = combined_df['total_bedrooms'] / combined_df['total_rooms']
    combined_df['population_per_household'] = combined_df['population'] / combined_df['households']

    # Drop original total_rooms and total_bedrooms if desired,
    # but for RandomForestRegressor, keeping them might also be fine.
    # For simplicity and potential multi-collinearity, we can keep them for now.

    # Identify numerical features for imputation and scaling
    numerical_cols = combined_df.select_dtypes(include=np.number).columns.tolist()

    # --- Handle Missing Values using SimpleImputer (median strategy) --- [2, 4, 5, 6, 8, 26]
    # 'total_bedrooms' is a common column with missing values in this dataset.
    imputer = SimpleImputer(strategy='median')
    combined_df[numerical_cols] = imputer.fit_transform(combined_df[numerical_cols])

    # --- Feature Scaling using StandardScaler --- [3, 7, 9, 12, 16]
    scaler = StandardScaler()
    combined_df[numerical_cols] = scaler.fit_transform(combined_df[numerical_cols])

    # Split back into training and testing sets
    X_train_processed = combined_df[combined_df['source'] == 'train'].drop('source', axis=1)
    X_test_processed = combined_df[combined_df['source'] == 'test'].drop('source', axis=1)
    
    # Ensure column order consistency
    X_test_processed = X_test_processed[X_train_processed.columns]

    return X_train_processed, y_train, X_test_processed, X_test_original_index

def train_and_predict(X_train, y_train, X_test):
    """
    Trains a RandomForestRegressor model, performs cross-validation,
    and makes predictions on the test set.
    """
    # Initialize RandomForestRegressor [11, 13, 15, 17, 20]
    # Using a fixed random_state for reproducibility
    # n_estimators and max_features are common hyperparameters to tune
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_features=0.8, max_depth=15, min_samples_leaf=5)

    # Perform cross-validation [10, 18, 19, 22, 24]
    # scoring='neg_mean_squared_error' is used because cross_val_score maximizes the score,
    # and we want to minimize RMSE, so we negate MSE.
    cv_scores = cross_val_score(model, X_train, y_train,
                                 scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

    # Convert negative MSE scores to positive RMSE scores
    rmse_scores = np.sqrt(-cv_scores)
    final_validation_score = rmse_scores.mean()

    print(f'Final Validation Performance: {final_validation_score}')

    # Train the model on the full training data
    model.fit(X_train, y_train)

    # Make predictions on the processed test data
    predictions = model.predict(X_test)

    return predictions

if __name__ == "__main__":
    # Define paths to datasets
    train_file = "./input/train.csv"
    test_file = "./input/test.csv"

    # Load and preprocess data
    X_train_processed, y_train, X_test_processed, _ = load_and_preprocess_data(train_file, test_file)

    # Train model and get predictions
    test_predictions = train_and_predict(X_train_processed, y_train, X_test_processed)

    # Prepare submission file
    submission_df = pd.DataFrame({'median_house_value': test_predictions})

    # Save submission file in the specified format
    submission_df.to_csv('submission.csv', index=False, header=False)

    print("Submission file created successfully!")

