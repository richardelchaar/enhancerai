
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

# Define file paths
TRAIN_FILE = "./input/train.csv"
TEST_FILE = "./input/test.csv"
SUBMISSION_FILE = "submission.csv"

def load_data(file_path):
    """Loads a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def preprocess_data(df, is_train=True, scaler=None, imputer=None, train_columns=None):
    """
    Preprocesses the data by handling missing values, scaling features,
    and separating target variable if it's training data.
    Ensures consistent column order and presence between training and test sets.
    """
    df_copy = df.copy() # Work on a copy to avoid modifying original df

    # Handle 'ocean_proximity' if it exists.
    # The provided dataset description in the prompt doesn't include it,
    # but the original code had this logic, implying it might be present in some datasets.
    # If it were present and categorical, one-hot encoding would be more appropriate.
    if 'ocean_proximity' in df_copy.columns:
        df_copy = df_copy.drop(columns=['ocean_proximity'])

    if is_train:
        X = df_copy.drop('median_house_value', axis=1)
        y = df_copy['median_house_value']
        # Capture the columns after all initial feature selection/dropping
        final_feature_columns = X.columns.tolist()
    else:
        # For test data, ensure it has the same columns as the processed training data.
        # This is critical if the original test_df has different columns than train_df
        # after initial drops (like 'ocean_proximity' potentially).
        if train_columns is None:
            raise ValueError("train_columns must be provided for test data preprocessing.")
        
        # Align test data columns with training data columns
        missing_cols = set(train_columns) - set(df_copy.columns)
        for c in missing_cols:
            df_copy[c] = np.nan # Add missing columns and fill with NaN
        
        # Ensure the order of columns matches training data and drop extra ones
        X = df_copy[train_columns]
        y = None # No target for test data
        final_feature_columns = train_columns


    # Impute missing values
    if imputer is None:
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
    else:
        X_imputed = imputer.transform(X)
    
    # Recreate DataFrame with proper columns and index
    X_imputed_df = pd.DataFrame(X_imputed, columns=final_feature_columns, index=X.index)

    # Scale numerical features
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed_df)
    else:
        X_scaled = scaler.transform(X_imputed_df)
    
    # Recreate DataFrame with proper columns and index
    X_scaled_df = pd.DataFrame(X_scaled, columns=final_feature_columns, index=X.index)

    if is_train:
        return X_scaled_df, y, scaler, imputer, final_feature_columns
    else:
        return X_scaled_df, scaler, imputer

def train_and_predict():
    """
    Trains an XGBoost model using K-fold cross-validation and makes predictions.
    """
    # Load data
    train_df = load_data(TRAIN_FILE)
    test_df = load_data(TEST_FILE)

    # Preprocess training data
    X_train, y_train, scaler, imputer, train_processed_columns = preprocess_data(train_df, is_train=True)

    # Preprocess test data using the same scaler, imputer, and column set as training
    X_test, _, _ = preprocess_data(test_df, is_train=False, scaler=scaler, imputer=imputer, train_columns=train_processed_columns)

    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'seed': 42
    }

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    validation_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dval = xgb.DMatrix(X_val_fold, label=y_val_fold)
        dtest = xgb.DMatrix(X_test)

        # EarlyStopping callback
        # The 'minimize' argument is deprecated/removed in newer XGBoost versions for EarlyStopping callback.
        # The metric_name and data_name imply whether to minimize or maximize automatically based on the metric.
        early_stopping_callback = xgb.callback.EarlyStopping(
            rounds=50,  # Number of rounds with no improvement
            # minimize=True, # This argument is no longer supported
            data_name="validation_0",
            metric_name="rmse"
        )

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'validation_0')],
            callbacks=[early_stopping_callback],
            verbose_eval=False
        )

        # Predict using the best iteration found by early stopping
        oof_preds[val_index] = model.predict(dval, iteration_range=(0, model.best_iteration))
        test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration)) / n_splits
        
        val_rmse = np.sqrt(mean_squared_error(y_val_fold, oof_preds[val_index]))
        validation_scores.append(val_rmse)

    # Calculate the average validation score from K-fold
    final_validation_score = np.mean(validation_scores)
    print(f"Final Validation Performance: {final_validation_score}")

    # Create submission file
    submission_df = pd.DataFrame({'median_house_value': test_preds})
    submission_df.to_csv(SUBMISSION_FILE, index=False)

if __name__ == "__main__":
    # Ensure the input directory exists for the purpose of running the script directly.
    if not os.path.exists("./input"):
        os.makedirs("./input")
    
    # Create dummy train and test CSVs for local testing if they don't exist,
    # matching the structure provided in the problem description.
    if not os.path.exists(TRAIN_FILE):
        dummy_train_data = {
            'longitude': np.random.uniform(-125, -114, 1000),
            'latitude': np.random.uniform(32, 42, 1000),
            'housing_median_age': np.random.randint(1, 50, 1000),
            'total_rooms': np.random.randint(100, 5000, 1000),
            'total_bedrooms': np.random.randint(50, 1000, 1000),
            'population': np.random.randint(100, 3000, 1000),
            'households': np.random.randint(50, 800, 1000),
            'median_income': np.random.uniform(0.5, 10, 1000),
            'median_house_value': np.random.uniform(50000, 500000, 1000)
        }
        # Add some NaNs to total_bedrooms to test imputer
        dummy_train_data['total_bedrooms'][np.random.choice(1000, 50, replace=False)] = np.nan
        pd.DataFrame(dummy_train_data).to_csv(TRAIN_FILE, index=False)
        print(f"Created dummy {TRAIN_FILE}")

    if not os.path.exists(TEST_FILE):
        dummy_test_data = {
            'longitude': np.random.uniform(-125, -114, 200),
            'latitude': np.random.uniform(32, 42, 200),
            'housing_median_age': np.random.randint(1, 50, 200),
            'total_rooms': np.random.randint(100, 5000, 200),
            'total_bedrooms': np.random.randint(50, 1000, 200),
            'population': np.random.randint(100, 3000, 200),
            'households': np.random.randint(50, 800, 200),
            'median_income': np.random.uniform(0.5, 10, 200)
        }
        # Add some NaNs to total_bedrooms to test imputer
        dummy_test_data['total_bedrooms'][np.random.choice(200, 10, replace=False)] = np.nan
        pd.DataFrame(dummy_test_data).to_csv(TEST_FILE, index=False)
        print(f"Created dummy {TEST_FILE}")

    train_and_predict()

