
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import torch
from pytorch_tabnet.tab_model import TabNetRegressor

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
    # If it were present and categorical, one-hot encoding would be more appropriate.
    # For now, drop it as per the base solution's original approach.
    if 'ocean_proximity' in df_copy.columns:
        df_copy = df_copy.drop(columns=['ocean_proximity'])

    if is_train:
        # Separate target variable for training data
        X = df_copy.drop('median_house_value', axis=1)
        y = df_copy['median_house_value']
        # Capture the columns after all initial feature selection/dropping
        final_feature_columns = X.columns.tolist()
    else:
        # For test data, ensure it has the same columns as the processed training data.
        if train_columns is None:
            raise ValueError("train_columns must be provided for test data preprocessing.")
        
        # Align test data columns with training data columns
        missing_cols = set(train_columns) - set(df_copy.columns)
        for c in missing_cols:
            df_copy[c] = np.nan # Add missing columns and fill with NaN
        
        # Ensure the order of columns matches training data and drop extra ones
        X = df_copy[train_columns]
        y = None # No target for test data
        final_feature_columns = train_columns # This will be train_columns

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
    Trains an XGBoost model and a TabNet model using K-fold cross-validation
    and makes ensembled predictions.
    """
    # Ensure reproducibility for both models
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load data
    train_df = load_data(TRAIN_FILE)
    test_df = load_data(TEST_FILE)

    # Preprocess training data
    X_train_processed, y_train, scaler, imputer, train_processed_columns = preprocess_data(train_df, is_train=True)

    # Preprocess test data using the same scaler, imputer, and column set as training
    X_test_processed, _, _ = preprocess_data(test_df, is_train=False, scaler=scaler, imputer=imputer, train_columns=train_processed_columns)

    # Initialize K-fold cross-validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    # Initialize OOF (Out-Of-Fold) and test predictions for both models
    oof_preds_xgb = np.zeros(len(X_train_processed))
    test_preds_xgb = np.zeros(len(X_test_processed))
    
    oof_preds_tabnet = np.zeros(len(X_train_processed))
    test_preds_tabnet = np.zeros(len(X_test_processed))
    
    # XGBoost parameters
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'seed': SEED
    }

    print("Starting K-fold training...")

    for fold, (train_index, val_index) in enumerate(kf.split(X_train_processed, y_train)):
        print(f"--- Fold {fold + 1}/{n_splits} ---")

        # Split data for the current fold
        X_train_fold, X_val_fold = X_train_processed.iloc[train_index], X_train_processed.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # --- XGBoost Model Training ---
        dtrain_xgb = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dval_xgb = xgb.DMatrix(X_val_fold, label=y_val_fold)
        dtest_xgb = xgb.DMatrix(X_test_processed)

        # Reduced early stopping rounds to potentially speed up training
        early_stopping_callback_xgb = xgb.callback.EarlyStopping(
            rounds=30, # Reduced from 50 to make training stop earlier if no improvement
            data_name="validation_0",
            metric_name="rmse",
            save_best=True # Ensures the best iteration model is kept
        )

        model_xgb = xgb.train(
            xgb_params,
            dtrain_xgb,
            num_boost_round=1000, # Max rounds can be high, actual rounds controlled by early stopping
            evals=[(dval_xgb, 'validation_0')],
            callbacks=[early_stopping_callback_xgb],
            verbose_eval=False
        )

        oof_preds_xgb[val_index] = model_xgb.predict(dval_xgb, iteration_range=(0, model_xgb.best_iteration))
        test_preds_xgb += model_xgb.predict(dtest_xgb, iteration_range=(0, model_xgb.best_iteration)) / n_splits
        
        val_rmse_xgb = np.sqrt(mean_squared_error(y_val_fold, oof_preds_xgb[val_index]))
        print(f"XGBoost Fold {fold + 1} Val RMSE: {val_rmse_xgb:.4f}")

        # --- TabNet Model Training ---
        # Convert data to NumPy arrays and ensure correct types and shapes for TabNet
        X_train_fold_np = X_train_fold.values.astype(np.float32)
        y_train_fold_np = y_train_fold.values.astype(np.float32).reshape(-1, 1)
        X_val_fold_np = X_val_fold.values.astype(np.float32)
        y_val_fold_np = y_val_fold.values.astype(np.float32).reshape(-1, 1)
        X_test_processed_np = X_test_processed.values.astype(np.float32)

        model_tabnet = TabNetRegressor(verbose=0,
                                        optimizer_fn=torch.optim.Adam,
                                        optimizer_params=dict(lr=2e-2),
                                        scheduler_params={"step_size":50, "gamma":0.9},
                                        scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                        seed=SEED + fold # Vary seed slightly for diversity across folds
                                        )

        model_tabnet.fit(X_train=X_train_fold_np, y_train=y_train_fold_np,
                         eval_set=[(X_val_fold_np, y_val_fold_np)],
                         eval_metric=['rmse'],
                         max_epochs=70, # Reduced from 100 to decrease training time
                         patience=15, # Reduced from 20 to make early stopping more aggressive
                         batch_size=1024, virtual_batch_size=128)
        
        oof_preds_tabnet[val_index] = model_tabnet.predict(X_val_fold_np).flatten()
        test_preds_tabnet += model_tabnet.predict(X_test_processed_np).flatten() / n_splits

        val_rmse_tabnet = np.sqrt(mean_squared_error(y_val_fold_np, oof_preds_tabnet[val_index].reshape(-1, 1)))
        print(f"TabNet Fold {fold + 1} Val RMSE: {val_rmse_tabnet:.4f}")

    # --- Ensemble Predictions ---
    # Ensemble OOF predictions for final validation score
    oof_preds_ensemble = (oof_preds_xgb + oof_preds_tabnet) / 2
    final_validation_score = np.sqrt(mean_squared_error(y_train, oof_preds_ensemble))
    print(f"Final Validation Performance: {final_validation_score}")

    # Ensemble test predictions
    final_test_preds = (test_preds_xgb + test_preds_tabnet) / 2

    # Create submission file
    submission_df = pd.DataFrame({'median_house_value': final_test_preds})
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission file '{SUBMISSION_FILE}' created successfully.")


if __name__ == "__main__":
    # Ensure the input directory exists for the purpose of running the script directly.
    if not os.path.exists("./input"):
        os.makedirs("./input")
    
    # Create dummy train and test CSVs for local testing if they don't exist,
    # matching the structure provided in the problem description.
    # Added 'ocean_proximity' to dummy data to test its handling.
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
            'median_house_value': np.random.uniform(50000, 500000, 1000),
            'ocean_proximity': np.random.choice(['NEAR BAY', 'INLAND', '<1H OCEAN'], 1000)
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
            'median_income': np.random.uniform(0.5, 10, 200),
            'ocean_proximity': np.random.choice(['NEAR BAY', 'INLAND', '<1H OCEAN'], 200)
        }
        # Add some NaNs to total_bedrooms to test imputer
        dummy_test_data['total_bedrooms'][np.random.choice(200, 10, replace=False)] = np.nan
        pd.DataFrame(dummy_test_data).to_csv(TEST_FILE, index=False)
        print(f"Created dummy {TEST_FILE}")

    train_and_predict()
