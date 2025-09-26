
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(train_path, test_path):
    """
    Loads training and testing data, applies feature engineering, and sets up
    a ColumnTransformer for consistent preprocessing.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Separate target variable from training data
    y_train_raw = train_df['median_house_value']
    X_train_raw = train_df.drop('median_house_value', axis=1)

    # --- Feature Engineering (applied to raw dataframes) ---
    # These new features will be part of the numerical columns for the preprocessor
    # Apply to both train and test to ensure consistent columns for the preprocessor fit
    def apply_feature_engineering(df):
        df_copy = df.copy()
        df_copy['rooms_per_household'] = df_copy['total_rooms'] / df_copy['households']
        df_copy['bedrooms_per_room'] = df_copy['total_bedrooms'] / df_copy['total_rooms']
        df_copy['population_per_household'] = df_copy['population'] / df_copy['households']
        return df_copy

    X_train_fe = apply_feature_engineering(X_train_raw)
    X_test_fe = apply_feature_engineering(test_df)

    # Identify numerical and categorical columns
    # The 'ocean_proximity' column was not present in the provided dataset schema.
    # Therefore, it has been removed from the categorical_cols list.
    categorical_cols = [] 
    
    # Ensure all original and engineered numerical columns are included
    all_cols_after_fe = X_train_fe.columns.tolist()
    numerical_cols = [col for col in all_cols_after_fe if col not in categorical_cols]

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Keeping median as in base solution
        ('scaler', StandardScaler())
    ])

    # Initialize transformers list for ColumnTransformer
    transformers = [
        ('num', numerical_transformer, numerical_cols)
    ]

    # Only add categorical transformer if there are categorical columns
    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), # Handles potential NaNs in categorical columns
            ('onehot', OneHotEncoder(handle_unknown='ignore')) # Handles new categories in test set
        ])
        transformers.append(('cat', categorical_transformer, categorical_cols))

    # Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough' # Ensure no columns are accidentally dropped
    )

    # Return feature-engineered dataframes and the configured preprocessor
    # The ColumnTransformer will be fit within the model pipelines on the raw data (which now includes FE)
    return X_train_fe, y_train_raw, X_test_fe, preprocessor

def train_and_predict(X_train_fe, y_train_raw, X_test_fe, preprocessor):
    """
    Trains multiple RandomForestRegressor models with the specified preprocessor,
    performs manual cross-validation for ensemble performance,
    and makes ensembled predictions on the test set.
    """
    # Define Model A (Base solution inspired)
    model_A_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, 
                                            max_features=0.8, max_depth=15, min_samples_leaf=5))
    ])

    # Define Model B (Reference solution inspired with different params for diversity)
    model_B_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=150, random_state=43, n_jobs=-1,
                                            max_features='sqrt', max_depth=20, min_samples_leaf=3))
    ])

    # --- Manual K-Fold Cross-Validation for Ensemble Performance ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ensemble_rmse_scores = []
    
    for train_index, val_index in kf.split(X_train_fe):
        X_train_fold, X_val_fold = X_train_fe.iloc[train_index], X_train_fe.iloc[val_index]
        y_train_fold, y_val_fold = y_train_raw.iloc[train_index], y_train_raw.iloc[val_index]

        # Fit models on the training fold (includes preprocessing steps)
        model_A_pipeline.fit(X_train_fold, y_train_fold)
        model_B_pipeline.fit(X_train_fold, y_train_fold)

        # Get predictions on the validation fold
        y_pred_A = model_A_pipeline.predict(X_val_fold)
        y_pred_B = model_B_pipeline.predict(X_val_fold)

        # Ensemble predictions (simple average)
        ensemble_pred_val = (y_pred_A + y_pred_B) / 2

        # Calculate RMSE for the ensemble
        fold_rmse = np.sqrt(mean_squared_error(y_val_fold, ensemble_pred_val))
        ensemble_rmse_scores.append(fold_rmse)

    final_validation_score = np.mean(ensemble_rmse_scores)
    print(f'Final Validation Performance: {final_validation_score}')

    # --- Train models on the full training data and make final predictions ---
    # Fit pipelines on the full feature-engineered training data
    model_A_pipeline.fit(X_train_fe, y_train_raw)
    model_B_pipeline.fit(X_train_fe, y_train_raw)

    # Make predictions on the feature-engineered test data
    predictions_A = model_A_pipeline.predict(X_test_fe)
    predictions_B = model_B_pipeline.predict(X_test_fe)

    # Ensemble final test predictions
    test_predictions = (predictions_A + predictions_B) / 2

    return test_predictions

if __name__ == "__main__":
    # Define paths to datasets
    train_file = "./input/train.csv"
    test_file = "./input/test.csv"

    # Load and preprocess data (get feature-engineered dataframes and the preprocessor object)
    X_train_fe, y_train_raw, X_test_fe, preprocessor = load_and_preprocess_data(train_file, test_file)

    # Train models and get ensembled predictions
    test_predictions = train_and_predict(X_train_fe, y_train_raw, X_test_fe, preprocessor)

    # Prepare submission file
    submission_df = pd.DataFrame({'median_house_value': test_predictions})

    # Save submission file in the specified format
    submission_df.to_csv('submission.csv', index=False, header=False)

    print("Submission file created successfully!")
