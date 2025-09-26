
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Dummy Data Generation (if input files are not present) ---
def generate_dummy_data(train_path="./input/train.csv", test_path="./input/test.csv"):
    if not os.path.exists("./input"):
        os.makedirs("./input")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Generating dummy train/test data for demonstration...")
        from sklearn.datasets import make_regression
        
        n_samples_train = 1000
        n_samples_test = 300
        n_features = 8 # Corresponds to the input features in the problem description

        # Train data
        X_train_dummy, y_train_dummy = make_regression(n_samples=n_samples_train, n_features=n_features, n_informative=n_features, random_state=42)
        train_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
        train_df_dummy = pd.DataFrame(X_train_dummy, columns=train_columns)
        # Scale target to be positive and somewhat realistic for median_house_value
        train_df_dummy['median_house_value'] = np.abs(y_train_dummy * 1000) + 50000 + (np.random.rand(n_samples_train) * 500000) # Ensure positive and some range
        
        # Test data
        X_test_dummy, _ = make_regression(n_samples=n_samples_test, n_features=n_features, n_informative=n_features, random_state=43)
        test_df_dummy = pd.DataFrame(X_test_dummy, columns=train_columns)
        
        # Add some realistic value ranges to dummy data for better representation
        for df in [train_df_dummy, test_df_dummy]:
            df['longitude'] = np.random.uniform(-124.3, -114.2, len(df))
            df['latitude'] = np.random.uniform(32.5, 41.9, len(df))
            df['housing_median_age'] = np.random.randint(1, 52, len(df))
            df['total_rooms'] = np.random.randint(2, 40000, len(df))
            df['total_bedrooms'] = df['total_rooms'] * np.random.uniform(0.05, 0.2, len(df))
            df['population'] = np.random.randint(3, 35000, len(df))
            df['households'] = df['population'] * np.random.uniform(0.2, 0.4, len(df))
            df['median_income'] = np.random.uniform(0.5, 15, len(df))
            # Introduce some NaNs in total_bedrooms, as it's common in this dataset
            nan_indices = np.random.choice(df.index, int(len(df) * 0.01), replace=False) # 1% NaNs
            df.loc[nan_indices, 'total_bedrooms'] = np.nan

        train_df_dummy.to_csv(train_path, index=False)
        test_df_dummy.to_csv(test_path, index=False)
        print(f"Dummy data generated at {train_path} and {test_path}")
    else:
        print("Input data already exists, skipping dummy data generation.")

# Execute dummy data generation to ensure files are present
generate_dummy_data()

# --- Helper function for OOF and Test Predictions (Regression) ---
def get_oof_and_test_preds(model_constructor, X_train, y_train, X_test, folds):
    oof_preds = np.zeros(len(X_train))
    test_preds_list = []

    for fold, (train_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = model_constructor() # Instantiate model using the constructor function
        model.fit(X_train_fold, y_train_fold)
        
        oof_preds[val_idx] = model.predict(X_val_fold)
        test_preds_list.append(model.predict(X_test))
        
    final_test_preds = np.mean(test_preds_list, axis=0)
    return oof_preds, final_test_preds

# Define model constructor functions for regression
def linear_reg_model():
    return LinearRegression()

def rf_reg_model():
    return RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, min_samples_leaf=5, n_jobs=-1)

def gb_reg_model():
    return GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# --- Main Ensemble Script ---

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Identify features and target column
TARGET = 'median_house_value'
FEATURES = [col for col in train_df.columns if col != TARGET]
X = train_df[FEATURES]
y = train_df[TARGET]
test_X = test_df[FEATURES]

# Impute missing values with median (specifically 'total_bedrooms' is often missing in this type of dataset)
for col in FEATURES:
    if X[col].isnull().any():
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        test_X[col].fillna(median_val, inplace=True) # Impute test data as well

# Use KFold for cross-validation for regression tasks
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

print("--- Running individual solutions for OOF and test predictions across folds ---")

# Solution 1: Linear Regression
oof_preds_s1, final_test_preds_s1 = get_oof_and_test_preds(linear_reg_model, X, y, test_X, kf)
print("Solution 1 (Linear Regression) completed.")

# Solution 2: Random Forest Regressor
oof_preds_s2, final_test_preds_s2 = get_oof_and_test_preds(rf_reg_model, X, y, test_X, kf)
print("Solution 2 (Random Forest Regressor) completed.")

# Solution 3: Gradient Boosting Regressor
oof_preds_s3, final_test_preds_s3 = get_oof_and_test_preds(gb_reg_model, X, y, test_X, kf)
print("Solution 3 (Gradient Boosting Regressor) completed.")

# Create individual submission files
pd.DataFrame({'median_house_value': final_test_preds_s1}).to_csv('submission_solution1.csv', index=False)
pd.DataFrame({'median_house_value': final_test_preds_s2}).to_csv('submission_solution2.csv', index=False)
pd.DataFrame({'median_house_value': final_test_preds_s3}).to_csv('submission_solution3.csv', index=False)
print("\nIndividual solution submission files created: submission_solution1.csv, submission_solution2.csv, submission_solution3.csv")


print("\n--- Ensembling predictions ---")

# Stack OOF predictions for ensemble validation
stacked_oof_preds = np.column_stack([oof_preds_s1, oof_preds_s2, oof_preds_s3])

# Apply Weighted Averaging for OOF (Validation) predictions
# Initial weights are set equally as suggested by the ensemble plan
weights = [1/3, 1/3, 1/3]
ensemble_oof_preds = (weights[0] * stacked_oof_preds[:, 0] +
                      weights[1] * stacked_oof_preds[:, 1] +
                      weights[2] * stacked_oof_preds[:, 2])

# Calculate the final validation performance using the ensembled OOF predictions
final_validation_score = np.sqrt(mean_squared_error(y, ensemble_oof_preds))
print(f"Final Validation Performance: {final_validation_score}")

# --- Ensemble Test Predictions ---
ensemble_test_preds = (weights[0] * final_test_preds_s1 +
                       weights[1] * final_test_preds_s2 +
                       weights[2] * final_test_preds_s3)

# Generate Final Submission file
final_submission_df = pd.DataFrame({'median_house_value': ensemble_test_preds})
final_submission_df.to_csv('submission.csv', index=False)
print("Ensembled submission file 'submission.csv' created.")
