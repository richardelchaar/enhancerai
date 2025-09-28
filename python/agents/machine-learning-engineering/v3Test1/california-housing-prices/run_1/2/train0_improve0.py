
# Suppress verbose model output to prevent token explosion
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
# Suppress LightGBM verbosity
os.environ['LIGHTGBM_VERBOSITY'] = '-1'
# Suppress XGBoost verbosity  
os.environ['XGBOOST_VERBOSITY'] = '0'
# Suppress sklearn warnings
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate target variable
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']
X_test = test_df.copy()

# Preprocessing function
def preprocess_data(df):
    # Impute missing values for 'total_bedrooms' first, as it's a common source of NaNs in this dataset
    # and would affect ratio calculations.
    imputer_bedrooms = SimpleImputer(strategy='median')
    if 'total_bedrooms' in df.columns:
        df['total_bedrooms'] = imputer_bedrooms.fit_transform(df[['total_bedrooms']])

    # Feature Engineering: ratio features
    # Add a small epsilon to denominators to prevent ZeroDivisionError.
    # This addresses the float division by zero error directly.
    epsilon = 1e-6

    df['rooms_per_household'] = df['total_rooms'] / (df['households'] + epsilon)
    df['bedrooms_per_room'] = df['total_bedrooms'] / (df['total_rooms'] + epsilon)
    df['population_per_household'] = df['population'] / (df['households'] + epsilon)
    df['median_income_per_room'] = df['median_income'] / (df['total_rooms'] / (df['population'] + epsilon) + epsilon) # Added another ratio feature
    
    # Identify all numerical columns for imputation (including newly created ratios)
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Impute any remaining missing values in numerical columns, primarily for original NaNs
    # in columns other than 'total_bedrooms' if any, and for any NaNs resulting from ratios if epsilon wasn't enough
    # or if any source column itself was NaN.
    imputer_numerical = SimpleImputer(strategy='median')
    df[numerical_cols] = imputer_numerical.fit_transform(df[numerical_cols])

    return df

# Apply preprocessing
X = preprocess_data(X)
X_test = preprocess_data(X_test)

# Align columns - crucial if different features were engineered or missing in train/test sets
train_cols = X.columns
test_cols = X_test.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X[c] = 0

X_test = X_test[train_cols] # Ensure order is the same

# Scaling numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for easier handling with column names, although numpy arrays are fine for XGBoost
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Model Training
# Using KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Added 'tree_method' and 'gpu_id' for potential GPU acceleration if available,
# removed 'silent' and 'verbose' if not explicitly supported in newer XGBoost, using verbosity=0 instead.
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000,
                           learning_rate=0.05, random_state=42, n_jobs=-1,
                           subsample=0.7, colsample_bytree=0.7, verbosity=0) # verbosity=0 suppresses console output

oof_predictions = np.zeros(X.shape[0])
test_predictions = np.zeros(X_test.shape[0])

for fold, (train_index, val_index) in enumerate(kf.split(X_scaled_df, y)):
    X_train, X_val = X_scaled_df.iloc[train_index], X_scaled_df.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    xgb_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False) # verbose=False suppresses per-round evaluation output
    oof_predictions[val_index] = xgb_reg.predict(X_val)
    test_predictions += xgb_reg.predict(X_test_scaled_df) / kf.n_splits

# Calculate overall RMSE
final_validation_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
print(f"Final Validation Performance: {final_validation_rmse}")

# Prepare submission file
submission_df = pd.DataFrame({'median_house_value': test_predictions})
submission_df.to_csv("submission.csv", index=False)
