
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from xgboost import XGBRegressor # Import XGBoost
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load the datasets
# Assuming the files are in the './input/' directory as per common Kaggle setup
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate target variable from features
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Split the training data into training and validation sets
# Using a fixed random_state for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical features for imputation (all features except target are numerical here)

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

numerical_features = X_train.select_dtypes(include=np.number).columns

# Identify numerical features that have missing values in the training set
features_with_na = X_train[numerical_features].columns[X_train[numerical_features].isnull().any()].tolist()

# Create missing value indicator columns for each split
# This must be done BEFORE imputation to capture original NaNs
X_train_indicators = pd.DataFrame(index=X_train.index)
X_val_indicators = pd.DataFrame(index=X_val.index)
test_df_indicators = pd.DataFrame(index=test_df.index)

for col in features_with_na:
    X_train_indicators[f'{col}_is_missing'] = X_train[col].isnull()
    X_val_indicators[f'{col}_is_missing'] = X_val[col].isnull()
    test_df_indicators[f'{col}_is_missing'] = test_df[col].isnull()

# Impute missing values for numerical features
# Using SimpleImputer with a median strategy is robust to outliers.
imputer = SimpleImputer(strategy='median')

# Fit imputer on training data and transform all splits for numerical features
# It's crucial to fit only on training data to prevent data leakage.
X_train_imputed_numerical = pd.DataFrame(imputer.fit_transform(X_train[numerical_features]),
                                         columns=numerical_features, index=X_train.index)
X_val_imputed_numerical = pd.DataFrame(imputer.transform(X_val[numerical_features]),
                                       columns=numerical_features, index=X_val.index)
test_df_imputed_numerical = pd.DataFrame(imputer.transform(test_df[numerical_features]),
                                         columns=numerical_features, index=test_df.index)

# Combine imputed numerical features with the new missing value indicator columns
X_train_imputed = pd.concat([X_train_imputed_numerical, X_train_indicators], axis=1)
X_val_imputed = pd.concat([X_val_imputed_numerical, X_val_indicators], axis=1)
test_df_imputed = pd.concat([test_df_imputed_numerical, test_df_indicators], axis=1)


# --- Model Training ---

# 1. Initialize and train the LightGBM Regressor model (from base solution)
model_lightgbm = lgb.LGBMRegressor(objective='regression', n_estimators=500, 
                                   learning_rate=0.05, random_state=42, n_jobs=-1)
model_lightgbm.fit(X_train_imputed, y_train)

# 2. Initialize and train the XGBoost Regressor model (from reference solution)
model_xgboost = XGBRegressor(objective='reg:squarederror', n_estimators=500, 
                             learning_rate=0.05, random_state=42, n_jobs=-1)
model_xgboost.fit(X_train_imputed, y_train)

# --- Prediction and Ensemble ---

# Make predictions on the validation set for both models
y_val_pred_lightgbm = model_lightgbm.predict(X_val_imputed)
y_val_pred_xgboost = model_xgboost.predict(X_val_imputed)

# Ensemble the predictions by simple averaging
y_val_pred_ensembled = (y_val_pred_lightgbm + y_val_pred_xgboost) / 2

# Calculate Root Mean Squared Error (RMSE) on the ensembled validation predictions
rmse_val_ensembled = np.sqrt(mean_squared_error(y_val, y_val_pred_ensembled))

# Print the final ensembled validation performance
print(f"Final Validation Performance: {rmse_val_ensembled}")

# Generate predictions for the test dataset for submission (optional, but good practice for Kaggle)
test_predictions_lightgbm = model_lightgbm.predict(test_df_imputed)
test_predictions_xgboost = model_xgboost.predict(test_df_imputed)

# Ensemble test predictions
final_test_predictions = (test_predictions_lightgbm + test_predictions_xgboost) / 2

# Create submission file (uncomment to save if needed)
# submission_df = pd.DataFrame({'median_house_value': final_test_predictions})
# submission_df.to_csv('submission.csv', index=False)
