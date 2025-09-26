
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer # Used by Solution 2

# Load datasets once globally
train_df_global = pd.read_csv("./input/train.csv")
test_df_global = pd.read_csv("./input/test.csv")

# --- Solution 1 Preprocessing and Training ---
# Create copies to allow independent preprocessing as per original solutions
train_df_sol1 = train_df_global.copy()
test_df_sol1 = test_df_global.copy()

# Preprocessing for Solution 1: Handle missing values
# The 'total_bedrooms' column often contains a few missing values in this dataset.
# Filling with the median is a robust strategy for tree-based models and a simple approach.
# Median for training data calculated from its own train_df copy
train_df_sol1['total_bedrooms'].fillna(train_df_sol1['total_bedrooms'].median(), inplace=True)
# Median for test data calculated from its own test_df copy (matches original solution's behavior)
test_df_sol1['total_bedrooms'].fillna(test_df_sol1['total_bedrooms'].median(), inplace=True)

# Define features (X) and target (y) for Solution 1
features_sol1 = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income']
target_sol1 = 'median_house_value'

X_sol1 = train_df_sol1[features_sol1]
y_sol1 = train_df_sol1[target_sol1]

# Split training data for Solution 1 into a training set and a hold-out validation set
# A 80/20 split is commonly used. random_state ensures reproducibility.
# This y_val_sol1 will be used as the common ground truth for the final ensemble RMSE.
X_train_sol1, X_val_sol1, y_train_sol1, y_val_sol1 = train_test_split(X_sol1, y_sol1, test_size=0.2, random_state=42)

# --- Model Initialization for Solution 1 ---
lgbm_model_sol1 = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42)
xgb_model_sol1 = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_jobs=-1)

# --- Model Training for Solution 1 ---
lgbm_model_sol1.fit(X_train_sol1, y_train_sol1)
xgb_model_sol1.fit(X_train_sol1, y_train_sol1)

# --- Validation Predictions and Ensemble for Solution 1 ---
y_pred_val_lgbm_sol1 = lgbm_model_sol1.predict(X_val_sol1)
y_pred_val_xgb_sol1 = xgb_model_sol1.predict(X_val_sol1)

# Define weights based on ablation study findings (LightGBM is stronger in Sol 1)
weight_lgbm_sol1 = 0.7
weight_xgb_sol1 = 0.3
y_pred_val_ensemble_sol1 = (y_pred_val_lgbm_sol1 * weight_lgbm_sol1) + (y_pred_val_xgb_sol1 * weight_xgb_sol1)

# --- Test Predictions for Solution 1 ---
X_test_sol1 = test_df_sol1[features_sol1]
test_predictions_lgbm_sol1 = lgbm_model_sol1.predict(X_test_sol1)
test_predictions_xgb_sol1 = xgb_model_sol1.predict(X_test_sol1)
test_predictions_ensemble_sol1 = (test_predictions_lgbm_sol1 * weight_lgbm_sol1) + (test_predictions_xgb_sol1 * weight_xgb_sol1)


# --- Solution 2 Preprocessing and Training ---
# Create copies for Solution 2 to maintain independent processing logic
train_df_sol2 = train_df_global.copy()
test_df_sol2 = test_df_global.copy()

# Separate target variable from features for Solution 2
X_sol2_full = train_df_sol2.drop("median_house_value", axis=1)
y_sol2_full = train_df_sol2["median_house_value"]

# Split the training data into training and validation sets for Solution 2
# Using a fixed random_state for reproducibility and to ensure validation sets align with Sol1.
# The '_' is used for the y_val_sol2 as we will use y_val_sol1 for the final RMSE calculation.
X_train_sol2_raw, X_val_sol2_raw, y_train_sol2, _ = train_test_split(X_sol2_full, y_sol2_full, test_size=0.2, random_state=42)

# Identify numerical features for imputation (select_dtypes will exclude 'ocean_proximity')
numerical_features_sol2 = X_train_sol2_raw.select_dtypes(include=np.number).columns

# Impute missing values for Solution 2
# SimpleImputer with median strategy is robust to outliers and matches original Solution 2.
imputer_sol2 = SimpleImputer(strategy='median')

# Fit imputer on training data and transform all splits for Solution 2
X_train_imputed_sol2 = pd.DataFrame(imputer_sol2.fit_transform(X_train_sol2_raw[numerical_features_sol2]),
                               columns=numerical_features_sol2, index=X_train_sol2_raw.index)
X_val_imputed_sol2 = pd.DataFrame(imputer_sol2.transform(X_val_sol2_raw[numerical_features_sol2]),
                             columns=numerical_features_sol2, index=X_val_sol2_raw.index)
test_df_imputed_sol2 = pd.DataFrame(imputer_sol2.transform(test_df_sol2[numerical_features_sol2]),
                               columns=numerical_features_sol2, index=test_df_sol2.index)

# --- Model Training for Solution 2 ---
model_lightgbm_sol2 = lgb.LGBMRegressor(objective='regression', n_estimators=500,
                                   learning_rate=0.05, random_state=42, n_jobs=-1)
model_lightgbm_sol2.fit(X_train_imputed_sol2, y_train_sol2)

model_xgboost_sol2 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500,
                             learning_rate=0.05, random_state=42, n_jobs=-1)
model_xgboost_sol2.fit(X_train_imputed_sol2, y_train_sol2)

# --- Prediction and Ensemble for Solution 2 ---
y_val_pred_lightgbm_sol2 = model_lightgbm_sol2.predict(X_val_imputed_sol2)
y_val_pred_xgboost_sol2 = model_xgboost_sol2.predict(X_val_imputed_sol2)

# Ensemble the predictions by simple averaging for Solution 2
y_val_pred_ensembled_sol2 = (y_val_pred_lightgbm_sol2 + y_val_pred_xgboost_sol2) / 2

# --- Test Predictions for Solution 2 ---
test_predictions_lightgbm_sol2 = model_lightgbm_sol2.predict(test_df_imputed_sol2)
test_predictions_xgboost_sol2 = model_xgboost_sol2.predict(test_df_imputed_sol2)
final_test_predictions_sol2 = (test_predictions_lightgbm_sol2 + test_predictions_xgboost_sol2) / 2


# --- Final Ensemble of Solution 1 and Solution 2 ---

# Combine validation predictions from both solutions by simple arithmetic average
final_validation_predictions = (y_pred_val_ensemble_sol1 + y_val_pred_ensembled_sol2) / 2

# Calculate the final Root Mean Squared Error (RMSE) on the ensembled validation predictions
# using y_val_sol1 as the common ground truth.
rmse_final_ensemble = np.sqrt(mean_squared_error(y_val_sol1, final_validation_predictions))

# Print the final validation performance as required
print(f"Final Validation Performance: {rmse_final_ensemble}")

# Combine test predictions from both solutions by simple arithmetic average for the final submission
final_submission_predictions = (test_predictions_ensemble_sol1 + final_test_predictions_sol2) / 2

# Create the submission file in the specified format
submission_df = pd.DataFrame({'median_house_value': final_submission_predictions})
submission_df.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully.")
