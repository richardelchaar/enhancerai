
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# --- Common Data Loading and Splitting ---
# Load datasets
train_df_full = pd.read_csv("./input/train.csv")
test_df_original = pd.read_csv("./input/test.csv") # Keep original for separate preprocessing branches

# Define overall features (X) and target (y) for the common split
X_full = train_df_full.drop("median_house_value", axis=1)
y_full = train_df_full["median_house_value"]

# Split training data into a training set and a hold-out validation set
# This common split ensures y_val_common is identical for meta-learning.
X_train_common, X_val_common, y_train_common, y_val_common = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

# --- Solution 1 Block ---
# Prepare data specifically for Solution 1's preprocessing and feature set
print("--- Preparing and training for Solution 1 ---")
X_train_sol1_prep = X_train_common.copy()
X_val_sol1_prep = X_val_common.copy()
X_test_sol1_prep = test_df_original.copy()

# Solution 1 Preprocessing: Handle missing values in 'total_bedrooms'
# Fill with median from the training set to prevent data leakage.
sol1_total_bedrooms_median = X_train_sol1_prep['total_bedrooms'].median()
X_train_sol1_prep['total_bedrooms'].fillna(sol1_total_bedrooms_median, inplace=True)
X_val_sol1_prep['total_bedrooms'].fillna(sol1_total_bedrooms_median, inplace=True)
X_test_sol1_prep['total_bedrooms'].fillna(sol1_total_bedrooms_median, inplace=True)

# Define features used by Solution 1
features_sol1 = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                 'total_bedrooms', 'population', 'households', 'median_income']

X_train_sol1 = X_train_sol1_prep[features_sol1]
X_val_sol1 = X_val_sol1_prep[features_sol1]
X_test_sol1 = X_test_sol1_prep[features_sol1]


# Initialize and Train Solution 1 Models
lgbm_model_sol1 = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42)
xgb_model_sol1 = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_jobs=-1)

print("Training LightGBM model for Solution 1...")
lgbm_model_sol1.fit(X_train_sol1, y_train_common)
print("LightGBM model training complete for Solution 1.")

print("Training XGBoost model for Solution 1...")
xgb_model_sol1.fit(X_train_sol1, y_train_common)
print("XGBoost model training complete for Solution 1.")

# Make predictions for Solution 1
y_pred_val_lgbm_sol1 = lgbm_model_sol1.predict(X_val_sol1)
y_pred_val_xgb_sol1 = xgb_model_sol1.predict(X_val_sol1)

# Solution 1's internal ensemble weights
weight_lgbm_sol1 = 0.7
weight_xgb_sol1 = 0.3

# Solution 1's ensembled validation predictions
val_preds_sol1 = (y_pred_val_lgbm_sol1 * weight_lgbm_sol1) + (y_pred_val_xgb_sol1 * weight_xgb_sol1)

# Solution 1's ensembled test predictions
test_predictions_lgbm_sol1 = lgbm_model_sol1.predict(X_test_sol1)
test_predictions_xgb_sol1 = xgb_model_sol1.predict(X_test_sol1)
test_preds_sol1 = (test_predictions_lgbm_sol1 * weight_lgbm_sol1) + (test_predictions_xgb_sol1 * weight_xgb_sol1)
print("--- Solution 1 predictions complete ---")


# --- Solution 2 Block ---
# Prepare data specifically for Solution 2's preprocessing and feature set
print("\n--- Preparing and training for Solution 2 ---")
X_train_sol2_prep = X_train_common.copy()
X_val_sol2_prep = X_val_common.copy()
X_test_sol2_prep = test_df_original.copy() # Use original test_df for imputation

# Identify numerical features for imputation (Solution 2 style)
numerical_features_sol2 = X_train_sol2_prep.select_dtypes(include=np.number).columns

# Solution 2 Preprocessing: Impute missing values with SimpleImputer (median strategy)
imputer_sol2 = SimpleImputer(strategy='median')

# Fit imputer on training data and transform all splits
X_train_sol2 = pd.DataFrame(imputer_sol2.fit_transform(X_train_sol2_prep[numerical_features_sol2]),
                            columns=numerical_features_sol2, index=X_train_sol2_prep.index)
X_val_sol2 = pd.DataFrame(imputer_sol2.transform(X_val_sol2_prep[numerical_features_sol2]),
                          columns=numerical_features_sol2, index=X_val_sol2_prep.index)
X_test_sol2 = pd.DataFrame(imputer_sol2.transform(X_test_sol2_prep[numerical_features_sol2]),
                           columns=numerical_features_sol2, index=X_test_sol2_prep.index)

# Initialize and Train Solution 2 Models
model_lightgbm_sol2 = lgb.LGBMRegressor(objective='regression', n_estimators=500,
                                        learning_rate=0.05, random_state=42, n_jobs=-1)
model_xgboost_sol2 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500,
                                      learning_rate=0.05, random_state=42, n_jobs=-1)

print("Training LightGBM model for Solution 2...")
model_lightgbm_sol2.fit(X_train_sol2, y_train_common)
print("LightGBM model training complete for Solution 2.")

print("Training XGBoost model for Solution 2...")
model_xgboost_sol2.fit(X_train_sol2, y_train_common)
print("XGBoost model training complete for Solution 2.")

# Make predictions for Solution 2
y_val_pred_lightgbm_sol2 = model_lightgbm_sol2.predict(X_val_sol2)
y_val_pred_xgboost_sol2 = model_xgboost_sol2.predict(X_val_sol2)

# Solution 2's internal ensemble (simple averaging)
val_preds_sol2 = (y_val_pred_lightgbm_sol2 + y_val_pred_xgboost_sol2) / 2

# Solution 2's ensembled test predictions
test_predictions_lightgbm_sol2 = model_lightgbm_sol2.predict(X_test_sol2)
test_predictions_xgboost_sol2 = model_xgboost_sol2.predict(X_test_sol2)
test_preds_sol2 = (test_predictions_lightgbm_sol2 + test_predictions_xgboost_sol2) / 2
print("--- Solution 2 predictions complete ---")


# --- Meta-Learning Ensemble ---
print("\n--- Training Meta-Learner ---")
# Prepare data for meta-learner training (using validation predictions from base models)
meta_X_train = np.column_stack((val_preds_sol1, val_preds_sol2))
meta_y_train = y_val_common # True target values for the validation set

# Initialize and train the Linear Regression meta-model
# fit_intercept=False and positive=True to learn non-negative weights as proportions
meta_model = LinearRegression(fit_intercept=False, positive=True)
meta_model.fit(meta_X_train, meta_y_train)

# Predict on the validation set using the meta-model
meta_val_predictions = meta_model.predict(meta_X_train)

# Calculate RMSE for the meta-ensemble on the validation set
rmse_meta_val = np.sqrt(mean_squared_error(meta_y_train, meta_val_predictions))
print(f"Final Validation Performance: {rmse_meta_val}")

# --- Final Test Predictions and Submission ---
print("\n--- Generating final test predictions and submission file ---")
# Prepare test data for meta-learner prediction
meta_X_test = np.column_stack((test_preds_sol1, test_preds_sol2))

# Generate final predictions on the test set using the trained meta-model
final_test_predictions = meta_model.predict(meta_X_test)

# Create the submission file
submission_df = pd.DataFrame({'median_house_value': final_test_predictions})
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")
