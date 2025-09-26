
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor

# --- Configuration (from both solutions) ---
TRAIN_FILE = 'input/train.csv'
TEST_FILE = 'input/test.csv'
TARGET_COLUMN = 'median_house_value'
RANDOM_SEED = 42
VALIDATION_SPLIT_RATIO = 0.2

# --- 1. Unified Data Preprocessing ---
# Load the datasets
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# Separate target variable from features
X = train_df.drop(TARGET_COLUMN, axis=1)
y = train_df[TARGET_COLUMN]

# Handle missing values in 'total_bedrooms' using SimpleImputer (from Solution 2)
# Apply to both training features and test features
imputer = SimpleImputer(strategy='median')

# Fit imputer on X and transform X and test_df
X_imputed = imputer.fit_transform(X)
test_imputed = imputer.transform(test_df)

# Convert back to DataFrame, preserving column names
X = pd.DataFrame(X_imputed, columns=X.columns)
test_df = pd.DataFrame(test_imputed, columns=test_df.columns)

# --- 2. Consistent Data Splitting ---
# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VALIDATION_SPLIT_RATIO, random_state=RANDOM_SEED
)

# --- 3. Individual Base Model Training (on X_train, y_train) ---

# LightGBM Model (Solution 1 default parameters)
model_lgbm_s1 = lgb.LGBMRegressor(random_state=RANDOM_SEED)
model_lgbm_s1.fit(X_train, y_train)

# XGBoost Model (Solution 1 default parameters)
model_xgb_s1 = xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_SEED)
model_xgb_s1.fit(X_train, y_train)

# CatBoost Regressor (Solution 2 specific parameters)
model_cb_s2 = CatBoostRegressor(
    loss_function='RMSE',
    random_seed=RANDOM_SEED,
    verbose=False,
    iterations=1000,
    learning_rate=0.05,
    early_stopping_rounds=50
)
model_cb_s2.fit(X_train, y_train, eval_set=(X_val, y_val))

# LightGBM Model (Solution 2 specific parameters)
lgbm_s2_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': RANDOM_SEED,
    'boosting_type': 'gbdt',
}
model_lgbm_s2 = lgb.LGBMRegressor(**lgbm_s2_params)
model_lgbm_s2.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])

# --- 4. Generate Validation Predictions ---
y_pred_val_lgbm_s1 = model_lgbm_s1.predict(X_val)
y_pred_val_xgb_s1 = model_xgb_s1.predict(X_val)
y_pred_val_cb_s2 = model_cb_s2.predict(X_val)
y_pred_val_lgbm_s2 = model_lgbm_s2.predict(X_val)

# --- 5. Meta-Learner Setup (Validation) ---
# Stack the out-of-fold predictions from base models
X_meta_val = np.column_stack((y_pred_val_lgbm_s1,
                              y_pred_val_xgb_s1,
                              y_pred_val_cb_s2,
                              y_pred_val_lgbm_s2))

# --- 6. Meta-Learner Training ---
# Initialize and train the Ridge meta-learner (from Solution 1)
meta_learner = Ridge(random_state=RANDOM_SEED)
meta_learner.fit(X_meta_val, y_val)

# --- 7. Validation Performance Evaluation ---
# Generate the ensemble prediction using the trained meta-learner
y_pred_val_ensemble = meta_learner.predict(X_meta_val)
rmse_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble))

# Print the final validation performance
print(f"Final Validation Performance: {rmse_ensemble}")

# --- 8. Final Prediction Generation ---

# Retrain all four base models on the *entire* preprocessed training dataset (X, y)

# LightGBM (Solution 1)
full_model_lgbm_s1 = lgb.LGBMRegressor(random_state=RANDOM_SEED)
full_model_lgbm_s1.fit(X, y)

# XGBoost (Solution 1)
full_model_xgb_s1 = xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_SEED)
full_model_xgb_s1.fit(X, y)

# CatBoost (Solution 2) - Train on full data without eval_set for final model
full_model_cb_s2 = CatBoostRegressor(
    loss_function='RMSE',
    random_seed=RANDOM_SEED,
    verbose=False,
    iterations=1000, # Use max iterations from validation phase
    learning_rate=0.05
    # early_stopping_rounds is not used here as there's no separate validation set
)
full_model_cb_s2.fit(X, y)

# LightGBM (Solution 2) - Train on full data without eval_set for final model
full_model_lgbm_s2 = lgb.LGBMRegressor(**lgbm_s2_params)
full_model_lgbm_s2.fit(X, y) # n_estimators will be used directly

# Generate test predictions from each fully trained base model
test_predictions_lgbm_s1 = full_model_lgbm_s1.predict(test_df)
test_predictions_xgb_s1 = full_model_xgb_s1.predict(test_df)
test_predictions_cb_s2 = full_model_cb_s2.predict(test_df)
test_predictions_lgbm_s2 = full_model_lgbm_s2.predict(test_df)

# Stack these four sets of test predictions horizontally to form the final test meta-feature matrix
X_meta_test = np.column_stack((test_predictions_lgbm_s1,
                               test_predictions_xgb_s1,
                               test_predictions_cb_s2,
                               test_predictions_lgbm_s2))

# Use the *trained* Ridge meta-learner to make the final ensembled predictions on this test meta-feature matrix
final_ensemble_test_predictions = meta_learner.predict(X_meta_test)

# --- Submission (as per Solution 2's print format) ---
print(TARGET_COLUMN)
for pred_value in final_ensemble_test_predictions:
    print(f"{pred_value}")
