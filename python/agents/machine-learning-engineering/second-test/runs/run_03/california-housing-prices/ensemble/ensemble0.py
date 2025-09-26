
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Changed to Regressor
from sklearn.metrics import mean_squared_error  # Changed to MSE for regression
import os

# --- Setup for Reproducibility and Data Generation (for demonstration) ---
# In a real Kaggle scenario, these files would already exist in './input'.
# These blocks ensure the code is runnable out-of-the-box for testing.
if not os.path.exists('./input'):
    os.makedirs('./input')

# Define target column name
TARGET_COL = 'median_house_value'
# Define a temporary ID column name for internal merging, as the original dataset has no explicit ID.
ID_COL = 'id_for_merge' 

# Create a dummy train.csv if it doesn't exist, matching the problem description's structure
train_file_path = './input/train.csv'
if not os.path.exists(train_file_path):
    np.random.seed(42)
    num_samples = 1000
    # Columns as per the problem description
    cols_train = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', TARGET_COL]
    dummy_data_train = pd.DataFrame(np.random.rand(num_samples, len(cols_train)), columns=cols_train)
    
    # Generate somewhat more realistic values for dummy data
    dummy_data_train['longitude'] = np.random.uniform(-125, -114, num_samples)
    dummy_data_train['latitude'] = np.random.uniform(32, 42, num_samples)
    dummy_data_train['housing_median_age'] = np.random.randint(1, 52, num_samples)
    dummy_data_train['total_rooms'] = np.random.randint(100, 10000, num_samples)
    dummy_data_train['total_bedrooms'] = np.random.randint(50, 2000, num_samples)
    dummy_data_train['population'] = np.random.randint(100, 5000, num_samples)
    dummy_data_train['households'] = np.random.randint(50, 1500, num_samples)
    dummy_data_train['median_income'] = np.random.uniform(0.5, 15, num_samples)
    dummy_data_train[TARGET_COL] = np.random.randint(15000, 500000, num_samples)
    
    dummy_data_train.to_csv(train_file_path, index=False)

# Create a dummy test.csv if it doesn't exist, matching the problem description's structure
test_file_path = './input/test.csv'
if not os.path.exists(test_file_path):
    np.random.seed(43)
    num_test_samples = 300
    # Columns as per the problem description (without the target)
    cols_test = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
    dummy_data_test = pd.DataFrame(np.random.rand(num_test_samples, len(cols_test)), columns=cols_test)

    # Generate somewhat more realistic values for dummy data
    dummy_data_test['longitude'] = np.random.uniform(-125, -114, num_test_samples)
    dummy_data_test['latitude'] = np.random.uniform(32, 42, num_test_samples)
    dummy_data_test['housing_median_age'] = np.random.randint(1, 52, num_test_samples)
    dummy_data_test['total_rooms'] = np.random.randint(100, 10000, num_test_samples)
    dummy_data_test['total_bedrooms'] = np.random.randint(50, 2000, num_test_samples)
    dummy_data_test['population'] = np.random.randint(100, 5000, num_test_samples)
    dummy_data_test['households'] = np.random.randint(50, 1500, num_test_samples)
    dummy_data_test['median_income'] = np.random.uniform(0.5, 15, num_test_samples)
    
    dummy_data_test.to_csv(test_file_path, index=False)


# --- Global Data Loading and Validation Split ---
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# FEATURES definition based on the actual dataset structure
FEATURES = [col for col in train_df.columns if col not in [TARGET_COL]]

X = train_df[FEATURES]
y = train_df[TARGET_COL]
X_test = test_df[FEATURES]

# Split training data into training and validation sets for local evaluation
# Removed 'stratify=y' as it's for classification, not regression.
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create an 'id' column for test_df and validation split for internal merging
# These 'ids' will be the original DataFrame indices, ensuring alignment for ensemble
test_df_ids = pd.DataFrame(test_df.index, columns=[ID_COL])
val_ids_df = pd.DataFrame(X_val_split.index, columns=[ID_COL])


# --- Python Solution 1 ---
print("Running Solution 1...")
model1 = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5, n_jobs=-1) # Changed to Regressor
model1.fit(X_train_split, y_train_split)

# Generate test predictions and save
predictions1_test = model1.predict(X_test) # Changed from predict_proba
solution1_submission = pd.DataFrame({ID_COL: test_df_ids[ID_COL], 'Prediction': predictions1_test})
solution1_submission.to_csv('solution1_predictions.csv', index=False)

# Generate validation predictions and save
predictions1_val = model1.predict(X_val_split) # Changed from predict_proba
solution1_val_df = pd.DataFrame({ID_COL: val_ids_df[ID_COL], 'Prediction': predictions1_val})
solution1_val_df.to_csv('solution1_val_predictions.csv', index=False)
print(f"Solution 1 validation RMSE: {np.sqrt(mean_squared_error(y_val_split, predictions1_val)):.4f}")


# --- Python Solution 2 ---
print("Running Solution 2...")
model2 = RandomForestRegressor(n_estimators=70, random_state=43, max_depth=6, n_jobs=-1) # Changed to Regressor
model2.fit(X_train_split, y_train_split)

predictions2_test = model2.predict(X_test) # Changed from predict_proba
solution2_submission = pd.DataFrame({ID_COL: test_df_ids[ID_COL], 'Prediction': predictions2_test})
solution2_submission.to_csv('solution2_predictions.csv', index=False)

predictions2_val = model2.predict(X_val_split) # Changed from predict_proba
solution2_val_df = pd.DataFrame({ID_COL: val_ids_df[ID_COL], 'Prediction': predictions2_val})
solution2_val_df.to_csv('solution2_val_predictions.csv', index=False)
print(f"Solution 2 validation RMSE: {np.sqrt(mean_squared_error(y_val_split, predictions2_val)):.4f}")


# --- Python Solution 3 ---
print("Running Solution 3...")
model3 = RandomForestRegressor(n_estimators=60, random_state=44, max_depth=5, n_jobs=-1) # Changed to Regressor
model3.fit(X_train_split, y_train_split)

predictions3_test = model3.predict(X_test) # Changed from predict_proba
solution3_submission = pd.DataFrame({ID_COL: test_df_ids[ID_COL], 'Prediction': predictions3_test})
solution3_submission.to_csv('solution3_predictions.csv', index=False)

predictions3_val = model3.predict(X_val_split) # Changed from predict_proba
solution3_val_df = pd.DataFrame({ID_COL: val_ids_df[ID_COL], 'Prediction': predictions3_val})
solution3_val_df.to_csv('solution3_val_predictions.csv', index=False)
print(f"Solution 3 validation RMSE: {np.sqrt(mean_squared_error(y_val_split, predictions3_val)):.4f}")


# --- Ensemble Plan Implementation ---

print("\n--- Implementing Ensemble Plan ---")

# 2. Load Predictions
print("Loading predictions from individual solutions...")
# Load test set predictions
df1_test = pd.read_csv('solution1_predictions.csv')
df2_test = pd.read_csv('solution2_predictions.csv')
df3_test = pd.read_csv('solution3_predictions.csv')

# Load validation set predictions
df1_val = pd.read_csv('solution1_val_predictions.csv')
df2_val = pd.read_csv('solution2_val_predictions.csv')
df3_val = pd.read_csv('solution3_val_predictions.csv')

# Merge test predictions based on ID_COL
ensemble_df_test = df1_test.merge(df2_test, on=ID_COL, suffixes=('_s1', '_s2'))
ensemble_df_test = ensemble_df_test.merge(df3_test, on=ID_COL)
ensemble_df_test = ensemble_df_test.rename(columns={'Prediction': 'Prediction_s3'})

# Merge validation predictions based on ID_COL
ensemble_df_val = df1_val.merge(df2_val, on=ID_COL, suffixes=('_s1', '_s2'))
ensemble_df_val = ensemble_df_val.merge(df3_val, on=ID_COL)
ensemble_df_val = ensemble_df_val.rename(columns={'Prediction': 'Prediction_s3'})


# 3. Perform Simple Averaging

# Calculate average for test predictions
ensemble_df_test['Prediction'] = (
    ensemble_df_test['Prediction_s1'] +
    ensemble_df_test['Prediction_s2'] +
    ensemble_df_test['Prediction_s3']
) / 3

# Calculate average for validation predictions
ensemble_df_val['Prediction'] = (
    ensemble_df_val['Prediction_s1'] +
    ensemble_df_val['Prediction_s2'] +
    ensemble_df_val['Prediction_s3']
) / 3

# Calculate ensemble validation metric (RMSE)
# Ensure y_val_split and ensemble_df_val predictions are aligned by ID for accurate metric calculation
# Create a DataFrame from y_val_split and val_ids_df, then merge with ensemble_df_val
y_val_aligned_df = pd.DataFrame({ID_COL: val_ids_df[ID_COL], TARGET_COL: y_val_split.values})

# Merge ensemble_df_val with y_val_aligned_df to get the true target values
ensemble_df_val_with_targets = ensemble_df_val.merge(y_val_aligned_df, on=ID_COL)
final_ensemble_val_score = np.sqrt(mean_squared_error(ensemble_df_val_with_targets[TARGET_COL], ensemble_df_val_with_targets['Prediction']))


# 4. Generate Final Submission
# The final submission should NOT include the ID_COL and should NOT have a header, as per the submission format.
final_submission = ensemble_df_test[['Prediction']]
final_submission.to_csv('final_ensemble_submission.csv', index=False, header=False)

print(f"\nFinal ensemble submission saved to 'final_ensemble_submission.csv'")
print(f"Final Validation Performance: {final_ensemble_val_score}")

