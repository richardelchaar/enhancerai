
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from pytorch_tabnet.tab_model import TabNetRegressor
import torch

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Load data
try:
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the ./input directory.")
    # Exit is not allowed, so re-raise or handle gracefully for demonstration
    raise

# Separate target variable
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Identify numerical features (all are numerical in this dataset)
numerical_features = X.columns.tolist()

# Handle potential missing values (though dataset description doesn't explicitly mention them, good practice)
# Fill with median for numerical features
for col in numerical_features:
    if X[col].isnull().any():
        X[col].fillna(X[col].median(), inplace=True)
    if test_df[col].isnull().any():
        test_df[col].fillna(test_df[col].median(), inplace=True)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df)

X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_features)
test_scaled_df = pd.DataFrame(test_scaled, columns=numerical_features)


# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42
)

# --- Model Training ---

# 1. Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf_val = rf_model.predict(X_val)
y_pred_rf_test = rf_model.predict(test_scaled_df)

# 2. Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb_val = gb_model.predict(X_val)
y_pred_gb_test = gb_model.predict(test_scaled_df)

# 3. TabNet Regressor
# Reduce max_epochs to prevent timeout, from original 200 to 50
tabnet_model = TabNetRegressor(
    seed=42,
    verbose=0,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size": 50, "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='sparsemax',
)

# Convert to numpy for TabNet
X_train_np = X_train.values
y_train_np = y_train.values.reshape(-1, 1)
X_val_np = X_val.values
y_val_np = y_val.values.reshape(-1, 1)
test_scaled_np = test_scaled_df.values

tabnet_model.fit(
    X_train=X_train_np,
    y_train=y_train_np,
    eval_set=[(X_val_np, y_val_np)],
    eval_metric=["rmse"],
    max_epochs=50,  # Reduced max_epochs to address timeout
    patience=10,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
)
y_pred_tabnet_val = tabnet_model.predict(X_val_np).flatten()
y_pred_tabnet_test = tabnet_model.predict(test_scaled_np).flatten()

# --- Ensemble Prediction ---
# Simple averaging ensemble
y_pred_ensemble_val = (y_pred_rf_val + y_pred_gb_val + y_pred_tabnet_val) / 3
y_pred_ensemble_test = (y_pred_rf_test + y_pred_gb_test + y_pred_tabnet_test) / 3

# Calculate RMSE on validation set
final_validation_rmse = np.sqrt(mean_squared_error(y_val, y_pred_ensemble_val))

# Print final validation performance
print(f"Final Validation Performance: {final_validation_rmse}")

# Create submission file
submission_df = pd.DataFrame({"median_house_value": y_pred_ensemble_test})
submission_df.to_csv("submission.csv", index=False)

# Optional: You can print a confirmation message
# print("Submission file created successfully!")

