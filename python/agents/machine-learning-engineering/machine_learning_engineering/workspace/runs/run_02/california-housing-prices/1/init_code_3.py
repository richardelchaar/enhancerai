
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from pytorch_tabnet.tab_model import TabNetRegressor

# Load the datasets
train_df = pd.read_csv("./input/train.csv")
# test_df is not used for validation metric calculation, but often useful for feature engineering
# test_df = pd.read_csv("./input/test.csv")

# Define features (X) and target (y)
TARGET = 'median_house_value'
# All columns except the target are features
FEATURES = [col for col in train_df.columns if col != TARGET]

X = train_df[FEATURES]
y = train_df[TARGET]

# Handle missing values
# The 'total_bedrooms' column is known to have missing values in this dataset.
# We'll use SimpleImputer with a 'median' strategy.
imputer = SimpleImputer(strategy='median')

# Fit the imputer on the training features and transform them
X_imputed = imputer.fit_transform(X)

# Convert the imputed numpy array back to a DataFrame for easier handling if needed,
# but TabNet will convert it to numpy internally or expects numpy.
# For consistency with TabNet input, we'll keep it as numpy array for splitting.
X_imputed_df = pd.DataFrame(X_imputed, columns=FEATURES)


# Split the training data into training and validation sets
# A fixed random_state is used for reproducibility.
# The validation set will be 20% of the training data.
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
    X_imputed, y.values, test_size=0.2, random_state=42
)

# Reshape y for TabNet
y_train_np = y_train_np.reshape(-1, 1)
y_val_np = y_val_np.reshape(-1, 1)

# Initialize and train the TabNet Regressor
# Parameters are chosen to be reasonable defaults based on the example.
# 'eval_metric' is set to 'rmse' as per the task's metric for early stopping.
model = TabNetRegressor(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":50, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax', # or 'sparsemax'
    verbose=0, # Suppress verbose output
    seed=42 # For reproducibility
)

# Train the model on the training data
model.fit(
    X_train=X_train_np, y_train=y_train_np,
    eval_set=[(X_val_np, y_val_np)],
    eval_metric=['rmse'], # Monitor RMSE on validation set
    max_epochs=200, # Increased epochs for potentially better convergence
    patience=30, # Early stopping if no improvement for 30 epochs
    batch_size=1024,
    virtual_batch_size=128
)

# Make predictions on the validation set
y_pred_val = model.predict(X_val_np)

# Evaluate the model using Root Mean Squared Error (RMSE)
rmse_val = np.sqrt(mean_squared_error(y_val_np, y_pred_val))

# Print the final validation performance in the required format
print(f"Final Validation Performance: {rmse_val}")
