
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pytorch_tabnet.tab_model import TabNetRegressor
import torch

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv") # Load test data for later prediction

# Preprocessing
# Fill missing total_bedrooms with the median
# Addressing FutureWarning: avoid inplace=True with chained assignment
train_df['total_bedrooms'] = train_df['total_bedrooms'].fillna(train_df['total_bedrooms'].median())
test_df['total_bedrooms'] = test_df['total_bedrooms'].fillna(train_df['total_bedrooms'].median()) # Use training median for test set

# Define features and target
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income']
target = 'median_house_value'

X = train_df[features]
y = train_df[target]

# Split training data for validation (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to NumPy arrays and ensure correct types and shapes for TabNet
# TabNet expects float32 and target as a 2D array (n_samples, 1)
X_train_np = X_train.values.astype(np.float32)
y_train_np = y_train.values.astype(np.float32).reshape(-1, 1)
X_val_np = X_val.values.astype(np.float32)
y_val_np = y_val.values.astype(np.float32).reshape(-1, 1)

# Prepare test data for prediction
X_test_np = test_df[features].values.astype(np.float32)

# Initialize TabNet Regressor
# Removed 'random_state' from TabNetRegressor constructor as it's not a valid argument.
# Reproducibility is handled by torch.manual_seed and np.random.seed.
model = TabNetRegressor(verbose=0,
                        optimizer_fn=torch.optim.Adam,
                        optimizer_params=dict(lr=2e-2),
                        scheduler_params={"step_size":50, "gamma":0.9},
                        scheduler_fn=torch.optim.lr_scheduler.StepLR)

# Train the model
# max_epochs: number of training epochs
# patience: number of epochs without improvement to wait before early stopping
# eval_set is used for early stopping and reporting metrics during training
model.fit(X_train=X_train_np, y_train=y_train_np,
          eval_set=[(X_val_np, y_val_np)],
          eval_metric=['rmse'],
          max_epochs=100, patience=20,
          batch_size=1024, virtual_batch_size=128)

# Make predictions on the validation set
y_val_pred = model.predict(X_val_np)

# Evaluate RMSE on the validation set
rmse_val = np.sqrt(mean_squared_error(y_val_np, y_val_pred))

# Print the final validation performance
print(f'Final Validation Performance: {rmse_val}')

# Make predictions on the test set
test_predictions = model.predict(X_test_np).flatten()

# Create submission file
submission_df = pd.DataFrame({'median_house_value': test_predictions})
submission_df.to_csv('submission.csv', index=False, header=True)
