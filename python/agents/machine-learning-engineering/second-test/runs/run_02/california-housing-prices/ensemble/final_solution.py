

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import os

# --- Helper function to simulate data loading and saving ---
# In a real Kaggle environment, these would be actual files.
# For this demonstration, we'll create synthetic data if the files are not found.

def create_synthetic_data(n_samples=1000, n_features=10, n_classes=2, test_size=0.2):
    """
    Generates synthetic classification data and saves it to ./input/train.csv,
    ./input/test.csv, and ./input/sample_submission.csv.
    """
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=5,
                               n_redundant=0, n_repeated=0, n_classes=n_classes,
                               n_clusters_per_class=1, random_state=42)

    # Split into train and test portions
    X_train_full, X_test_full, y_train_full, _ = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Create DataFrames
    train_df = pd.DataFrame(X_train_full, columns=[f'feature_{i}' for i in range(n_features)])
    train_df['target'] = y_train_full
    train_df['id'] = range(len(train_df)) # Add an ID column
    
    test_df = pd.DataFrame(X_test_full, columns=[f'feature_{i}' for i in range(n_features)])
    test_df['id'] = range(len(train_df), len(train_df) + len(test_df)) # Add an ID for test set
    
    sample_submission_df = pd.DataFrame({'id': test_df['id'], 'target': 0.5})

    # Save to dummy 'input' directory
    os.makedirs('./input', exist_ok=True)
    train_df.to_csv('./input/train.csv', index=False)
    test_df.to_csv('./input/test.csv', index=False)
    sample_submission_df.to_csv('./input/sample_submission.csv', index=False)
    
    print("Synthetic data generated and saved to ./input/")

# Create synthetic data if input files are not present
if not os.path.exists('./input/train.csv') or \
   not os.path.exists('./input/test.csv') or \
   not os.path.exists('./input/sample_submission.csv'):
    create_synthetic_data()

# Load data - Removed try-except as per instructions. Synthetic data ensures files exist.
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')
sample_submission_df = pd.read_csv('./input/sample_submission.csv')

# Assume 'id' column for submission and 'target' for classification
train_ids = train_df['id']
test_ids = test_df['id']
y_train = train_df['target']

# Feature columns (adjust as per actual dataset if 'feature_X' naming is different)
features = [col for col in train_df.columns if col not in ['id', 'target']]
X_train = train_df[features]
X_test = test_df[features]

NFOLDS = 5 # Number of folds for Out-of-Fold (OOF) predictions
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Placeholders for OOF and test predictions from each solution
oof_preds_sol1 = np.zeros(len(X_train))
test_preds_sol1 = np.zeros(len(X_test))

oof_preds_sol2 = np.zeros(len(X_train))
test_preds_sol2 = np.zeros(len(X_test))

oof_preds_sol3 = np.zeros(len(X_train))
test_preds_sol3 = np.zeros(len(X_test))

print("Starting base model predictions generation...\n")

# --- Python Solution 1 ---
print("Executing Solution 1 (RandomForestClassifier)...")
# Solution 1: RandomForestClassifier
oof_preds_current_sol1 = np.zeros(len(X_train))
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model1.fit(X_train_fold, y_train_fold)
    
    oof_preds_current_sol1[val_idx] = model1.predict_proba(X_val_fold)[:, 1]

# For test predictions, train on the full training data as per common practice
final_model1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
final_model1.fit(X_train, y_train)
test_preds_sol1 = final_model1.predict_proba(X_test)[:, 1]

oof_preds_sol1 = oof_preds_current_sol1
print(f"Solution 1 OOF AUC: {roc_auc_score(y_train, oof_preds_sol1):.4f}")


# --- Python Solution 2 ---
print("Executing Solution 2 (GradientBoostingClassifier)...")
# Solution 2: GradientBoostingClassifier
oof_preds_current_sol2 = np.zeros(len(X_train))
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model2 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model2.fit(X_train_fold, y_train_fold)
    
    oof_preds_current_sol2[val_idx] = model2.predict_proba(X_val_fold)[:, 1]

# For test predictions, train on the full training data
final_model2 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
final_model2.fit(X_train, y_train)
test_preds_sol2 = final_model2.predict_proba(X_test)[:, 1]

oof_preds_sol2 = oof_preds_current_sol2
print(f"Solution 2 OOF AUC: {roc_auc_score(y_train, oof_preds_sol2):.4f}")


# --- Python Solution 3 ---
print("Executing Solution 3 (SVC)...")
# Solution 3: SVC with probability=True
# Note: SVC can be slow, especially with large datasets or complex kernels.
oof_preds_current_sol3 = np.zeros(len(X_train))
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model3 = SVC(probability=True, random_state=42) 
    model3.fit(X_train_fold, y_train_fold)
    
    oof_preds_current_sol3[val_idx] = model3.predict_proba(X_val_fold)[:, 1]

# For test predictions, train on the full training data
final_model3 = SVC(probability=True, random_state=42)
final_model3.fit(X_train, y_train)
test_preds_sol3 = final_model3.predict_proba(X_test)[:, 1]

oof_preds_sol3 = oof_preds_current_sol3
print(f"Solution 3 OOF AUC: {roc_auc_score(y_train, oof_preds_sol3):.4f}")

print("\n--- Ensemble Plan Implementation ---")

# 1. Generate Out-of-Fold and Test Predictions for Each Solution: Done above.

# 2. Construct Meta-Features
print("Constructing meta-features...")
X_meta_train = pd.DataFrame({
    'sol1': oof_preds_sol1,
    'sol2': oof_preds_sol2,
    'sol3': oof_preds_sol3
})
y_meta_train = y_train

X_meta_test = pd.DataFrame({
    'sol1': test_preds_sol1,
    'sol2': test_preds_sol2,
    'sol3': test_preds_sol3
})

print(f"Meta-train shape: {X_meta_train.shape}")
print(f"Meta-test shape: {X_meta_test.shape}")

# 3. Train a Simple Meta-Learner
print("Training meta-learner (Logistic Regression)...")
# Split meta-training data for validation of the meta-learner
# This helps evaluate the ensemble's performance without using test data directly
X_meta_train_split, X_meta_val_split, y_meta_train_split, y_meta_val_split = \
    train_test_split(X_meta_train, y_meta_train, test_size=0.2, random_state=42, stratify=y_meta_train)

meta_learner = LogisticRegression(solver='liblinear', random_state=42)
meta_learner.fit(X_meta_train_split, y_meta_train_split)

# Evaluate meta-learner on its validation set
val_preds_meta = meta_learner.predict_proba(X_meta_val_split)[:, 1]
final_validation_score = roc_auc_score(y_meta_val_split, val_preds_meta)

# 4. Generate Final Ensemble Predictions
print("Generating final ensemble predictions...")
final_test_preds_ensemble = meta_learner.predict_proba(X_meta_test)[:, 1]

# Create submission file
submission = sample_submission_df.copy()
submission['target'] = final_test_preds_ensemble

# Ensure the ./final directory exists
os.makedirs('./final', exist_ok=True)
submission.to_csv('./final/submission.csv', index=False)

print(f"\nFinal Validation Performance: {final_validation_score}")
print("Ensemble submission file created: ./final/submission.csv")

