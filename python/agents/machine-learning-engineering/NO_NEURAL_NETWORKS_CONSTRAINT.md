# Neural Networks Constraint Implementation

## Overview

Added explicit constraints across all agent prompts to prevent the system from suggesting or implementing neural network / deep learning approaches. The framework now strictly uses traditional ML algorithms only.

---

## Rationale

Traditional ML algorithms (LightGBM, XGBoost, CatBoost, RandomForest, etc.) are:
- ✅ **Faster to train** - No need for GPU infrastructure
- ✅ **More interpretable** - Easier to understand feature importance
- ✅ **Less data-hungry** - Work well with smaller datasets
- ✅ **More robust** - Less prone to overfitting on tabular data
- ✅ **Faster iteration** - Quick experimentation cycles

Neural networks are:
- ❌ Slower to train (require GPUs for reasonable speed)
- ❌ Harder to debug and interpret
- ❌ Require careful architecture design
- ❌ Often overkill for tabular data competitions
- ❌ Slower iteration cycles

---

## Files Modified

### 1. **Initialization Agent** (`machine_learning_engineering/sub_agents/initialization/prompt.py`)

#### MODEL_RETRIEVAL_INSTR (line 23)
Added constraint during model search phase:
```python
- **IMPORTANT: Only suggest traditional ML algorithms (e.g., LightGBM, XGBoost, CatBoost, 
  RandomForest, Ridge, Lasso). DO NOT suggest neural networks, deep learning, or 
  PyTorch/TensorFlow models.**
```

#### MODEL_EVAL_INSTR (line 47)
Added constraint during model implementation:
```python
- **CRITICAL: Only use traditional ML algorithms (LightGBM, XGBoost, CatBoost, RandomForest, 
  Ridge, Lasso, etc.). DO NOT use neural networks, deep learning, PyTorch, TensorFlow, or Keras.**
```

Also **removed** the misleading line:
- ❌ "Use PyTorch rather than TensorFlow. Use CUDA if you need."

---

### 2. **Refinement Agent** (`machine_learning_engineering/sub_agents/refinement/prompt.py`)

#### PLAN_GENERATION_BASELINE_INSTR (lines 215-217)
Added constraint for baseline refinement runs:
```python
**CRITICAL CONSTRAINT:**
- **ONLY use traditional ML algorithms**: LightGBM, XGBoost, CatBoost, RandomForest, 
  Ridge, Lasso, ElasticNet, GradientBoosting, etc.
- **DO NOT use neural networks, deep learning, PyTorch, TensorFlow, Keras, or any 
  neural architecture**
```

Also updated "Model Selection" to explicitly list traditional ML models only.

#### PLAN_GENERATION_ENHANCED_INSTR (lines 280-282)
Added same constraint for strategic refinement runs (Run 1+):
```python
**CRITICAL CONSTRAINT:**
- **ONLY use traditional ML algorithms**: LightGBM, XGBoost, CatBoost, RandomForest, 
  Ridge, Lasso, ElasticNet, GradientBoosting, etc.
- **DO NOT use neural networks, deep learning, PyTorch, TensorFlow, Keras, or any 
  neural architecture**
```

Updated focus areas:
- Changed "Model Architecture: add/remove layers" → "Model Selection: switch between traditional ML"
- Removed "dropout" from regularization (neural network concept)
- Changed "transfer learning" → "automated feature selection with traditional ML models"

---

### 3. **Enhancer Agent** (`machine_learning_engineering/sub_agents/enhancer/prompt.py`)

#### Strategic Goals Guidance (lines 29-31)
Added constraint for enhancer's strategic goal suggestions:
```python
**CRITICAL CONSTRAINT:**
- **ONLY suggest traditional ML algorithms**: LightGBM, XGBoost, CatBoost, RandomForest, 
  Ridge, Lasso, ElasticNet, GradientBoosting, etc.
- **DO NOT suggest neural networks, deep learning, PyTorch, TensorFlow, Keras, or any 
  neural architecture**
```

Updated valid target agent phase descriptions:
- "refinement": Now specifies "model selection (traditional ML only)"
- "ensemble": Now specifies "using traditional ML models"

---

## Approved Traditional ML Algorithms

The following algorithms are **allowed** and encouraged:

### Tree-Based Methods
- ✅ LightGBM (Microsoft's gradient boosting)
- ✅ XGBoost (Extreme Gradient Boosting)
- ✅ CatBoost (Yandex's gradient boosting, handles categorical features well)
- ✅ RandomForest
- ✅ ExtraTrees
- ✅ GradientBoosting (sklearn's implementation)
- ✅ HistGradientBoosting (sklearn's histogram-based boosting)

### Linear Methods
- ✅ Ridge Regression (L2 regularization)
- ✅ Lasso Regression (L1 regularization)
- ✅ ElasticNet (L1 + L2 regularization)
- ✅ Logistic Regression (for classification)
- ✅ SGDRegressor / SGDClassifier

### Other Traditional Methods
- ✅ Support Vector Machines (SVM)
- ✅ K-Nearest Neighbors (KNN)
- ✅ Naive Bayes
- ✅ Decision Trees

### Ensemble Strategies
- ✅ Stacking (using traditional ML as base/meta learners)
- ✅ Blending
- ✅ Weighted averaging
- ✅ Voting classifiers/regressors

---

## Forbidden Approaches

The following are **NOT allowed**:

### Neural Network Frameworks
- ❌ PyTorch
- ❌ TensorFlow
- ❌ Keras
- ❌ JAX
- ❌ MXNet

### Neural Architectures
- ❌ Multi-layer Perceptrons (MLP)
- ❌ Convolutional Neural Networks (CNN)
- ❌ Recurrent Neural Networks (RNN, LSTM, GRU)
- ❌ Transformers
- ❌ Autoencoders
- ❌ GANs
- ❌ Any deep learning architecture

### Neural Network Concepts
- ❌ Backpropagation
- ❌ Dropout layers
- ❌ Batch normalization
- ❌ Attention mechanisms
- ❌ Embedding layers (unless for traditional feature engineering)
- ❌ Transfer learning (unless with traditional ML)

---

## Impact on Framework Behavior

### Run 0 (Discovery)
- Google Search will only return traditional ML models
- Model evaluation will only implement tree-based or linear models
- Initialization will create baseline solutions using LightGBM, XGBoost, or similar

### Run 1+ (Refinement)
- Ablation studies focus on traditional ML components
- Refinement plans only suggest traditional ML improvements
- Hyperparameter tuning targets tree/linear model parameters
- Model selection switches between traditional algorithms only

### Enhancer Strategic Goals
- Will only suggest traditional ML optimization strategies
- Ensemble goals will combine traditional models
- No neural network architectures in strategic recommendations

---

## Verification

To verify the constraint is working, check that generated code:

1. ✅ Only imports: `lightgbm`, `xgboost`, `catboost`, `sklearn.*`
2. ✅ Only uses models like: `LGBMRegressor`, `XGBRegressor`, `CatBoostRegressor`, `RandomForestRegressor`
3. ❌ Never imports: `torch`, `tensorflow`, `keras`, `tf`
4. ❌ Never mentions: "neural", "deep learning", "MLP", "CNN", "dropout"

---

## Testing

Run the pipeline and verify no neural networks are suggested:

```bash
python -m dotenv run -- python run_meta.py --task_name california-housing-prices --num_runs 2
```

Check generated files:
- `run_0/1/init_code_*.py` - Should use LightGBM/XGBoost/CatBoost
- `run_1/1/ablation_0.py` - Should ablate traditional ML components only
- `run_1/1/train1_*.py` - Should only tune traditional ML hyperparameters

---

## Summary

All agent prompts now have explicit, consistent constraints preventing neural network usage. The framework will focus exclusively on traditional ML algorithms, which are more suitable for most tabular data competitions and offer faster iteration cycles.

**Constraint Location Summary:**
- ✅ Initialization: Model retrieval + model evaluation
- ✅ Refinement: Both baseline and enhanced planning prompts
- ✅ Enhancer: Strategic goal generation
- ✅ Applies to: Discovery, refinement, and ensemble phases

The system is now optimized for traditional ML workflows! 🚀

