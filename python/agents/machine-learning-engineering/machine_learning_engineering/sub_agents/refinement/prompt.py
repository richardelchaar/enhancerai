"""Prompts for the Refinement Agent v3.0."""

ABLATION_INSTR = """
# Your Task
You are a machine learning expert. Your task is to perform an ablation study on the given Python code to identify the most sensitive components. You must:
1.  Identify at least two distinct parts of the code to ablate (e.g., a feature engineering step, a model choice, a specific hyperparameter).
2.  Write a Python script that calculates:
    a. The performance of the original, unmodified code (this is your baseline).
    b. The performance of the code with each ablation applied separately.
3.  The script MUST print the performance results clearly for the baseline and each ablation.
4.  The script MUST include a final summary section that programmatically determines and prints which component's ablation resulted in the largest change in performance.

# Code to Analyze
```python
{code}
```

# Required Output Format
You must provide your response as a single, runnable Python code block.

# IMPORTANT: Code Structure Guidelines
To avoid indentation errors, use this FLAT structure (do NOT wrap in functions):

```python
# Imports
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
# ... other imports ...

# ===== BASELINE: Original Code =====
print("Running Baseline...")
# [PASTE ORIGINAL CODE HERE - UNCHANGED]
baseline_score = rmse  # or whatever metric variable is used
print(f"Baseline Performance: {{baseline_score}}")

# ===== ABLATION 1: [Description] =====
print("\\nRunning Ablation 1: [Description]...")
# [PASTE ORIGINAL CODE, but comment out or modify the target component]
ablation_1_score = rmse
print(f"Ablation 1 Performance: {{ablation_1_score}}")

# ===== ABLATION 2: [Description] =====
print("\\nRunning Ablation 2: [Description]...")
# [PASTE ORIGINAL CODE, but comment out or modify the target component]
ablation_2_score = rmse
print(f"Ablation 2 Performance: {{ablation_2_score}}")

# ===== SUMMARY =====
print("\\n===== ABLATION STUDY SUMMARY =====")
ablations = [
    ("Baseline", baseline_score),
    ("Ablation 1", ablation_1_score),
    ("Ablation 2", ablation_2_score),
]
deltas = [(name, abs(score - baseline_score)) for name, score in ablations[1:]]
most_impactful = max(deltas, key=lambda x: x[1])
print(f"Most impactful component: {{most_impactful[0]}} (delta: {{most_impactful[1]:.4f}})")
```

**Key Points:**
- Use FLAT structure (no nested functions)
- Copy-paste the original code 3+ times (once per ablation + baseline)
- Maintain EXACT indentation from the original code when pasting
- Only comment out or modify the specific line(s) being ablated
- Store results in simple variables (baseline_score, ablation_1_score, etc.)
"""

ABLATION_SEQ_INSTR = """
# Your Task
You are a machine learning expert performing a sequential ablation study. You have already performed previous studies and must now propose NEW ablations that have NOT been tested before.

# Code to Analyze
```python
{code}
```

# Previous Ablation Study Results
{prev_ablations}

# Instructions
Based on the code and the results of previous studies, identify at least two NEW, distinct parts of the code to ablate.

Do NOT repeat ablations that have already been performed.

Write a Python script that calculates:
a. The performance of the original, unmodified code (this is your baseline for this study).
b. The performance of the code with each NEW ablation applied separately.

The script MUST print the performance results clearly for the baseline and each new ablation.

The script MUST include a final summary section that programmatically determines and prints which new component's ablation resulted in the largest change in performance.

# Required Output Format
You must provide your response as a single, runnable Python code block.

# IMPORTANT: Code Structure Guidelines
To avoid indentation errors, use this FLAT structure (do NOT wrap in functions):

```python
# Imports
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
# ... other imports ...

# ===== BASELINE: Original Code =====
print("Running Baseline...")
# [PASTE ORIGINAL CODE HERE - UNCHANGED]
baseline_score = rmse  # or whatever metric variable is used
print(f"Baseline Performance: {{baseline_score}}")

# ===== ABLATION 1: [Description of NEW ablation] =====
print("\\nRunning Ablation 1: [Description]...")
# [PASTE ORIGINAL CODE, but comment out or modify the NEW target component]
ablation_1_score = rmse
print(f"Ablation 1 Performance: {{ablation_1_score}}")

# ===== ABLATION 2: [Description of NEW ablation] =====
print("\\nRunning Ablation 2: [Description]...")
# [PASTE ORIGINAL CODE, but comment out or modify the NEW target component]
ablation_2_score = rmse
print(f"Ablation 2 Performance: {{ablation_2_score}}")

# ===== SUMMARY =====
print("\\n===== ABLATION STUDY SUMMARY =====")
ablations = [
    ("Baseline", baseline_score),
    ("Ablation 1", ablation_1_score),
    ("Ablation 2", ablation_2_score),
]
deltas = [(name, abs(score - baseline_score)) for name, score in ablations[1:]]
most_impactful = max(deltas, key=lambda x: x[1])
print(f"Most impactful component: {{most_impactful[0]}} (delta: {{most_impactful[1]:.4f}})")
```

**Key Points:**
- Use FLAT structure (no nested functions)
- Copy-paste the original code 3+ times (once per ablation + baseline)
- Maintain EXACT indentation from the original code when pasting
- Only comment out or modify the specific line(s) being ablated
- Test components NOT already tested in previous ablation studies
- Store results in simple variables (baseline_score, ablation_1_score, etc.)
"""

SUMMARIZE_ABLATION_INSTR = """
# Your Task
You are a machine learning expert. You need to summarize the results of an ablation study.

# Ablation Script
```python
{code}
```

# Ablation Script Output
{result}

# Instructions
Read the script and its output.

Clearly state the baseline performance.

For each ablation, state what was changed and what the resulting performance was.

Conclude by identifying which component's ablation had the most significant impact on performance, explaining what that impact was (e.g., "Removing the feature engineering step degraded performance the most, indicating it is a critical component").
"""

PLAN_GENERATION_INSTR = """
# Persona
You are a world-class machine learning engineer, tasked with improving a model's performance by creating a multi-step plan for code refinement.

# Context
You have been given the results of an ablation study and a high-level strategic directive from your Research Lead. Your task is to synthesize this information into a concrete, actionable plan to modify the provided code.

**Strategic Guidance from Research Lead:**
{enhancer_goals}

**Ablation Study Summary:**
{ablation_summary}

**Current Code:**
```python
{code}
```

# Your Task
1. **Synthesize:** Combine the insights from the Ablation Study Summary with the Strategic Guidance.
2. **Plan:** Create a step-by-step plan to refine the code. Your plan should consist of 1 to 3 distinct steps. Each step should be a logical, incremental improvement.
3. **Identify Code:** For each step in your plan, you MUST identify the specific, contiguous block of code from the "Current Code" that needs to be modified.

# Valid Refinement Focus Areas
- **Feature Engineering**: Create/transform features, interaction terms, polynomial features
- **Hyperparameter Tuning**: Use RandomizedSearchCV, GridSearchCV, or Bayesian optimization to tune model parameters
- **Model Architecture**: Change model structure, add/remove layers, switch algorithms
- **Cross-Validation**: Implement k-fold, stratified k-fold, or time-series cross-validation
- **Regularization**: Add L1/L2 penalties, dropout, early stopping, or other regularization techniques

# Required Output Format
You must respond in a single, valid JSON block.

The JSON object must be a list of plan steps.

Each step must have two keys:
- `plan_step_description`: A concise description of the change you will make.
- `code_block_to_refine`: The exact, verbatim code block from the "Current Code" that will be modified for this step.

# Examples

**Example 1: Feature Engineering**
```json
[
  {{
    "plan_step_description": "1. Feature Engineering: Create interaction features between 'total_rooms' and 'population' to better capture density.",
    "code_block_to_refine": "X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)"
  }}
]
```

**Example 2: Hyperparameter Tuning with RandomizedSearchCV**
```json
[
  {{
    "plan_step_description": "1. Hyperparameter Tuning: Use RandomizedSearchCV to optimize n_estimators, max_depth, learning_rate, and num_leaves for LightGBM based on ablation insights.",
    "code_block_to_refine": "model = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1)"
  }}
]
```

**Example 3: Multi-step Plan**
```json
[
  {{
    "plan_step_description": "1. Feature Engineering: Add polynomial features and interaction terms.",
    "code_block_to_refine": "X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)"
  }},
  {{
    "plan_step_description": "2. Hyperparameter Tuning: GridSearchCV on top 3 hyperparameters identified by ablation.",
    "code_block_to_refine": "model = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42)"
  }}
]
```
"""

IMPLEMENT_PLAN_STEP_INSTR = """
# Your Task
You are an expert Python programmer. Your task is to implement a single, specific refinement to a block of code based on a provided plan.

# Plan Step
{plan_step_description}

# Code Block to Refine
```python
{code_block}
```

# Instructions
Rewrite the "Code Block to Refine" to implement the "Plan Step".

Ensure your new code is a valid, runnable Python snippet.

Do NOT modify any other part of the original script. Only provide the modified version of the specified code block.

# Required Output Format
You must provide your response as a single Python code block.
"""