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

# IMPORTANT: Code Structure Guidelines
To avoid indentation errors, use this FLAT structure (do NOT wrap in functions):

```python
# Imports
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
# ... other imports ...

# ===== Data Loading =====
# Assume files exist. DO NOT add dummy data or file checks.
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')

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

# Required Output Format
You must provide your response as a single, runnable Python code block following the structure above.

# CRITICAL Rules
- **No Dummy Data**: The script must assume `train.csv` and `test.csv` exist in `./input/`. Do not check for them or create fake data. The script MUST fail with `FileNotFoundError` if they are missing.
- **FLAT Structure**: Do not define functions. All code must be at the top level to avoid indentation errors.
- **No `try/except` Blocks**: Handle potential errors with `if` checks, not `try/except`.
- **Preserve Original Code**: When creating the baseline and ablations, copy the user's code exactly, only modifying the specific part being ablated for that section.
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

# IMPORTANT: Code Structure Guidelines
To avoid indentation errors, use this FLAT structure (do NOT wrap in functions):

```python
# Imports
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
# ... other imports ...

# ===== Data Loading =====
# Assume files exist. DO NOT add dummy data or file checks.
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')

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

# Required Output Format
You must provide your response as a single, runnable Python code block following the structure above.

# CRITICAL Rules
- **No Dummy Data**: The script must assume `train.csv` and `test.csv` exist in `./input/`. Do not check for them or create fake data. The script MUST fail with `FileNotFoundError` if they are missing.
- **FLAT Structure**: Do not define functions. All code must be at the top level to avoid indentation errors.
- **No `try/except` Blocks**: Handle potential errors with `if` checks, not `try/except`.
- **Preserve Original Code**: When creating the baseline and ablations, copy the user's code exactly, only modifying the specific part being ablated for that section.
- **Propose NEW Ablations**: Do not repeat ablations from previous studies.
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

# BASELINE PROMPT - For Run 0 (no enhancer guidance)
# Mimics original MLE-STAR behavior: ablation-driven, avoids expensive operations
PLAN_GENERATION_BASELINE_INSTR = """
# Persona
You are a Kaggle grandmaster working to improve a machine learning solution based on ablation study insights.

# Context
You have performed an ablation study on the current code to understand which components are most critical for performance.

**Ablation Study Summary:**
{ablation_summary}

**Current Code:**
```python
{code}
```

# Your Task
Based on the ablation study results, create a **{num_steps_required}-step plan** to improve the code's performance.

**Important Constraints:**
- Focus on changes that the ablation study identified as high-impact
- **Avoid time-consuming operations** like hyperparameter search over very large spaces or extensive feature generation
- Prefer simple, effective improvements that can be implemented and tested quickly
- Each step should target a different aspect of the code

# Valid Refinement Focus Areas (Baseline Run)
- **Feature Engineering**: Create/transform features based on domain knowledge and ablation insights
- **Model Selection**: Try a different traditional ML model (e.g., switch between LightGBM, XGBoost, CatBoost, RandomForest) if ablation shows current model is limited
- **Regularization**: Add or adjust L1/L2 penalties based on overfitting signals
- **Data Preprocessing**: Improve missing value handling, feature scaling, categorical encoding
- **Simple Architecture Changes**: Adjust basic model parameters (e.g., max_depth, n_estimators) without grid search

**CRITICAL CONSTRAINT:**
- **ONLY use traditional ML algorithms**: LightGBM, XGBoost, CatBoost, RandomForest, Ridge, Lasso, ElasticNet, GradientBoosting, etc.
- **DO NOT use neural networks, deep learning, PyTorch, TensorFlow, Keras, or any neural architecture**

# Required Output Format
You must respond with **exactly {num_steps_required} steps** in a single, valid JSON array.

Each step must have two keys:
- `plan_step_description`: A concise description of the change (include step number: "1.", "2.", etc.)
- `code_block_to_refine`: The exact, verbatim code block from the "Current Code" that will be modified

**Example Output:**
```json
[
  {{
    "plan_step_description": "1. Feature Engineering: Create interaction features between 'total_rooms' and 'population' based on ablation showing feature importance.",
    "code_block_to_refine": "X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)"
  }},
  {{
    "plan_step_description": "2. Regularization: Add L2 regularization since ablation showed overfitting on training set.",
    "code_block_to_refine": "model = RandomForestRegressor(n_estimators=100, random_state=42)"
  }}
]
```

Respond with ONLY the JSON array. No additional text.
"""

# ENHANCED PROMPT - For Run 1+ (with enhancer strategic guidance)
# Emphasizes MANDATORY implementation of all enhancer goals
PLAN_GENERATION_ENHANCED_INSTR = """
# Persona
You are a world-class machine learning engineer implementing strategic directives from your Research Lead.

# Context
Your Research Lead has analyzed the previous run(s) and identified critical strategic goals for this iteration. These goals are **MANDATORY** - you must implement ALL of them.

**STRATEGIC GOALS (MANDATORY - MUST IMPLEMENT ALL):**
{enhancer_goals}

**Ablation Study Summary (use to determine HOW and WHERE):**
{ablation_summary}

**Current Code:**
```python
{code}
```

# Your Task
You **MUST** create a plan with **exactly {num_steps_required} steps** that implements **ALL strategic goals** from your Research Lead.

# CRITICAL PLANNING RULES - READ CAREFULLY
1. **STRICT PRIORITY ORDERING**: You MUST implement strategic goals IN THE ORDER THEY APPEAR above.
   - Your Step 1 MUST implement the FIRST strategic goal listed (whichever priority number it has)
   - Your Step 2 MUST implement the SECOND strategic goal listed (whichever priority number it has)
   - Your Step 3 MUST implement the THIRD strategic goal listed (if it exists)
   - The strategic goals are ALREADY SORTED by priority - DO NOT reorder them
   - Example: If you see "Priority 1" first and "Priority 3" second, then Step 1 implements Priority 1, Step 2 implements Priority 3
   
2. **MANDATORY STEP NUMBERING**: Each `plan_step_description` MUST start with YOUR step number (1, 2, 3...) followed by a period.
   - "1. Feature Engineering: ..."
   - "2. Hyperparameter Tuning: ..."
   - "3. Regularization: ..."
   
3. **EXPLICIT PRIORITY REFERENCE**: Each step description MUST explicitly reference the ORIGINAL priority number from the strategic goal.
   - If the first goal listed is "Priority 1: feature_engineering", write: "1. Feature Engineering (Strategic Goal Priority 1): ..."
   - If the second goal listed is "Priority 3: hyperparameter_tuning", write: "2. Hyperparameter Tuning (Strategic Goal Priority 3): ..."
   - YOUR step numbers (1, 2, 3) follow the ORDER of goals. THEIR priority numbers stay as-is in the description.

# Plan Step Requirements
- For **all tasks EXCEPT feature engineering**, each step must have two keys:
  - `plan_step_description`: A concise description of the change.
  - `code_block_to_refine`: The exact, verbatim code block from "Current Code" that will be modified.
- For **feature engineering** tasks, each step must have two DIFFERENT keys:
  - `plan_step_description`: A concise description of the new features.
  - `feature_engineering_function`: A complete, self-contained Python function that takes a pandas DataFrame as input and returns the DataFrame with the new features added. This function must not have any external dependencies other than standard libraries like pandas and numpy.

**Additional Requirements:**
1. **Every strategic goal MUST be addressed** - if there are 3 goals and 3 steps, each step implements one goal
2. **Use ablation insights** to determine the BEST location and approach for implementing each goal
3. **Be specific** - each step must clearly state which strategic goal it implements

# Valid Refinement Focus Areas (Strategic Run)
- **Feature Engineering**: Create/transform features, interaction terms, polynomial features, domain-specific features
- **Hyperparameter Tuning**: Use RandomizedSearchCV, GridSearchCV, or Bayesian optimization to systematically tune parameters
- **Model Selection**: Switch between traditional ML algorithms (LightGBM, XGBoost, CatBoost, RandomForest, Ridge, Lasso)
- **Cross-Validation**: Implement k-fold, stratified k-fold, or time-series cross-validation for robust evaluation
- **Regularization**: Add L1/L2 penalties or other regularization techniques
- **Advanced Techniques**: Stacking, blending, automated feature selection with traditional ML models

**CRITICAL CONSTRAINT:**
- **ONLY use traditional ML algorithms**: LightGBM, XGBoost, CatBoost, RandomForest, Ridge, Lasso, ElasticNet, GradientBoosting, etc.
- **DO NOT use neural networks, deep learning, PyTorch, TensorFlow, Keras, or any neural architecture**

# Required Output Format
You must respond with **exactly {num_steps_required} steps** in a single, valid JSON array.

Each step **MUST** clearly indicate which strategic goal it implements.

Format:
- `plan_step_description`: "N. [Goal Name]: [Specific implementation approach]"
- `code_block_to_refine`: The exact, verbatim code block from "Current Code"
- OR `feature_engineering_function`: A string containing a full Python function.

**Example Output:**
Suppose you receive strategic goals in this order:
- Priority 1: feature_engineering
- Priority 3: hyperparameter_tuning

Your output must be:
```json
[
  {{
    "plan_step_description": "1. Feature Engineering (Strategic Goal Priority 1): Create 'rooms_per_person' and 'population_per_household' to capture density ratios.",
    "feature_engineering_function": "def add_extra_features(df):\\n    df['rooms_per_person'] = df['total_rooms'] / df['population']\\n    df['population_per_household'] = df['population'] / df['households']\\n    return df"
  }},
  {{
    "plan_step_description": "2. Hyperparameter Tuning (Strategic Goal Priority 3): Implement RandomizedSearchCV for LightGBM.",
    "code_block_to_refine": "lgbm_model = lgb.LGBMRegressor(**lgbm_params)"
  }}
]
```
Note: Step 1 implements the first goal (Priority 1), Step 2 implements the second goal (Priority 3, not Priority 2!).

Respond with ONLY the JSON array. No additional text.
"""

IMPLEMENT_FEATURE_ENGINEERING_INSTR = """
# Your Task
You are an expert Python programmer. Your task is to integrate a new feature engineering function into an existing script.

# Plan
{plan_step_description}

# Feature Engineering Function to Add
```python
{feature_engineering_function}
```

# Full Code to Modify
```python
{full_code}
```

# Instructions
1.  **Insert Function**: Copy the provided "Feature Engineering Function to Add" and insert it into the "Full Code to Modify" script. A good place is typically after the imports but before the main script logic begins.
2.  **Apply to a new 'X_processed' DataFrame**: After the main data loading and initial `X` and `y` separation, create a new DataFrame `X_processed` by applying the feature engineering function to `X`.
3.  **Apply to 'test_df'**: Find the line where `test_df` is loaded. Immediately after it, apply the *same* feature engineering function to `test_df` to ensure consistency.
4.  **Update References**: Go through the rest of the script. Change all subsequent uses of `X` (e.g., in `train_test_split`, model training) to use the new `X_processed` DataFrame.
5.  **Return the ENTIRE modified script.**

# CRITICAL REQUIREMENTS
- You **MUST** apply the function to **BOTH** the training data (`X`) and the test data (`test_df`). Failure to do so will cause a `KeyError`.
- Ensure all downstream code uses the new DataFrame that contains the engineered features.
- The final output must be a single, complete, runnable Python script.

# Required Output Format
You must provide your response as a single Python code block containing the entire modified script.
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
Your **only job** is to apply the change described in the "Plan Step" to the "Code Block to Refine".
- **Implement ONLY the requested change.**
- **Do NOT** add comments, refactor, or "improve" the code in any other way.
- **Do NOT** change variable names or logic that is not directly related to the plan step.
- Simply apply the specific change described in the plan.
- The output must be ONLY the modified code block.

# CRITICAL REQUIREMENTS:
1.  **Correct Indentation - EXTREMELY IMPORTANT**:
    - Use EXACTLY 4 spaces per indent level. NO TABS.
    - **PRESERVE THE EXACT INDENTATION LEVEL** of the original code block
    - If the original code starts at indentation level 0, your output MUST start at level 0
    - If you add a for-loop or if-statement, indent its contents by exactly 4 spaces
    - **EVERY LINE must have consistent indentation** - Python will fail with IndentationError otherwise
    - Use a code editor with visible spaces/tabs to verify indentation before submitting
2.  **No `return` Statements**: Do not use `return` unless it's INSIDE a function definition body.
3.  **No New Functions**: Inline the logic; do not define new helper functions unless they already exist.
4.  **No `try/except` Blocks**: Handle errors by checking conditions beforehand (e.g., `if x is not None:`).
5.  **Match Variable Names**: All variable names must EXACTLY match those in the original code.
6.  **Valid Python Syntax**: The final code block MUST be syntactically valid Python that can execute.
7.  **Preserve Imports**: Ensure all necessary imports are present if you add a new library.
8.  **Maintain Scope**: The code must work in the same scope and context as the original block.
9.  **Copy-Paste Safety**: Your output should be directly copy-pastable and runnable without any manual fixes.
10. **XGBoost Early Stopping**: DO NOT use early stopping for XGBoost models (version compatibility issues). Only use early stopping for LightGBM.

# Common Indentation Mistakes to AVOID:
- ❌ Mixing spaces and tabs
- ❌ Inconsistent indentation within loops (e.g., loop body at wrong level)
- ❌ Return statement not indented inside function
- ❌ Code after a for/if statement not indented
- ❌ Over-indenting or under-indenting relative to the original code

# Required Output Format
You must provide your response as a single Python code block wrapped in ```.
"""