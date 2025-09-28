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

# Required Output Format
You must respond in a single, valid JSON block.

The JSON object must be a list of plan steps.

Each step must have two keys:
- `plan_step_description`: A concise description of the change you will make.
- `code_block_to_refine`: The exact, verbatim code block from the "Current Code" that will be modified for this step.

# Example
```json
[
  {{
    "plan_step_description": "1. Feature Engineering: Create interaction features between 'total_rooms' and 'population' to better capture density.",
    "code_block_to_refine": "X['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)"
  }},
  {{
    "plan_step_description": "2. Hyperparameter Tuning: Tune the 'n_estimators' and 'learning_rate' of the LightGBM model.",
    "code_block_to_refine": "model = lgb.LGBMRegressor(objective='regression_l1', metric='rmse', random_state=42, n_estimators=1000)"
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