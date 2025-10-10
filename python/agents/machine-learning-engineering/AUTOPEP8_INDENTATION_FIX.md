# Autopep8 Indentation Fix Implementation

## What Was Implemented

Added automatic indentation fixing using `autopep8` to the code execution pipeline.

**Files Modified:**
1. `pyproject.toml` / `poetry.lock` - Added autopep8 dependency
2. `machine_learning_engineering/shared_libraries/code_util.py` - Added autopep8 preprocessing

## How It Works

Before executing any generated code, we now run it through `autopep8.fix_code()` with aggressive mode:

```python
code_text = autopep8.fix_code(
    code_text,
    options={
        'aggressive': 2,  # Level 2 aggressive fixes
        'max_line_length': 200,  # Allow longer lines for ML code
    }
)
```

This happens in `run_python_code()` **before** the suppression code is added and **before** the file is saved.

---

## What Autopep8 CAN Fix ✅

### 1. **Uniform Over-Indentation** ✅
When the entire code block is shifted right:

```python
# BEFORE (all lines have 4 extra spaces):
    import pandas as pd
    import numpy as np
    X = load_data()
    model.fit(X)

# AFTER (dedented to column 0):
import pandas as pd
import numpy as np
X = load_data()
model.fit(X)
```

### 2. **PEP 8 Style Violations** ✅
- Inconsistent spacing around operators
- Missing blank lines between functions
- Trailing whitespace

---

## What Autopep8 CANNOT Fix ❌

### 1. **Structural Syntax Errors** ❌
Code placed outside proper blocks:

```python
# BEFORE (if statement outside try block):
try:
    if not os.path.exists('./file.csv'):
        print("Creating file...")

if not os.path.exists('./other.csv'):  # ❌ Should be inside try!
        print("Creating other file...")
        
    data = pd.read_csv('./file.csv')  # ❌ Indented but outside if!
except FileNotFoundError:
    print("Error")

# AFTER autopep8:
# NO CHANGE - autopep8 can't fix structural errors!
```

This is a **logic error**, not a style error. The LLM put code in the wrong scope.

### 2. **Mixed Indentation at Random Levels** ❌
When code alternates between different indentation levels unpredictably:

```python
# BEFORE:
import pandas as pd
    X = load_data()     # 4 spaces (wrong!)
y = X['target']         # 0 spaces (inconsistent!)
        model.fit(X, y) # 8 spaces (very wrong!)

# AFTER autopep8:
# NO CHANGE or PARTIAL CHANGE - too inconsistent to fix automatically
```

---

## What This Fixes in Practice

Based on testing, autopep8 will help with:

1. ✅ **LLM over-indenting entire ablation scripts** (common when LLM thinks it's inside a function)
2. ✅ **Consistent style across generated code blocks**
3. ✅ **Minor indentation drift** (e.g., all lines shifted by 2-4 spaces consistently)

**But it won't fix:**
1. ❌ **The specific error in your example** (if statement outside try block)
2. ❌ **Code in wrong scope/block**
3. ❌ **Severely malformed code structure**

---

## For Your Specific Error

Your example has this problem:

```python
try:
    if not os.path.exists('./input/train.csv'):  # ✓ Inside try
        # ... code ...

if not os.path.exists('./input/test.csv'):      # ❌ Outside try!
        # ... code ...
```

**This requires a prompt fix, not a code fix.**

### Recommended Prompt Addition

Add this to `ABLATION_INSTR` and `ABLATION_SEQ_INSTR`:

```
# CRITICAL BLOCK STRUCTURE RULES:
1. If you open a try block, ALL related code must be inside it before the except/finally
2. Do NOT place if statements outside the try block if they should be inside
3. Maintain consistent block structure - every line after "try:" should be indented until "except:"
4. Double-check that your code blocks have consistent scope

WRONG EXAMPLE:
try:
    if condition1:
        do_something()
if condition2:  # ❌ WRONG - outside try block!
        do_something_else()
    more_code()  # ❌ WRONG - confusing indentation
except:
    pass

CORRECT EXAMPLE:
try:
    if condition1:
        do_something()
    if condition2:  # ✓ CORRECT - inside try block
        do_something_else()
    more_code()  # ✓ CORRECT - inside try block
except:
    pass
```

---

## Testing the Fix

To verify autopep8 is working:

```bash
# Test that autopep8 is installed and working
poetry run python -c "import autopep8; print('✓ autopep8 loaded')"

# Run your pipeline - any over-indented code will be automatically fixed
python -m dotenv run -- python run_meta.py --task_name california-housing-prices --num_runs 2
```

Check the generated Python files in `machine_learning_engineering/workspace/*/ablation_*.py` - they should have cleaner indentation even if the LLM output was messy.

---

## Summary

- ✅ **Installed:** autopep8 via Poetry
- ✅ **Integrated:** Into `code_util.py` execution pipeline
- ✅ **Will fix:** Over-indented code blocks, PEP 8 violations
- ⚠️ **Won't fix:** Structural syntax errors (need prompt improvements)
- 💡 **Next step:** Enhance ablation prompts to prevent structural errors

The fix is live and will help reduce indentation errors, but some prompt improvements are still needed for the most severe cases.

