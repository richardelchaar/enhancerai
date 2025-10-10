# Eliminating try/except Blocks to Fix Indentation Errors

## Problem Diagnosis

The user identified that indentation errors in ablation scripts are primarily caused by **try/except blocks**. When LLMs copy-paste code 3+ times for ablation variants, they lose track of which code belongs inside the `try` block, leading to:

```python
# PROBLEMATIC PATTERN:
try:
    if not os.path.exists('./input/train.csv'):
        print("Creating file...")

if not os.path.exists('./input/test.csv'):  # ❌ Outside try block!
        print("Creating other file...")       # ❌ Wrong indentation
        
    data = pd.read_csv('./file.csv')         # ❌ Confusing placement
except FileNotFoundError:
    print("Error")
```

The LLM loses track of block scope and produces code that's syntactically invalid.

---

## Root Cause

1. **Long code blocks**: Ablation scripts require copying code 3-4 times (baseline + ablations)
2. **Nested scope**: try/except adds an extra level of indentation
3. **LLM limitations**: Probabilistic models struggle with mechanical copy-paste over long sequences
4. **Scope tracking**: LLMs lose track of whether they're inside or outside the try block

Similar issues can occur with deeply nested if/else blocks.

---

## Solution: Eliminate try/except Blocks

Instead of trying to teach the LLM to handle try/except correctly, we **eliminate the problematic construct entirely**.

### Strategy

1. **No try/except blocks**: Explicitly forbid them in all prompts
2. **Assume files exist**: Let errors occur naturally (they'll be caught by the debug loop)
3. **Defensive checks**: Use simple if conditions for predictable errors (e.g., null checks)
4. **Flat code structure**: Minimize nesting depth

---

## Files Modified

### 1. **Ablation Prompts** (`refinement/prompt.py`)

#### Updated: ABLATION_INSTR and ABLATION_SEQ_INSTR

**Before (lines 68-73):**
```
**CRITICAL - Block Structure Rules:**
- If original code has try/except blocks, ensure ALL code sections are properly inside
- Do NOT place if statements or other code outside the try block
- Every line after "try:" must be consistently indented until "except:"
```

**After (lines 68-91):**
```
**CRITICAL - Avoid Indentation Problems:**
- **DO NOT generate try/except blocks** - they cause indentation errors
- **Instead of try/except**: Check conditions before operations
- **Keep if/else blocks simple**: Avoid deeply nested if/else
- **All code at the same scope should start at the same column** (0, 4, 8, etc.)

**Example - Handle errors WITHOUT try/except:**
```python
# WRONG - try/except causes indentation errors:
# try:
#     train_df = pd.read_csv('./input/train.csv')
# except FileNotFoundError:
#     print("Error loading data")

# CORRECT - Assume files exist:
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')
# Handle missing values directly
if train_df['column'].isnull().any():
    train_df['column'].fillna(0, inplace=True)
```
```

**Impact:** 
- Ablation scripts will now have flat structure
- No try/except blocks to lose track of
- Simpler, more maintainable code

---

### 2. **Initialization Prompts** (`initialization/prompt.py`)

#### Updated: MODEL_EVAL_INSTR (line 58-59)

**Before:**
```
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior.
```

**After:**
```
- Do not use exit() function in the Python code.
- **CRITICAL: Do NOT use try/except blocks** - they cause indentation errors. 
  Assume files exist and handle errors by checking conditions (e.g., `if data.isnull().any():`).
- Keep if/else blocks simple and avoid deep nesting.
```

#### Updated: BUG_REFINE_INSTR (line 87)

**After:**
```
- **CRITICAL: Do NOT use try/except blocks** - they cause indentation errors. 
  Assume files exist and handle errors by checking conditions.
```

#### Updated: CODE_INTEGRATION_INSTR (line 122)

**After:**
```
- **CRITICAL: Do NOT use try/except blocks** - they cause indentation errors. 
  Assume files exist and handle errors by checking conditions.
```

#### Updated: CHECK_DATA_USE_INSTR (line 135)

**Before:**
```
Do not bypass using try-except.
DO NOT USE TRY and EXCEPT; just occur error so we can debug it!
```

**After:**
```
**CRITICAL: Do NOT use try/except blocks** - they cause indentation errors. 
Let errors occur naturally so we can debug them.
```

**Impact:**
- Initial solutions won't have try/except blocks
- When solutions are debugged/refined, no try/except will be added
- Cleaner baseline code for ablation studies

---

### 3. **Plan Implementation Prompt** (`refinement/prompt.py`)

#### Updated: IMPLEMENT_PLAN_STEP_INSTR (line 367)

**Before:**
```
CRITICAL REQUIREMENTS:
1. Ensure your code has CORRECT INDENTATION
2. Do NOT use `return` statements unless inside a function
3. Do NOT define functions unless absolutely necessary
4. Ensure all variable names match
```

**After:**
```
CRITICAL REQUIREMENTS:
1. Ensure your code has CORRECT INDENTATION
2. Do NOT use `return` statements unless inside a function
3. Do NOT define functions unless absolutely necessary
4. **Do NOT use try/except blocks** - they cause indentation errors
5. Ensure all variable names match
```

**Impact:**
- When implementing refinement plan steps, no try/except will be generated
- Cleaner code modifications

---

## Alternative Error Handling Patterns

### Pattern 1: Assume Files Exist (Preferred)

```python
# Just read the files - let errors occur naturally
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')
```

**Rationale:** 
- Files SHOULD exist (placed there by the framework)
- If they don't, the error is a real bug that should be fixed
- The debug loop will catch and fix any issues

### Pattern 2: Defensive Null Checks

```python
# Handle expected data quality issues
if train_df['column'].isnull().any():
    train_df['column'].fillna(train_df['column'].median(), inplace=True)

# Check for unexpected empty data
if len(train_df) == 0:
    print("Warning: Training data is empty")
```

**Rationale:**
- Missing values are expected in real data
- Simple if checks don't cause indentation issues
- No need for try/except

### Pattern 3: Single-line conditionals (If Necessary)

```python
# Optional: Inline checks for edge cases
data = pd.read_csv(file) if os.path.exists(file) else pd.DataFrame()
```

**Rationale:**
- Very flat structure
- No block indentation to track
- Use sparingly (can reduce readability)

---

## What About Real Errors?

**Q:** What if the code actually has an error?

**A:** The framework has a **debug loop** built into every agent:
1. Code is executed
2. If it fails (returncode != 0), error is captured
3. Debug agent receives: original code + error message
4. Debug agent fixes the code
5. Fixed code is re-executed

This is BETTER than try/except because:
- ✅ Errors are visible and fixed properly
- ✅ Root causes are addressed, not hidden
- ✅ The framework learns from mistakes

**Q:** Won't this make the code fragile?

**A:** No, because:
- Files are guaranteed to exist (placed by the framework)
- Data quality issues are handled with defensive checks
- Real errors are caught and fixed by the debug loop
- Production ML code should fail fast, not silently

---

## Expected Impact

### Before (With try/except)

```python
try:
    train_df = pd.read_csv('./input/train.csv')
    if not os.path.exists('./input/test.csv'):
        # Create dummy test data
        test_df = pd.DataFrame()
    
test_df = pd.read_csv('./input/test.csv')  # ❌ Outside try block!
    X = train_df.drop('target', axis=1)     # ❌ Wrong indentation!
except FileNotFoundError:                   # ❌ Misaligned
    print("Error")
```

**Result:** IndentationError → execution fails

### After (Without try/except)

```python
# Load data (assume files exist)
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')

# Handle missing values
if train_df.isnull().any().any():
    train_df = train_df.fillna(train_df.median())

# Prepare features
X = train_df.drop('target', axis=1)
y = train_df['target']
```

**Result:** Clean, flat code → executes successfully

---

## Combined with autopep8

The no-try/except policy works synergistically with the autopep8 fix:

1. **No try/except** → Eliminates structural indentation errors
2. **autopep8** → Fixes minor style issues (tabs, spacing)
3. **Result** → Very robust code generation

### Defense in Depth

| Layer | Purpose | Handles |
|-------|---------|---------|
| **Prompt constraints** | Prevent try/except generation | Structural errors |
| **autopep8 preprocessing** | Fix style issues | Tabs, spacing, over-indentation |
| **Debug loop** | Fix runtime errors | Logic errors, data issues |

---

## Testing

To verify the fix works:

1. Run the pipeline:
   ```bash
   python -m dotenv run -- python run_meta.py --task_name california-housing-prices --num_runs 2
   ```

2. Check generated ablation scripts:
   ```bash
   cat machine_learning_engineering/workspace/california-housing-prices/run_0/1/ablation_0.py
   ```

3. Verify:
   - ✅ No `try:` or `except:` keywords
   - ✅ Flat code structure (minimal nesting)
   - ✅ Simple if checks for null values
   - ✅ Code executes without IndentationError

---

## Summary

**Changes Made:**
- ✅ Updated 6 prompts across initialization and refinement agents
- ✅ Added explicit "NO try/except" constraints
- ✅ Provided concrete examples of alternative patterns
- ✅ Applied to: model evaluation, bug fixing, code integration, ablation generation, plan implementation

**Expected Results:**
- ✅ Dramatically fewer indentation errors
- ✅ Cleaner, more maintainable generated code
- ✅ Faster debugging cycles (errors are explicit, not hidden)
- ✅ Better synergy with autopep8 preprocessing

**Philosophy:**
- Let errors occur naturally → debug loop fixes them
- Simple is better than complex
- Flat is better than nested
- Explicit is better than implicit

The framework now generates robust, maintainable code that's less prone to indentation errors! 🎯

