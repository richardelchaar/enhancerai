
# Suppress verbose model output to prevent token explosion
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
# Suppress LightGBM verbosity
os.environ['LIGHTGBM_VERBOSITY'] = '-1'
# Suppress XGBoost verbosity  
os.environ['XGBOOST_VERBOSITY'] = '0'
# Suppress sklearn warnings
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

I have completed the task of performing an ablation study on the provided Python code.

The script I generated identifies and ablates two distinct components:
1.  **Feature Engineering**: The `create_engineered_features` function, which creates `rooms_per_household`, `bedrooms_per_room`, and `population_per_household`.
2.  **Model Ensemble**: The step that averages predictions from `RandomForestRegressor` and `LGBMRegressor`. For the ablation, only the `LGBMRegressor` is used.

The script calculates and clearly prints the validation RMSE for:
*   The **baseline** (original code with both feature engineering and ensemble).
*   **Ablation: No Feature Engineering** (ensemble used, but engineered features removed).
*   **Ablation: No Ensemble (LightGBM Only)** (feature engineering used, but only LightGBM model for prediction).

Finally, it includes a summary section that programmatically determines which ablation led to the largest change (increase in RMSE) compared to the baseline, thereby identifying the most sensitive component.

No further outputs are needed as the request has been fully addressed by the provided Python script.