# Examples

## Cross-validation benchmark

A full 5-fold stratified cross-validation loop that injects 30% MAR missing values and evaluates imputation quality via MAE.

```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from tabrag_xai_imputer import RAGImputer

df = pd.read_csv("data/pima-indians-diabetes/pima_diabetes.csv")
X = df.drop(columns="target")
y = df["target"].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(cv.split(X.values, y), start=1):
    X_train = pd.DataFrame(X.values[train_idx], columns=X.columns)
    X_test  = pd.DataFrame(X.values[test_idx],  columns=X.columns)

    # Normalise — fit on train only to prevent data leakage
    scaler = MinMaxScaler().fit(X_train)
    X_train_norm = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
    X_test_norm  = pd.DataFrame(scaler.transform(X_test),  columns=X.columns)

    # Inject 30% MAR missing values
    import numpy as np
    rng = np.random.default_rng(fold)
    X_test_missing = X_test_norm.copy()
    mask = rng.random(X_test_missing.shape) < 0.30
    X_test_missing[mask] = np.nan

    # Fit and impute
    imputer = RAGImputer(
        n_neighbors=10,
        llm_api="gemini",
        llm_model_name="gemini-2.0-flash",
        dataset_name="Pima Indians Diabetes",
    )
    imputer.fit(X_train_norm)
    X_imputed = imputer.transform(X_test_missing)

    # MAE on imputed positions only
    mae = np.abs(X_imputed[mask] - X_test_norm.values[mask]).mean()
    print(f"Fold {fold} — MAE: {mae:.4f}")
```

---

## Explainability walkthrough

Demonstrate `.explain()` on a held-out test set with artificially injected MCAR missing values.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tabrag_xai_imputer import RAGImputer

DATASET_NAME = "Pima Indians Diabetes"
MISSING_RATE = 0.15
RANDOM_STATE = 42

df = pd.read_csv("data/pima-indians-diabetes/pima_diabetes.csv")
X = df.select_dtypes(include="number").drop(
    columns=["Outcome"], errors="ignore"
).astype(float)

X_train, X_test = train_test_split(X, test_size=0.1, random_state=RANDOM_STATE)

# Inject MCAR missing values
rng = np.random.default_rng(RANDOM_STATE)
X_test_missing = X_test.copy()
miss_mask = rng.random(X_test_missing.shape) < MISSING_RATE
for i, row_mask in enumerate(miss_mask):
    if row_mask.all():           # ensure at least one observed feature
        row_mask[rng.integers(0, X_test_missing.shape[1])] = False
    X_test_missing.iloc[i, row_mask] = np.nan

# Fit → impute → explain
imputer = RAGImputer(
    n_neighbors=5,
    feature_weighting="correlation",
    llm_model_name="gemini-2.0-flash",
    llm_api="gemini",
    dataset_name=DATASET_NAME,
    llm_batch_size=1,
)
imputer.fit(X_train)
X_imputed = imputer.transform(X_test_missing)
X_imputed_df = pd.DataFrame(X_imputed, columns=X_test.columns, index=X_test.index)

explanations = imputer.explain(X_test_missing, X_imputed_df)

# Pretty-print results
for idx, explanation in zip(
    X_test_missing.index[X_test_missing.isna().any(axis=1)],
    explanations,
):
    missing_cols = X_test_missing.columns[X_test_missing.loc[idx].isna()].tolist()
    imputed_vals = {col: round(X_imputed_df.loc[idx, col], 4) for col in missing_cols}
    print(f"\n[Row {idx}]")
    print(f"  Missing  : {missing_cols}")
    print(f"  Imputed  : {imputed_vals}")
    print(f"  Reasoning: {explanation.strip()}")
    print("-" * 70)
```

### Sample output

```
[Row 42]
  Missing  : ['Glucose', 'BMI']
  Imputed  : {'Glucose': 0.5312, 'BMI': 0.4827}
  Reasoning: The retrieved neighbors show a consistent pattern of moderate
  glucose levels (0.45–0.60) for patients in this age and blood-pressure
  range. The imputed Glucose value of 0.53 reflects the central tendency
  of the three closest neighbors, all of whom share similar Insulin and
  SkinThickness profiles. For BMI, the retrieved records cluster tightly
  around 0.47–0.52, and the absence of extreme Insulin values further
  supports a near-average BMI estimate of 0.48.
------------------------------------------------------------------------
```
