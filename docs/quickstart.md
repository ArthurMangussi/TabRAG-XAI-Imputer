# Quick Start

## Basic imputation

```python
import pandas as pd
from tabrag_xai_imputer import RAGImputer

# Load your data
df = pd.read_csv("your_dataset.csv")
X_train = df_train.drop(columns="target")   # complete rows for context
X_test_missing = df_test.drop(columns="target")  # rows with NaN to impute

# Create and fit the imputer
imputer = RAGImputer(
    n_neighbors=5,                        # context rows retrieved per query
    llm_model_name="gemini-2.0-flash",    # model identifier
    llm_api="gemini",                     # "gemini" | "open_router" | "gpt" | "claude"
    dataset_name="My Dataset",            # injected into the LLM prompt
)

imputer.fit(X_train)
X_imputed = imputer.transform(X_test_missing)
```

!!! note
    `fit()` stores complete rows and computes the correlation matrix used for retrieval weighting. It does **not** call the LLM — only `transform()` does.

## Scikit-learn pipeline

`RAGImputer` is fully sklearn-compatible:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tabrag_xai_imputer import RAGImputer

pipe = Pipeline([
    ("imputer", RAGImputer(
        n_neighbors=10,
        llm_model_name="gemini-2.0-flash",
        llm_api="gemini",
        dataset_name="My Dataset",
    )),
    ("scaler", StandardScaler()),
])

pipe.fit(X_train)
X_processed = pipe.transform(X_test_missing)
```

## Explainability

After imputation, call `explain()` to get an LLM-generated reasoning paragraph for every row that had missing values:

```python
explanations = imputer.explain(X_test_missing, X_imputed)

for i, text in enumerate(explanations):
    print(f"[Row {i}]\n{text}\n")
```

Explain only a subset of rows with `row_indices`:

```python
explanations = imputer.explain(X_test_missing, X_imputed, row_indices=[0, 3, 7])
```

## Choosing k (n_neighbors)

Our ablation study shows that **k = 10** provides the best accuracy–cost trade-off across continuous and mixed datasets. For high-dimensional categorical datasets, **k = 5** may be preferable to control prompt length and inference time.

## Choosing a model

| Provider | Model | Notes |
|---|---|---|
| Gemini | `gemini-2.0-flash` | Recommended default — fast and cost-effective |
| Gemini | `gemini-2.5-flash-lite` | Lowest cost option |
| OpenRouter | `openai/gpt-4.1-nano` | Good for comparison |
| Anthropic | `claude-sonnet-4-5` | High quality, higher cost |
