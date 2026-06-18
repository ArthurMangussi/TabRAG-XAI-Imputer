# TabRAG-XAI-Imputer

**Retrieval-Augmented Generation for Missing Data Imputation in Tabular Data**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/ArthurMangussi/TabRAG-XAI-Imputer/blob/main/LICENSE)
[![scikit-learn compatible](https://img.shields.io/badge/sklearn-compatible-orange.svg)](https://scikit-learn.org/)

**TabRAG-XAI-Imputer** is a novel imputation framework that combines correlation-weighted retrieval with the generative power of Large Language Models. Rather than relying solely on statistical distance (like KNN) or parametric knowledge (like zero-shot LLMs), TabRAG retrieves the most relevant complete records from your dataset and feeds them as grounded context to an LLM, enabling accurate, dataset-specific imputation.

Evaluated across 20 datasets under MAR and MNAR missing-data mechanisms, TabRAG achieves the **lowest overall MAE under MNAR** and **ranks second under MAR**, trailing only MICE.

---

## How it works

TabRAG operates in three stages for each incomplete row:

```
Incomplete row
      │
      ▼
┌─────────────────────────────┐
│  1. Correlation-weighted    │  Ranks observed features by their
│     context retrieval       │  correlation with missing features,
│                             │  then retrieves the k most similar
│                             │  complete rows via weighted distance
└─────────────┬───────────────┘
              │  k complete rows
              ▼
┌─────────────────────────────┐
│  2. Row serialisation       │  Converts retrieved rows and the
│                             │  incomplete query into structured
│                             │  key=value text representations
└─────────────┬───────────────┘
              │  Structured prompt
              ▼
┌─────────────────────────────┐
│  3. LLM-based generation    │  LLM predicts missing values using
│                             │  retrieved context as grounding;
│                             │  output enforced as CSV for parsing
└─────────────────────────────┘
```

---

## Key features

- **Sklearn-compatible** — drop-in replacement with `fit` / `transform` / `fit_transform`
- **Multi-provider LLM support** — Gemini, OpenAI, OpenRouter, Anthropic Claude
- **Correlation-weighted retrieval** — smarter neighbour selection than uniform KNN
- **XAI explainability** — call `.explain()` to get LLM-generated reasoning per imputed row
- **Batch processing** — configurable `llm_batch_size` to balance speed and cost

---

## Quick example

```python
from tabrag_xai_imputer import RAGImputer

imputer = RAGImputer(
    n_neighbors=5,
    llm_model_name="gemini-2.0-flash",
    llm_api="gemini",
    dataset_name="Pima Indians Diabetes",
)

imputer.fit(X_train)
X_imputed = imputer.transform(X_test_missing)

# Optional: get LLM explanations for each imputed row
explanations = imputer.explain(X_test_missing, X_imputed)
```

---

## Supported LLM providers

| Provider | `llm_api` value | Extra |
|---|---|---|
| Google Gemini | `"gemini"` | `pip install "tabrag-xai-imputer[gemini]"` |
| OpenAI | `"gpt"` | `pip install "tabrag-xai-imputer[openai]"` |
| OpenRouter | `"open_router"` | `pip install "tabrag-xai-imputer[openai]"` |
| Anthropic Claude | `"claude"` | `pip install "tabrag-xai-imputer[claude]"` |

> **Recommended:** Gemini Flash offers the best accuracy-to-cost ratio across our benchmarks.
