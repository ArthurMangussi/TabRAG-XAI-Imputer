# Installation

## Requirements

- Python 3.10 or later
- An API key for at least one supported LLM provider

## Install from PyPI

```bash
pip install tabrag-xai-imputer
```

Install with the extra(s) for your chosen LLM provider:

=== "Gemini (recommended)"
    ```bash
    pip install "tabrag-xai-imputer[gemini]"
    ```

=== "OpenAI / OpenRouter"
    ```bash
    pip install "tabrag-xai-imputer[openai]"
    ```

=== "Anthropic Claude"
    ```bash
    pip install "tabrag-xai-imputer[claude]"
    ```

=== "All providers"
    ```bash
    pip install "tabrag-xai-imputer[all]"
    ```

## API key configuration

There are two ways to supply your API key.

**Option 1 — pass it directly (recommended):**

```python
from tabrag_xai_imputer import RAGImputer

imputer = RAGImputer(
    llm_api="gemini",
    llm_model_name="gemini-2.0-flash",
    api_key="your_gemini_key_here",
)
```

**Option 2 — `.env` file (fallback):**

Place the relevant key(s) in a `.env` file at the root of your project. Only the key for the provider you use is required.

```env
API_KEY_GEMINI=your_gemini_key_here
API_KEY_OPEN_ROUTER=your_openrouter_key_here
API_KEY_GPT=your_openai_key_here
API_KEY_CLAUDE=your_anthropic_key_here
```

The library loads `.env` automatically via `python-dotenv` whenever `api_key` is not set. Alternatively, export the variable in your shell:

```bash
export API_KEY_GEMINI="your_gemini_key_here"
```

!!! warning "Keep your keys safe"
    Never commit `.env` to version control. The default `.gitignore` already excludes it.

## Install from source

```bash
git clone https://github.com/ArthurMangussi/TabRAG-XAI-Imputer.git
cd TabRAG-XAI-Imputer
pip install -e ".[all]"
```
