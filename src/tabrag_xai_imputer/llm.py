import logging
import os
import re
from io import StringIO


import pandas as pd
from dotenv import load_dotenv

_logger = logging.getLogger(__name__)

load_dotenv()


def tratar_dados_infor(data_array):
    raw_strings = data_array.flatten()
    split_data = [line.split(",") for line in raw_strings]
    df = pd.DataFrame(split_data)
    df = df.drop(columns=df.columns[0])
    df = df.apply(pd.to_numeric, errors="ignore")
    return df


def clean_and_parse_llm_data(response_text, expected_shape):
    match = re.search(r"```(?:csv)?\s*(.*?)\s*```", response_text, re.DOTALL)
    content = match.group(1).strip() if match else response_text.strip()

    for separator in [",", r"\s+"]:
        try:
            df_imputed = pd.read_csv(StringIO(content), sep=separator, engine="python")
            return df_imputed
        except Exception:
            continue

    raise ValueError(f"Could not parse LLM response. Expected shape {expected_shape}.")




def llm_impute(
    dataset_name: str,
    X_teste_norm_md: pd.DataFrame,
    model_name: str,
    api: str = "open_router",
) -> pd.DataFrame:
    """Impute missing values in a DataFrame using an LLM.

    Parameters
    ----------
    dataset_name : str
        Human-readable name injected into the prompt.
    X_teste_norm_md : pd.DataFrame
        DataFrame with NaN where values are missing.
    model_name : str
        Model identifier for the chosen provider.
    api : {"open_router", "gemini", "gpt", "claude"}, default="open_router"
        LLM provider backend.

    Returns
    -------
    pd.DataFrame
        Fully imputed DataFrame.
    """
    row_start = col_start = row_end = 0

    try:
        match api:
            case "open_router":
                from openai import OpenAI

                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("API_KEY_OPEN_ROUTER"),
                )
            case "gemini":
                from google import genai

                client = genai.Client(
                    api_key=os.getenv("API_KEY_GEMINI"),
                    http_options={"timeout": 10 * 60 * 1000},
                )
            case "gpt":
                from openai import OpenAI

                client = OpenAI(api_key=os.getenv("API_KEY_GPT"))
            case "claude":
                import anthropic

                client = anthropic.Anthropic(api_key=os.getenv("API_KEY_CLAUDE"))
            case _:
                raise ValueError(f"Unknown api: '{api}'")

        output = X_teste_norm_md.copy()

        batch_row = 40
        batch_col = 10
        iter_batch = 0

        n_rows, n_cols = X_teste_norm_md.shape

        for row_start in range(0, n_rows, batch_row):
            row_end = min(row_start + batch_row, n_rows)
            actual_start = row_start
            if (row_end - row_start) < batch_row and n_rows >= batch_row:
                actual_start = row_end - batch_row

            for col_start in range(0, n_cols, batch_col):
                col_end = min(col_start + batch_col, n_cols)
                batch_to_prompt = X_teste_norm_md.iloc[
                    actual_start:row_end, col_start:col_end
                ]
                _logger.info("Batch = %d", iter_batch)

                match api:
                    case "open_router":
                        response = client.responses.create(
                            model=model_name,
                            temperature=0.05,
                            input=adjust_prompt(
                                dataset_name=dataset_name, missing_data=batch_to_prompt
                            ),
                        )
                        imputed_value_str = response.output[0].content[0].text

                    case "gpt":
                        response = client.responses.create(
                            model=model_name,
                            input=adjust_prompt(
                                dataset_name=dataset_name, missing_data=batch_to_prompt
                            ),
                        )
                        imputed_value_str = response.output_text

                    case "gemini":
                        from google.genai import types

                        response = client.models.generate_content(
                            model=model_name,
                            contents=adjust_prompt(
                                dataset_name=dataset_name, missing_data=batch_to_prompt
                            ),
                            config=types.GenerateContentConfig(
                                temperature=0.1,
                                thinking_config=types.ThinkingConfig(thinking_budget=0),
                            ),
                        )
                        imputed_value_str = response.text.strip()

                    case "claude":
                        response = client.messages.create(
                            model=model_name,
                            max_tokens=10000,
                            messages=[
                                {
                                    "role": "user",
                                    "content": adjust_prompt(
                                        dataset_name=dataset_name,
                                        missing_data=batch_to_prompt,
                                    ),
                                }
                            ],
                            temperature=0.1,
                        )
                        imputed_value_str = response.content[0].text

                df_imputed = clean_and_parse_llm_data(
                    response_text=imputed_value_str,
                    expected_shape=batch_to_prompt.shape,
                )

                rows_needed = row_end - row_start
                clean_imputed_data = df_imputed.iloc[-rows_needed:, :]
                actual_rows = clean_imputed_data.shape[0]

                for col in clean_imputed_data.columns:
                    if col not in output.columns:
                        clean_imputed_data = clean_imputed_data.drop(columns=col)

                if actual_rows != rows_needed:
                    _logger.warning("LLM returned a different DataFrame shape")
                    output.iloc[row_start:row_end, col_start:col_end] = output.iloc[
                        row_start:row_end, col_start:col_end
                    ].values

                elif clean_imputed_data.empty:
                    _logger.warning("LLM returned an empty DataFrame")
                    output.iloc[row_start:row_end, col_start:col_end] = output.iloc[
                        row_start:row_end, col_start:col_end
                    ].values

                else:
                    try:
                        _logger.info("LLM imputation succeeded")
                        output.iloc[row_start:row_end, col_start:col_end] = (
                            clean_imputed_data.values
                        )
                    except Exception:
                        _logger.warning("LLM output required post-processing")
                        df_tratado = tratar_dados_infor(clean_imputed_data.values)
                        output.iloc[row_start:row_end, col_start:col_end] = (
                            df_tratado.values
                        )

                iter_batch += 1

    except Exception as e:
        _logger.error(
            "Error in batch [%d:%d, %d:%d]: %s",
            row_start,
            row_end,
            col_start,
            col_end,
            e,
        )
        raise ValueError(e) from e

    if output.isna().any().any():
        _logger.warning("LLM failed to impute some values — applying column-mean fallback")
        for col in output.columns:
            if output[col].isna().any():
                mean_val = output[col].astype(float).mean()
                fill_val = mean_val if not pd.isna(mean_val) else 0.5
                output[col] = output[col].fillna(fill_val)

    return output


