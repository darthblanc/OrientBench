import random
from dataclasses import dataclass
from typing import Literal

import pandas as pd

@dataclass
class Task:
    kind: Literal["cell_recall", "attr_scan", "comparison", "row_list"]
    question: str
    answer: str
    df_slice: pd.DataFrame


def _sample_slice(df: pd.DataFrame, rng: random.Random, max_rows: int = 15) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.reset_index(drop=True)
    idx = sorted(rng.sample(range(len(df)), max_rows))
    return df.iloc[idx].reset_index(drop=True)


def _non_id_cols(df: pd.DataFrame, id_col: str) -> list[str]:
    return [c for c in df.columns if c != id_col]


def _cell_recall(df: pd.DataFrame, id_col: str, rng: random.Random, max_rows: int = 15) -> Task:
    slice_df = _sample_slice(df, rng, max_rows)
    row = slice_df.sample(1, random_state=rng.randint(0, 2**31)).iloc[0]
    attr_col = rng.choice(_non_id_cols(slice_df, id_col))
    entity = str(row[id_col])
    answer = str(row[attr_col])
    return Task(
        kind="cell_recall",
        question=f"What is the {attr_col} of {entity}?",
        answer=answer,
        df_slice=slice_df,
    )


def _attr_scan(df: pd.DataFrame, id_col: str, rng: random.Random, max_rows: int = 15) -> Task:
    slice_df = _sample_slice(df, rng, max_rows)
    attr_col = rng.choice(_non_id_cols(slice_df, id_col))
    row = slice_df.sample(1, random_state=rng.randint(0, 2**31)).iloc[0]
    value = str(row[attr_col])
    answer = str(row[id_col])
    return Task(
        kind="attr_scan",
        question=f"Which {id_col} has {attr_col} equal to {value}?",
        answer=answer,
        df_slice=slice_df,
    )


def _comparison(df: pd.DataFrame, id_col: str, rng: random.Random, max_rows: int = 15) -> Task:
    slice_df = _sample_slice(df, rng, max_rows)
    numeric_cols = [c for c in _non_id_cols(slice_df, id_col)
                    if pd.api.types.is_numeric_dtype(slice_df[c])]
    if not numeric_cols:
        return _cell_recall(df, id_col, rng, max_rows)
    attr_col = rng.choice(numeric_cols)
    best_idx = slice_df[attr_col].idxmax()
    answer = str(slice_df.loc[best_idx, id_col])
    return Task(
        kind="comparison",
        question=f"Which {id_col} has the highest {attr_col}?",
        answer=answer,
        df_slice=slice_df,
    )


def _row_list(df: pd.DataFrame, id_col: str, rng: random.Random, max_rows: int = 15) -> Task:
    slice_df = _sample_slice(df, rng, max_rows)
    row = slice_df.sample(1, random_state=rng.randint(0, 2**31)).iloc[0]
    entity = str(row[id_col])
    parts = [f"{col}={row[col]}" for col in _non_id_cols(slice_df, id_col)]
    answer = ", ".join(parts)
    return Task(
        kind="row_list",
        question=f"List all attributes of {entity} as key=value pairs.",
        answer=answer,
        df_slice=slice_df,
    )


_GENERATORS = [_cell_recall, _attr_scan, _comparison, _row_list]


def generate_tasks(
    df: pd.DataFrame,
    id_col: str,
    n: int = 20,
    seed: int = 42,
    max_rows: int = 15,
    cols: list[str] | None = None,
) -> list[Task]:
    if cols is not None:
        keep = [id_col] + [c for c in cols if c != id_col and c in df.columns]
        df = df[keep]
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        gen = _GENERATORS[i % len(_GENERATORS)]
        tasks.append(gen(df, id_col, rng, max_rows))
    return tasks
