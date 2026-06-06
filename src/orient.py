import io
import pandas as pd


def to_row_wise(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


def to_col_wise(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    for col in df.columns:
        values = ",".join(str(v) for v in df[col])
        buf.write(f"{col},{values}\n")
    return buf.getvalue()
