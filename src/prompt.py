from typing import Literal
from src.tasks import Task
from src.orient import to_row_wise, to_col_wise

_TEMPLATE = """\
Here is a dataset in {label} CSV format:

{csv_text}
Question: {question}
Reply with only the exact answer value, nothing else."""


def build_prompt(task: Task, orientation: Literal["row", "col"]) -> str:
    if orientation == "row":
        csv_text = to_row_wise(task.df_slice)
        label = "row-wise"
    else:
        csv_text = to_col_wise(task.df_slice)
        label = "column-wise"
    return _TEMPLATE.format(label=label, csv_text=csv_text, question=task.question)
