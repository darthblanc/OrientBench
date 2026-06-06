from datetime import datetime
from pathlib import Path

import ollama
import pandas as pd

from src.prompt import build_prompt
from src.runners.base import BaseRunner
from src.tasks import generate_tasks


class OllamaRunner(BaseRunner):

    def __init__(self, model: str):
        self.model = model

    def _query(self, prompt: str) -> str:
        resp = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"num_ctx": 2048},
        )
        return resp.response.strip()

    def run(
        self,
        csv_path: str,
        id_col: str,
        n: int = 20,
        seed: int = 42,
        max_rows: int = 15,
        cols: list[str] | None = None,
    ) -> dict:
        df = pd.read_csv(csv_path)
        tasks = generate_tasks(df, id_col=id_col, n=n, seed=seed, max_rows=max_rows, cols=cols)
        dataset = Path(csv_path).stem

        results = []
        for i, task in enumerate(tasks):
            print(f"  [{i+1}/{n}] {task.kind}: {task.question[:60]}...", flush=True)
            row_answer = self._query(build_prompt(task, "row"))
            col_answer = self._query(build_prompt(task, "col"))
            results.append(self._build_result_entry(task, row_answer, col_answer))

        return {
            "dataset": dataset,
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "id_col": id_col,
            "results": results,
        }
