from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from src.prompt import build_prompt
from src.runners.base import BaseRunner
from src.tasks import generate_tasks

load_dotenv()


class OpenAIRunner(BaseRunner):

    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self._api_key = api_key
        self._client = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _query(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()

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
