import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from src.prompt import build_prompt
from src.runners.base import BaseRunner
from src.tasks import Task, generate_tasks

load_dotenv()

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}


class OpenAIRunner(BaseRunner):

    def __init__(self, model: str, poll_interval: int = 30, api_key: str | None = None):
        self.model = model
        self.poll_interval = poll_interval
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

    def build_batch_requests(self, tasks: list[Task]) -> list[dict]:
        requests = []
        for i, task in enumerate(tasks):
            for orientation in ("row", "col"):
                requests.append({
                    "custom_id": f"{i}_{orientation}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": [{"role": "user", "content": build_prompt(task, orientation)}],
                        "max_tokens": 256,
                    },
                })
        return requests

    def parse_batch_results(self, content: str, tasks: list[Task]) -> list[dict]:
        by_id: dict[str, str] = {}
        for line in content.strip().splitlines():
            if not line:
                continue
            item = json.loads(line)
            cid = item["custom_id"]
            response = item.get("response")
            if response and response.get("status_code") == 200:
                by_id[cid] = response["body"]["choices"][0]["message"]["content"].strip()
            else:
                by_id[cid] = ""

        results = []
        for i, task in enumerate(tasks):
            row_answer = by_id.get(f"{i}_row", "")
            col_answer = by_id.get(f"{i}_col", "")
            results.append(self._build_result_entry(task, row_answer, col_answer))
        return results

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

        logger.info(f"Building {len(tasks) * 2} batch requests...")
        raw_requests = self.build_batch_requests(tasks)
        jsonl_bytes = "\n".join(json.dumps(r) for r in raw_requests).encode()

        logger.info("Uploading batch file...")
        uploaded = self.client.files.create(
            file=("batch.jsonl", jsonl_bytes, "application/jsonl"),
            purpose="batch",
        )

        logger.info("Submitting batch...")
        batch = self.client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        logger.info(f"Batch ID: {batch.id}")

        while True:
            batch = self.client.batches.retrieve(batch.id)
            counts = batch.request_counts
            logger.info(f"  Status: {batch.status} — completed: {counts.completed}, failed: {counts.failed}, total: {counts.total}")
            if batch.status in _TERMINAL_STATUSES:
                break
            time.sleep(self.poll_interval)

        logger.info("Collecting results...")
        content = self.client.files.content(batch.output_file_id).text
        results = self.parse_batch_results(content, tasks)

        return {
            "dataset": dataset,
            "model": self.model,
            "mode": "batch",
            "batch_id": batch.id,
            "timestamp": datetime.now().isoformat(),
            "id_col": id_col,
            "results": results,
        }
