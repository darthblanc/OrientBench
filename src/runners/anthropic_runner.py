import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

from src.prompt import build_prompt
from src.runners.base import BaseRunner
from src.tasks import Task, generate_tasks

load_dotenv()


class AnthropicRunner(BaseRunner):

    def __init__(self, model: str, poll_interval: int = 30, api_key: str | None = None):
        self.model = model
        self.poll_interval = poll_interval
        self._api_key = api_key
        self._client = None

    @property
    def client(self) -> Anthropic:
        if self._client is None:
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def build_batch_requests(self, tasks: list[Task]) -> list[dict]:
        requests = []
        for i, task in enumerate(tasks):
            for orientation in ("row", "col"):
                requests.append({
                    "custom_id": f"{i}_{orientation}",
                    "params": {
                        "model": self.model,
                        "max_tokens": 256,
                        "messages": [{"role": "user", "content": build_prompt(task, orientation)}],
                    },
                })
        return requests

    @staticmethod
    def _extract_text(message: dict) -> str:
        for block in message.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                return block["text"].strip()
            if hasattr(block, "type") and block.type == "text":
                return block.text.strip()
        return ""

    def parse_batch_results(self, raw_results: list, tasks: list[Task]) -> list[dict]:
        by_id: dict[str, str] = {}
        for r in raw_results:
            cid = r["custom_id"] if isinstance(r, dict) else r.custom_id
            result = r["result"] if isinstance(r, dict) else r.result
            result_type = result["type"] if isinstance(result, dict) else result.type

            if result_type == "succeeded":
                message = result["message"] if isinstance(result, dict) else result.message
                by_id[cid] = self._extract_text(
                    message if isinstance(message, dict) else {"content": message.content}
                )
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
        n: int = 200,
        seed: int = 42,
        max_rows: int = 15,
        cols: list[str] | None = None,
    ) -> dict:
        df = pd.read_csv(csv_path)
        tasks = generate_tasks(df, id_col=id_col, n=n, seed=seed, max_rows=max_rows, cols=cols)
        dataset = Path(csv_path).stem

        print(f"Building {len(tasks) * 2} batch requests...")
        raw_requests = self.build_batch_requests(tasks)
        batch_requests = [
            Request(
                custom_id=r["custom_id"],
                params=MessageCreateParamsNonStreaming(**r["params"]),
            )
            for r in raw_requests
        ]

        print("Submitting batch...")
        batch = self.client.messages.batches.create(requests=batch_requests)
        print(f"Batch ID: {batch.id}")

        while True:
            batch = self.client.messages.batches.retrieve(batch.id)
            counts = batch.request_counts
            print(f"  Status: {batch.processing_status} — processing: {counts.processing}, done: {counts.succeeded + counts.errored}")
            if batch.processing_status == "ended":
                break
            time.sleep(self.poll_interval)

        print("Collecting results...")
        raw_results = list(self.client.messages.batches.results(batch.id))
        results = self.parse_batch_results(raw_results, tasks)

        return {
            "dataset": dataset,
            "model": self.model,
            "mode": "batch",
            "batch_id": batch.id,
            "timestamp": datetime.now().isoformat(),
            "id_col": id_col,
            "results": results,
        }
