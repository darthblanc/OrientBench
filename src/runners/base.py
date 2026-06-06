from abc import ABC, abstractmethod

from src.score import score, score_row_list
from src.tasks import Task


class BaseRunner(ABC):

    @abstractmethod
    def run(
        self,
        csv_path: str,
        id_col: str,
        n: int = 20,
        seed: int = 42,
        max_rows: int = 15,
        cols: list[str] | None = None,
    ) -> dict: ...

    @staticmethod
    def _score_task(task: Task, predicted: str) -> bool:
        if task.kind == "row_list":
            return score_row_list(predicted, task.answer)
        return score(predicted, task.answer)

    @staticmethod
    def _build_result_entry(task: Task, row_answer: str, col_answer: str) -> dict:
        return {
            "task_kind": task.kind,
            "question": task.question,
            "ground_truth": task.answer,
            "row_answer": row_answer,
            "col_answer": col_answer,
            "row_correct": BaseRunner._score_task(task, row_answer),
            "col_correct": BaseRunner._score_task(task, col_answer),
        }
