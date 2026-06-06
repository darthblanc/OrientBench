import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from src.runners.openai_runner import OpenAIRunner
from src.tasks import generate_tasks

SLICE = pd.DataFrame({
    "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "age": [30, 25, 35, 28, 31],
    "city": ["NYC", "LA", "Chicago", "NYC", "LA"],
    "score": [88, 72, 91, 65, 79],
})


def _mock_completion(text: str):
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def test_requires_model_arg():
    with pytest.raises(TypeError):
        OpenAIRunner()


def test_model_stored():
    assert OpenAIRunner(model="gpt-4o-mini").model == "gpt-4o-mini"


def test_query_calls_chat_completions():
    r = OpenAIRunner(model="gpt-4o-mini")
    r._client = MagicMock()
    r._client.chat.completions.create.return_value = _mock_completion("NYC")

    result = r._query("some prompt")

    r._client.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "some prompt"}],
        max_tokens=256,
    )
    assert result == "NYC"


def test_query_strips_whitespace():
    r = OpenAIRunner(model="gpt-4o-mini")
    r._client = MagicMock()
    r._client.chat.completions.create.return_value = _mock_completion("  NYC  ")
    assert r._query("x") == "NYC"


def test_run_output_schema(tmp_path):
    csv_file = tmp_path / "tiny.csv"
    SLICE.to_csv(csv_file, index=False)
    r = OpenAIRunner(model="gpt-4o-mini")
    r._client = MagicMock()
    r._client.chat.completions.create.return_value = _mock_completion("x")

    result = r.run(str(csv_file), id_col="name", n=4, seed=0)

    assert {"dataset", "model", "timestamp", "id_col", "results"} <= set(result.keys())


def test_run_dataset_name(tmp_path):
    csv_file = tmp_path / "my_data.csv"
    SLICE.to_csv(csv_file, index=False)
    r = OpenAIRunner(model="gpt-4o-mini")
    r._client = MagicMock()
    r._client.chat.completions.create.return_value = _mock_completion("x")

    result = r.run(str(csv_file), id_col="name", n=4, seed=0)
    assert result["dataset"] == "my_data"


def test_run_model_in_output(tmp_path):
    csv_file = tmp_path / "tiny.csv"
    SLICE.to_csv(csv_file, index=False)
    r = OpenAIRunner(model="gpt-4o")
    r._client = MagicMock()
    r._client.chat.completions.create.return_value = _mock_completion("x")

    result = r.run(str(csv_file), id_col="name", n=4, seed=0)
    assert result["model"] == "gpt-4o"


def test_run_results_length(tmp_path):
    csv_file = tmp_path / "tiny.csv"
    SLICE.to_csv(csv_file, index=False)
    r = OpenAIRunner(model="gpt-4o-mini")
    r._client = MagicMock()
    r._client.chat.completions.create.return_value = _mock_completion("x")

    result = r.run(str(csv_file), id_col="name", n=8, seed=0)
    assert len(result["results"]) == 8


def test_run_result_entry_keys(tmp_path):
    csv_file = tmp_path / "tiny.csv"
    SLICE.to_csv(csv_file, index=False)
    r = OpenAIRunner(model="gpt-4o-mini")
    r._client = MagicMock()
    r._client.chat.completions.create.return_value = _mock_completion("x")

    result = r.run(str(csv_file), id_col="name", n=4, seed=0)
    expected = {"task_kind", "question", "ground_truth", "row_answer", "col_answer", "row_correct", "col_correct"}
    for entry in result["results"]:
        assert set(entry.keys()) == expected


def test_runner_passes_api_key_to_client():
    with patch("src.runners.openai_runner.OpenAI") as mock_cls:
        r = OpenAIRunner(model="gpt-4o-mini", api_key="sk-test-456")
        _ = r.client
        mock_cls.assert_called_once_with(api_key="sk-test-456")


def test_runner_passes_none_when_no_key_provided():
    with patch("src.runners.openai_runner.OpenAI") as mock_cls:
        r = OpenAIRunner(model="gpt-4o-mini")
        _ = r.client
        mock_cls.assert_called_once_with(api_key=None)


def test_run_calls_query_twice_per_task(tmp_path):
    csv_file = tmp_path / "tiny.csv"
    SLICE.to_csv(csv_file, index=False)
    r = OpenAIRunner(model="gpt-4o-mini")
    r._client = MagicMock()
    r._client.chat.completions.create.return_value = _mock_completion("x")

    r.run(str(csv_file), id_col="name", n=3, seed=0)
    assert r._client.chat.completions.create.call_count == 6  # 3 tasks × 2 orientations
