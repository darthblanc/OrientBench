import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.runners.ollama_runner import OllamaRunner

SLICE = pd.DataFrame({
    "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "age": [30, 25, 35, 28, 31],
    "city": ["NYC", "LA", "Chicago", "NYC", "LA"],
    "score": [88, 72, 91, 65, 79],
})


def _mock_resp(text="Alice"):
    r = MagicMock()
    r.response = text
    return r


def test_requires_model_arg():
    with pytest.raises(TypeError):
        OllamaRunner()


def test_model_stored():
    assert OllamaRunner(model="llama3:8b").model == "llama3:8b"


def test_query_calls_ollama_generate():
    r = OllamaRunner(model="qwen2.5:3b")
    with patch("src.runners.ollama_runner.ollama.generate", return_value=_mock_resp("NYC")) as mock_gen:
        result = r._query("hello")
        mock_gen.assert_called_once_with(
            model="qwen2.5:3b",
            prompt="hello",
            options={"num_ctx": 2048},
        )
    assert result == "NYC"


def test_query_strips_whitespace():
    r = OllamaRunner(model="qwen2.5:3b")
    with patch("src.runners.ollama_runner.ollama.generate", return_value=_mock_resp("  NYC  ")):
        assert r._query("x") == "NYC"


def test_run_output_schema(tmp_path):
    csv_file = tmp_path / "tiny.csv"
    SLICE.to_csv(csv_file, index=False)
    r = OllamaRunner(model="qwen2.5:3b")
    with patch("src.runners.ollama_runner.ollama.generate", return_value=_mock_resp()):
        result = r.run(str(csv_file), id_col="name", n=4, seed=0)
    assert {"dataset", "model", "timestamp", "id_col", "results"} <= set(result.keys())


def test_run_dataset_name(tmp_path):
    csv_file = tmp_path / "my_data.csv"
    SLICE.to_csv(csv_file, index=False)
    r = OllamaRunner(model="qwen2.5:3b")
    with patch("src.runners.ollama_runner.ollama.generate", return_value=_mock_resp()):
        result = r.run(str(csv_file), id_col="name", n=4, seed=0)
    assert result["dataset"] == "my_data"


def test_run_model_in_output(tmp_path):
    csv_file = tmp_path / "tiny.csv"
    SLICE.to_csv(csv_file, index=False)
    r = OllamaRunner(model="llama3:8b")
    with patch("src.runners.ollama_runner.ollama.generate", return_value=_mock_resp()):
        result = r.run(str(csv_file), id_col="name", n=4, seed=0)
    assert result["model"] == "llama3:8b"


def test_run_results_length(tmp_path):
    csv_file = tmp_path / "tiny.csv"
    SLICE.to_csv(csv_file, index=False)
    r = OllamaRunner(model="qwen2.5:3b")
    with patch("src.runners.ollama_runner.ollama.generate", return_value=_mock_resp()):
        result = r.run(str(csv_file), id_col="name", n=8, seed=0)
    assert len(result["results"]) == 8


def test_run_result_entry_keys(tmp_path):
    csv_file = tmp_path / "tiny.csv"
    SLICE.to_csv(csv_file, index=False)
    r = OllamaRunner(model="qwen2.5:3b")
    with patch("src.runners.ollama_runner.ollama.generate", return_value=_mock_resp()):
        result = r.run(str(csv_file), id_col="name", n=4, seed=0)
    expected = {"task_kind", "question", "ground_truth", "row_answer", "col_answer", "row_correct", "col_correct"}
    for entry in result["results"]:
        assert set(entry.keys()) == expected


def test_run_calls_query_twice_per_task(tmp_path):
    csv_file = tmp_path / "tiny.csv"
    SLICE.to_csv(csv_file, index=False)
    r = OllamaRunner(model="qwen2.5:3b")
    with patch("src.runners.ollama_runner.ollama.generate", return_value=_mock_resp()) as mock_gen:
        r.run(str(csv_file), id_col="name", n=3, seed=0)
    assert mock_gen.call_count == 6
