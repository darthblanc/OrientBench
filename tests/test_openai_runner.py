import json

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

TASKS = generate_tasks(SLICE, id_col="name", n=4, seed=0)
runner = OpenAIRunner(model="gpt-4o-mini")


def _make_output_jsonl(n_tasks: int, answer: str = "x") -> str:
    lines = []
    for i in range(n_tasks):
        for orientation in ("row", "col"):
            lines.append(json.dumps({
                "id": f"batch_req_{i}_{orientation}",
                "custom_id": f"{i}_{orientation}",
                "response": {
                    "status_code": 200,
                    "request_id": f"req_{i}_{orientation}",
                    "body": {
                        "choices": [{"message": {"content": answer, "role": "assistant"}}]
                    },
                },
                "error": None,
            }))
    return "\n".join(lines)


def _mock_batch_client(output_jsonl: str = "") -> MagicMock:
    mock = MagicMock()

    uploaded = MagicMock()
    uploaded.id = "file-input-123"
    mock.files.create.return_value = uploaded

    batch = MagicMock()
    batch.id = "batch_abc"
    batch.status = "completed"
    batch.output_file_id = "file-output-456"
    batch.request_counts = MagicMock(completed=8, failed=0, total=8)
    mock.batches.create.return_value = batch
    mock.batches.retrieve.return_value = batch

    content = MagicMock()
    content.text = output_jsonl
    mock.files.content.return_value = content

    return mock


# --- _query (kept for direct use) ---

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


# --- build_batch_requests ---

def test_build_batch_requests_count():
    reqs = runner.build_batch_requests(TASKS)
    assert len(reqs) == len(TASKS) * 2


def test_build_batch_requests_ids():
    reqs = runner.build_batch_requests(TASKS)
    ids = [r["custom_id"] for r in reqs]
    assert any(i.endswith("_row") for i in ids)
    assert any(i.endswith("_col") for i in ids)


def test_build_batch_requests_unique_ids():
    reqs = runner.build_batch_requests(TASKS)
    ids = [r["custom_id"] for r in reqs]
    assert len(ids) == len(set(ids))


def test_build_batch_requests_method_and_url():
    reqs = runner.build_batch_requests(TASKS)
    for r in reqs:
        assert r["method"] == "POST"
        assert r["url"] == "/v1/chat/completions"


def test_build_batch_requests_has_model():
    reqs = runner.build_batch_requests(TASKS)
    for r in reqs:
        assert r["body"]["model"] == runner.model


def test_build_batch_requests_has_messages():
    reqs = runner.build_batch_requests(TASKS)
    for r in reqs:
        msgs = r["body"]["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert isinstance(msgs[0]["content"], str)
        assert len(msgs[0]["content"]) > 0


def test_build_batch_requests_has_max_tokens():
    reqs = runner.build_batch_requests(TASKS)
    for r in reqs:
        assert r["body"]["max_tokens"] == 256


# --- parse_batch_results ---

def test_parse_batch_results_structure():
    content = _make_output_jsonl(2, answer="NYC")
    results = runner.parse_batch_results(content, TASKS[:2])
    assert len(results) == 2
    for r in results:
        assert {"task_kind", "question", "ground_truth", "row_answer", "col_answer", "row_correct", "col_correct"} == set(r.keys())


def test_parse_batch_results_extracts_answer():
    content = _make_output_jsonl(1, answer="Alice")
    results = runner.parse_batch_results(content, TASKS[:1])
    assert results[0]["row_answer"] == "Alice"
    assert results[0]["col_answer"] == "Alice"


def test_parse_batch_results_error_handled():
    content = json.dumps({
        "id": "batch_req_0_row",
        "custom_id": "0_row",
        "response": None,
        "error": {"code": "server_error", "message": "internal error"},
    }) + "\n" + json.dumps({
        "id": "batch_req_0_col",
        "custom_id": "0_col",
        "response": {"status_code": 200, "request_id": "r", "body": {"choices": [{"message": {"content": "NYC", "role": "assistant"}}]}},
        "error": None,
    })
    results = runner.parse_batch_results(content, TASKS[:1])
    assert results[0]["row_answer"] == ""
    assert results[0]["col_answer"] == "NYC"


def test_parse_batch_results_missing_entry_defaults_to_empty():
    results = runner.parse_batch_results("", TASKS[:1])
    assert results[0]["row_answer"] == ""
    assert results[0]["col_answer"] == ""


# --- run (batch flow) ---

def test_run_output_schema(tmp_path):
    csv_file = tmp_path / "tiny.csv"
    SLICE.to_csv(csv_file, index=False)
    r = OpenAIRunner(model="gpt-4o-mini")
    r._client = _mock_batch_client(_make_output_jsonl(4))

    result = r.run(str(csv_file), id_col="name", n=4, seed=0)

    assert {"dataset", "model", "timestamp", "id_col", "results"} <= set(result.keys())


def test_run_output_schema_includes_batch_fields(tmp_path):
    csv_file = tmp_path / "tiny.csv"
    SLICE.to_csv(csv_file, index=False)
    r = OpenAIRunner(model="gpt-4o-mini")
    r._client = _mock_batch_client(_make_output_jsonl(4))

    result = r.run(str(csv_file), id_col="name", n=4, seed=0)

    assert result["mode"] == "batch"
    assert result["batch_id"] == "batch_abc"


def test_run_dataset_name(tmp_path):
    csv_file = tmp_path / "my_data.csv"
    SLICE.to_csv(csv_file, index=False)
    r = OpenAIRunner(model="gpt-4o-mini")
    r._client = _mock_batch_client(_make_output_jsonl(4))

    result = r.run(str(csv_file), id_col="name", n=4, seed=0)
    assert result["dataset"] == "my_data"


def test_run_model_in_output(tmp_path):
    csv_file = tmp_path / "tiny.csv"
    SLICE.to_csv(csv_file, index=False)
    r = OpenAIRunner(model="gpt-4o")
    r._client = _mock_batch_client(_make_output_jsonl(4))

    result = r.run(str(csv_file), id_col="name", n=4, seed=0)
    assert result["model"] == "gpt-4o"


def test_run_results_length(tmp_path):
    csv_file = tmp_path / "tiny.csv"
    SLICE.to_csv(csv_file, index=False)
    r = OpenAIRunner(model="gpt-4o-mini")
    r._client = _mock_batch_client(_make_output_jsonl(8))

    result = r.run(str(csv_file), id_col="name", n=8, seed=0)
    assert len(result["results"]) == 8


def test_run_result_entry_keys(tmp_path):
    csv_file = tmp_path / "tiny.csv"
    SLICE.to_csv(csv_file, index=False)
    r = OpenAIRunner(model="gpt-4o-mini")
    r._client = _mock_batch_client(_make_output_jsonl(4))

    result = r.run(str(csv_file), id_col="name", n=4, seed=0)
    expected = {"task_kind", "question", "ground_truth", "row_answer", "col_answer", "row_correct", "col_correct"}
    for entry in result["results"]:
        assert set(entry.keys()) == expected


def test_run_uploads_file_and_creates_batch(tmp_path):
    csv_file = tmp_path / "tiny.csv"
    SLICE.to_csv(csv_file, index=False)
    r = OpenAIRunner(model="gpt-4o-mini")
    mock = _mock_batch_client(_make_output_jsonl(4))
    r._client = mock

    r.run(str(csv_file), id_col="name", n=4, seed=0)

    mock.files.create.assert_called_once()
    mock.batches.create.assert_called_once_with(
        input_file_id="file-input-123",
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )


def test_run_polls_until_completed(tmp_path):
    csv_file = tmp_path / "tiny.csv"
    SLICE.to_csv(csv_file, index=False)
    r = OpenAIRunner(model="gpt-4o-mini", poll_interval=0)
    mock = MagicMock()

    uploaded = MagicMock()
    uploaded.id = "file-input-123"
    mock.files.create.return_value = uploaded

    in_progress = MagicMock()
    in_progress.id = "batch_abc"
    in_progress.status = "in_progress"
    in_progress.output_file_id = None
    in_progress.request_counts = MagicMock(completed=0, failed=0, total=8)
    mock.batches.create.return_value = in_progress

    completed = MagicMock()
    completed.id = "batch_abc"
    completed.status = "completed"
    completed.output_file_id = "file-output-456"
    completed.request_counts = MagicMock(completed=8, failed=0, total=8)
    mock.batches.retrieve.side_effect = [in_progress, completed]

    content = MagicMock()
    content.text = _make_output_jsonl(4)
    mock.files.content.return_value = content

    r._client = mock
    r.run(str(csv_file), id_col="name", n=4, seed=0)

    assert mock.batches.retrieve.call_count == 2


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
