import pandas as pd
import pytest
from unittest.mock import MagicMock
from unittest.mock import patch
from src.runners.anthropic_runner import AnthropicRunner
from src.tasks import generate_tasks

SLICE = pd.DataFrame({
    "name": ["Alice", "Bob", "Carol"],
    "age": [30, 25, 35],
    "city": ["NYC", "LA", "Chicago"],
})

TASKS = generate_tasks(SLICE, id_col="name", n=4, seed=0)

runner = AnthropicRunner(model="claude-haiku-4-5")


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


def test_build_batch_requests_has_model():
    reqs = runner.build_batch_requests(TASKS)
    for r in reqs:
        assert r["params"]["model"] == runner.model


def test_build_batch_requests_has_messages():
    reqs = runner.build_batch_requests(TASKS)
    for r in reqs:
        msgs = r["params"]["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert isinstance(msgs[0]["content"], str)
        assert len(msgs[0]["content"]) > 0


def test_parse_batch_results_structure():
    fake_results = [
        {"custom_id": "0_row", "result": {"type": "succeeded", "message": {"content": [{"type": "text", "text": "NYC"}]}}},
        {"custom_id": "0_col", "result": {"type": "succeeded", "message": {"content": [{"type": "text", "text": "NYC"}]}}},
        {"custom_id": "1_row", "result": {"type": "succeeded", "message": {"content": [{"type": "text", "text": "Alice"}]}}},
        {"custom_id": "1_col", "result": {"type": "succeeded", "message": {"content": [{"type": "text", "text": "Bob"}]}}},
    ]
    results = runner.parse_batch_results(fake_results, TASKS[:2])
    assert len(results) == 2
    for r in results:
        assert "task_kind" in r
        assert "question" in r
        assert "ground_truth" in r
        assert "row_answer" in r
        assert "col_answer" in r
        assert "row_correct" in r
        assert "col_correct" in r


def test_parse_batch_results_error_handled():
    fake_results = [
        {"custom_id": "0_row", "result": {"type": "errored", "error": {"type": "server_error"}}},
        {"custom_id": "0_col", "result": {"type": "succeeded", "message": {"content": [{"type": "text", "text": "NYC"}]}}},
    ]
    results = runner.parse_batch_results(fake_results, TASKS[:1])
    assert results[0]["row_answer"] == ""
    assert results[0]["row_correct"] is False


def test_extract_text_from_dict_block():
    msg = {"content": [{"type": "text", "text": "hello"}]}
    assert AnthropicRunner._extract_text(msg) == "hello"


def test_extract_text_from_sdk_object():
    block = MagicMock()
    block.type = "text"
    block.text = "world"
    assert AnthropicRunner._extract_text({"content": [block]}) == "world"


def test_extract_text_empty_content():
    assert AnthropicRunner._extract_text({"content": []}) == ""


def test_runner_passes_api_key_to_client():
    with patch("src.runners.anthropic_runner.Anthropic") as mock_cls:
        r = AnthropicRunner(model="claude-haiku-4-5", api_key="sk-test-123")
        _ = r.client
        mock_cls.assert_called_once_with(api_key="sk-test-123")


def test_runner_passes_none_when_no_key_provided():
    with patch("src.runners.anthropic_runner.Anthropic") as mock_cls:
        r = AnthropicRunner(model="claude-haiku-4-5")
        _ = r.client
        mock_cls.assert_called_once_with(api_key=None)


def test_run_output_schema_includes_batch_fields(tmp_path):
    csv_file = tmp_path / "tiny.csv"
    SLICE.to_csv(csv_file, index=False)

    fake_batch = MagicMock()
    fake_batch.id = "msgbatch_test123"
    fake_batch.processing_status = "ended"
    fake_batch.request_counts = MagicMock(processing=0, succeeded=8, errored=0)

    mock_client = MagicMock()
    mock_client.messages.batches.create.return_value = fake_batch
    mock_client.messages.batches.retrieve.return_value = fake_batch
    mock_client.messages.batches.results.return_value = iter([])

    r = AnthropicRunner(model="claude-haiku-4-5")
    r._client = mock_client

    result = r.run(str(csv_file), id_col="name", n=4, seed=0)

    assert result["mode"] == "batch"
    assert result["batch_id"] == "msgbatch_test123"
    assert result["model"] == "claude-haiku-4-5"
    assert "dataset" in result
    assert "timestamp" in result
    assert "results" in result
