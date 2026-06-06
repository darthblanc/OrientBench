import io
import json
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import src.api as api_module
from src.api import app, _runs

SAMPLE_CSV = "title,year\nFilm A,2000\nFilm B,2001\n"

FAKE_RESULT = {
    "dataset": "test",
    "model": "claude-haiku-4-5-20251001",
    "mode": "batch",
    "batch_id": "batch_abc",
    "timestamp": "2026-06-04T00:00:00",
    "id_col": "title",
    "results": [],
}


@pytest.fixture(autouse=True)
def clear_runs():
    _runs.clear()
    api_module.limiter._storage.reset()
    yield
    _runs.clear()


@pytest.fixture()
def client():
    return TestClient(app)


# --- Upload size limit ---

def test_post_run_passes_api_key_to_factory(client):
    with patch("src.api.RunnerFactory.create") as mock_factory:
        mock_factory.return_value.run.return_value = FAKE_RESULT
        client.post(
            "/run",
            data={"model": "claude-haiku-4-5", "id_col": "title", "api_key": "sk-ant-test"},
            files={"csv": ("test.csv", SAMPLE_CSV, "text/csv")},
        )
    mock_factory.assert_called_once_with(model="claude-haiku-4-5", api_key="sk-ant-test")


def test_post_run_passes_none_key_when_omitted(client):
    with patch("src.api.RunnerFactory.create") as mock_factory:
        mock_factory.return_value.run.return_value = FAKE_RESULT
        client.post(
            "/run",
            data={"model": "claude-haiku-4-5", "id_col": "title"},
            files={"csv": ("test.csv", SAMPLE_CSV, "text/csv")},
        )
    mock_factory.assert_called_once_with(model="claude-haiku-4-5", api_key=None)


def test_rejects_oversized_upload(client):
    big_csv = "a,b\n" + ("x," * 500 + "\n") * 1000
    response = client.post(
        "/run",
        data={"model": "claude-haiku-4-5", "id_col": "a"},
        files={"csv": ("big.csv", big_csv, "text/csv")},
        headers={"content-length": str(api_module.MAX_UPLOAD_BYTES + 1)},
    )
    assert response.status_code == 413


# --- POST /run ---

def test_post_run_returns_id_and_running_status(client):
    with patch("src.api.RunnerFactory.create", return_value=MagicMock(run=MagicMock(return_value=FAKE_RESULT))):
        response = client.post(
            "/run",
            data={"model": "claude-haiku-4-5-20251001", "id_col": "title", "n": "5"},
            files={"csv": ("test.csv", SAMPLE_CSV, "text/csv")},
        )

    assert response.status_code == 200
    body = response.json()
    assert "id" in body
    assert body["status"] == "running"


def test_post_run_each_call_gets_unique_id(client):
    mock_runner = MagicMock(run=MagicMock(return_value=FAKE_RESULT))
    with patch("src.api.RunnerFactory.create", return_value=mock_runner):
        r1 = client.post(
            "/run",
            data={"model": "claude-haiku-4-5-20251001", "id_col": "title"},
            files={"csv": ("a.csv", SAMPLE_CSV, "text/csv")},
        )
        r2 = client.post(
            "/run",
            data={"model": "claude-haiku-4-5-20251001", "id_col": "title"},
            files={"csv": ("b.csv", SAMPLE_CSV, "text/csv")},
        )

    assert r1.json()["id"] != r2.json()["id"]


# --- GET /run/{id} ---

def test_get_run_not_found(client):
    response = client.get("/run/does-not-exist")
    assert response.status_code == 404


def test_get_run_returns_injected_state(client):
    _runs["my-run"] = {"status": "done", "result": FAKE_RESULT}
    response = client.get("/run/my-run")
    assert response.status_code == 200
    assert response.json()["status"] == "done"


def test_get_run_failed_state(client):
    _runs["bad-run"] = {"status": "failed", "error": "something went wrong"}
    response = client.get("/run/bad-run")
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert "something went wrong" in response.json()["error"]


def test_get_run_clears_entry_after_done(client):
    _runs["my-run"] = {"status": "done", "result": FAKE_RESULT}
    client.get("/run/my-run")
    assert "my-run" not in _runs


def test_get_run_clears_entry_after_failed(client):
    _runs["bad-run"] = {"status": "failed", "error": "boom"}
    client.get("/run/bad-run")
    assert "bad-run" not in _runs


# --- Background execution ---

def test_run_completes_in_background(client):
    with patch("src.api.RunnerFactory.create", return_value=MagicMock(run=MagicMock(return_value=FAKE_RESULT))):
        response = client.post(
            "/run",
            data={"model": "claude-haiku-4-5-20251001", "id_col": "title", "n": "2"},
            files={"csv": ("test.csv", SAMPLE_CSV, "text/csv")},
        )

    run_id = response.json()["id"]

    for _ in range(50):
        if _runs.get(run_id, {}).get("status") != "running":
            break
        time.sleep(0.05)

    state = client.get(f"/run/{run_id}").json()
    assert state["status"] == "done"
    assert state["result"] == FAKE_RESULT
    assert run_id not in _runs


def test_run_failure_stored_in_state(client):
    mock_runner = MagicMock()
    mock_runner.run.side_effect = RuntimeError("model not found")

    with patch("src.api.RunnerFactory.create", return_value=mock_runner):
        response = client.post(
            "/run",
            data={"model": "bogus-model", "id_col": "title"},
            files={"csv": ("test.csv", SAMPLE_CSV, "text/csv")},
        )

    run_id = response.json()["id"]

    for _ in range(50):
        if _runs.get(run_id, {}).get("status") != "running":
            break
        time.sleep(0.05)

    state = client.get(f"/run/{run_id}").json()
    assert state["status"] == "failed"
    assert "model not found" in state["error"]
    assert run_id not in _runs


# --- POST /upload-batch helpers ---

# 3-row CSV with enough variety for generate_tasks to produce n=4 tasks
BATCH_CSV = "name,age,city\nAlice,30,NYC\nBob,25,LA\nCarol,35,Chicago\n"
BATCH_PARAMS = {"model": "claude-haiku-4-5", "id_col": "name", "n": "4", "seed": "0"}


def _anthropic_jsonl(n_tasks: int, answer: str = "x") -> str:
    lines = []
    for i in range(n_tasks):
        for orientation in ("row", "col"):
            lines.append(json.dumps({
                "custom_id": f"{i}_{orientation}",
                "result": {"type": "succeeded", "message": {"content": [{"type": "text", "text": answer}]}},
            }))
    return "\n".join(lines)


def _openai_jsonl(n_tasks: int, answer: str = "x") -> str:
    lines = []
    for i in range(n_tasks):
        for orientation in ("row", "col"):
            lines.append(json.dumps({
                "id": f"batch_req_{i}_{orientation}",
                "custom_id": f"{i}_{orientation}",
                "response": {"status_code": 200, "request_id": "r", "body": {"choices": [{"message": {"content": answer, "role": "assistant"}}]}},
                "error": None,
            }))
    return "\n".join(lines)


# --- POST /upload-batch ---

def test_upload_batch_anthropic_returns_result(client):
    response = client.post(
        "/upload-batch",
        data=BATCH_PARAMS,
        files={
            "csv": ("data.csv", BATCH_CSV, "text/csv"),
            "batch_results": ("results.jsonl", _anthropic_jsonl(4), "application/octet-stream"),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "claude-haiku-4-5"
    assert body["mode"] == "batch_upload"
    assert body["dataset"] == "data"
    assert body["id_col"] == "name"
    assert len(body["results"]) == 4
    for r in body["results"]:
        assert {"task_kind", "question", "ground_truth", "row_answer", "col_answer", "row_correct", "col_correct"} == set(r.keys())


def test_upload_batch_openai_returns_result(client):
    params = {**BATCH_PARAMS, "model": "gpt-4o-mini"}
    response = client.post(
        "/upload-batch",
        data=params,
        files={
            "csv": ("data.csv", BATCH_CSV, "text/csv"),
            "batch_results": ("results.jsonl", _openai_jsonl(4), "application/octet-stream"),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "gpt-4o-mini"
    assert body["mode"] == "batch_upload"
    assert len(body["results"]) == 4
    for r in body["results"]:
        assert {"task_kind", "question", "ground_truth", "row_answer", "col_answer", "row_correct", "col_correct"} == set(r.keys())


def test_upload_batch_answers_extracted(client):
    response = client.post(
        "/upload-batch",
        data=BATCH_PARAMS,
        files={
            "csv": ("data.csv", BATCH_CSV, "text/csv"),
            "batch_results": ("results.jsonl", _anthropic_jsonl(4, answer="Alice"), "application/octet-stream"),
        },
    )
    assert response.status_code == 200
    for r in response.json()["results"]:
        assert r["row_answer"] == "Alice"
        assert r["col_answer"] == "Alice"


def test_upload_batch_invalid_format_returns_400(client):
    response = client.post(
        "/upload-batch",
        data=BATCH_PARAMS,
        files={
            "csv": ("data.csv", BATCH_CSV, "text/csv"),
            "batch_results": ("results.jsonl", json.dumps({"foo": "bar"}), "application/octet-stream"),
        },
    )
    assert response.status_code == 400
    assert "Unrecognised" in response.json()["detail"]


def test_upload_batch_empty_file_returns_400(client):
    response = client.post(
        "/upload-batch",
        data=BATCH_PARAMS,
        files={
            "csv": ("data.csv", BATCH_CSV, "text/csv"),
            "batch_results": ("results.jsonl", "", "application/octet-stream"),
        },
    )
    assert response.status_code == 400


def test_upload_batch_malformed_jsonl_returns_400(client):
    response = client.post(
        "/upload-batch",
        data=BATCH_PARAMS,
        files={
            "csv": ("data.csv", BATCH_CSV, "text/csv"),
            "batch_results": ("results.jsonl", "not json at all!", "application/octet-stream"),
        },
    )
    assert response.status_code == 400


def test_upload_batch_dataset_name_from_filename(client):
    response = client.post(
        "/upload-batch",
        data=BATCH_PARAMS,
        files={
            "csv": ("movies.csv", BATCH_CSV, "text/csv"),
            "batch_results": ("results.jsonl", _anthropic_jsonl(4), "application/octet-stream"),
        },
    )
    assert response.status_code == 200
    assert response.json()["dataset"] == "movies"
