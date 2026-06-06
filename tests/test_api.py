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
