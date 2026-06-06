import json
import pytest
from src.registry import list_models, add_model, remove_model


def _seed(tmp_path, entries):
    p = tmp_path / "models.json"
    p.write_text(json.dumps(entries))
    return p


def test_list_returns_dict(tmp_path):
    p = _seed(tmp_path, {"claude-haiku-4-5": "anthropic"})
    assert isinstance(list_models(registry_path=p), dict)


def test_add_writes_to_json(tmp_path):
    p = _seed(tmp_path, {})
    add_model("qwen3:8b", "ollama", registry_path=p)
    assert json.loads(p.read_text())["qwen3:8b"] == "ollama"


def test_add_overwrites_existing(tmp_path):
    p = _seed(tmp_path, {"x": "ollama"})
    add_model("x", "anthropic", registry_path=p)
    assert json.loads(p.read_text())["x"] == "anthropic"


def test_remove_deletes_entry(tmp_path):
    p = _seed(tmp_path, {"x": "ollama"})
    remove_model("x", registry_path=p)
    assert "x" not in json.loads(p.read_text())


def test_remove_nonexistent_raises(tmp_path):
    p = _seed(tmp_path, {})
    with pytest.raises(KeyError):
        remove_model("nonexistent", registry_path=p)


def test_list_after_add(tmp_path):
    p = _seed(tmp_path, {})
    add_model("new-model", "ollama", registry_path=p)
    assert "new-model" in list_models(registry_path=p)
