import json
import pytest
from src.runners.factory import RunnerFactory
from src.runners.base import BaseRunner
from src.runners.ollama_runner import OllamaRunner
from src.runners.anthropic_runner import AnthropicRunner
from src.runners.openai_runner import OpenAIRunner


def _registry(tmp_path, entries):
    p = tmp_path / "models.json"
    p.write_text(json.dumps(entries))
    return p


def test_factory_creates_ollama_runner(tmp_path):
    p = _registry(tmp_path, {"qwen2.5:3b": "ollama"})
    assert isinstance(RunnerFactory.create("qwen2.5:3b", registry_path=p), OllamaRunner)


def test_factory_creates_anthropic_runner(tmp_path):
    p = _registry(tmp_path, {"claude-sonnet-4-6": "anthropic"})
    assert isinstance(RunnerFactory.create("claude-sonnet-4-6", registry_path=p), AnthropicRunner)


def test_factory_unknown_model_raises_value_error(tmp_path):
    p = _registry(tmp_path, {})
    with pytest.raises(ValueError, match="unknown-model"):
        RunnerFactory.create("unknown-model", registry_path=p)


def test_factory_error_message_includes_registry_hint(tmp_path):
    p = _registry(tmp_path, {})
    with pytest.raises(ValueError, match="src.registry add"):
        RunnerFactory.create("missing-model", registry_path=p)


def test_factory_passes_model_to_runner(tmp_path):
    p = _registry(tmp_path, {"qwen3:8b": "ollama"})
    runner = RunnerFactory.create("qwen3:8b", registry_path=p)
    assert runner.model == "qwen3:8b"


def test_factory_passes_extra_kwargs(tmp_path):
    p = _registry(tmp_path, {"claude-sonnet-4-6": "anthropic"})
    runner = RunnerFactory.create("claude-sonnet-4-6", poll_interval=5, registry_path=p)
    assert runner.poll_interval == 5


def test_ollama_runner_is_base_runner(tmp_path):
    p = _registry(tmp_path, {"qwen2.5:3b": "ollama"})
    assert isinstance(RunnerFactory.create("qwen2.5:3b", registry_path=p), BaseRunner)


def test_anthropic_runner_is_base_runner(tmp_path):
    p = _registry(tmp_path, {"claude-haiku-4-5": "anthropic"})
    assert isinstance(RunnerFactory.create("claude-haiku-4-5", registry_path=p), BaseRunner)


def test_factory_creates_openai_runner(tmp_path):
    p = _registry(tmp_path, {"gpt-4o-mini": "openai"})
    assert isinstance(RunnerFactory.create("gpt-4o-mini", registry_path=p), OpenAIRunner)


def test_openai_runner_is_base_runner(tmp_path):
    p = _registry(tmp_path, {"gpt-4o": "openai"})
    assert isinstance(RunnerFactory.create("gpt-4o", registry_path=p), BaseRunner)
