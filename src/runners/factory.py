import json
from pathlib import Path

from src.runners.base import BaseRunner

MODELS_PATH = Path(__file__).parents[2] / "models.json"

_PROVIDER_MAP: dict[str, type] = {}


class RunnerFactory:

    @staticmethod
    def create(model: str, registry_path: Path = MODELS_PATH, **kwargs) -> BaseRunner:
        registry = json.loads(registry_path.read_text())
        if model not in registry:
            raise ValueError(
                f"Model '{model}' not found in models.json. "
                f"Run: python -m src.registry add {model} <provider>"
            )
        provider = registry[model]
        if provider not in _PROVIDER_MAP:
            raise ValueError(f"Unknown provider '{provider}' for model '{model}'.")
        return _PROVIDER_MAP[provider](model=model, **kwargs)


from src.runners.ollama_runner import OllamaRunner
from src.runners.anthropic_runner import AnthropicRunner
from src.runners.openai_runner import OpenAIRunner

_PROVIDER_MAP["ollama"] = OllamaRunner
_PROVIDER_MAP["anthropic"] = AnthropicRunner
_PROVIDER_MAP["openai"] = OpenAIRunner
