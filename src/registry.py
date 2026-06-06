import json
from pathlib import Path

MODELS_PATH = Path(__file__).parent.parent / "models.json"


def list_models(registry_path: Path = MODELS_PATH) -> dict:
    return json.loads(registry_path.read_text())


def add_model(model: str, provider: str, registry_path: Path = MODELS_PATH) -> None:
    registry = list_models(registry_path)
    registry[model] = provider
    registry_path.write_text(json.dumps(registry, indent=2))


def remove_model(model: str, registry_path: Path = MODELS_PATH) -> None:
    registry = list_models(registry_path)
    if model not in registry:
        raise KeyError(model)
    del registry[model]
    registry_path.write_text(json.dumps(registry, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage the models.json registry")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all registered models")

    p_add = sub.add_parser("add", help="Add or update a model entry")
    p_add.add_argument("model", help="Model name (e.g. claude-sonnet-4-6)")
    p_add.add_argument("provider", help="Provider name (ollama or anthropic)")

    p_rm = sub.add_parser("remove", help="Remove a model entry")
    p_rm.add_argument("model", help="Model name to remove")

    args = parser.parse_args()

    if args.command == "list":
        registry = list_models()
        if not registry:
            print("(empty registry)")
        else:
            for model, provider in sorted(registry.items()):
                print(f"  {model:40s} {provider}")

    elif args.command == "add":
        add_model(args.model, args.provider)
        print(f"Added: {args.model} → {args.provider}")

    elif args.command == "remove":
        try:
            remove_model(args.model)
            print(f"Removed: {args.model}")
        except KeyError:
            print(f"Error: '{args.model}' not found in registry")
            raise SystemExit(1)
