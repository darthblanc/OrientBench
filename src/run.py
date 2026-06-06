import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CSV orientation experiment")
    parser.add_argument("csv_path", help="Path to the CSV dataset")
    parser.add_argument("--model", required=True, help="Model name (must be registered in models.json)")
    parser.add_argument("--id-col", required=True, dest="id_col", help="Column to use as entity identifier")
    parser.add_argument("--n", type=int, default=20, help="Number of tasks to generate (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--max-rows", type=int, default=15, dest="max_rows", help="Max context rows per task (default: 15)")
    parser.add_argument("--cols", type=str, default=None, help="Comma-separated columns to include (default: all)")
    return parser


if __name__ == "__main__":
    import json
    import logging
    from datetime import datetime
    from pathlib import Path
    from src.runners.factory import RunnerFactory

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = build_parser().parse_args()

    cols = [c.strip() for c in args.cols.split(",")] if args.cols else None
    runner = RunnerFactory.create(model=args.model)
    data = runner.run(args.csv_path, args.id_col, n=args.n, seed=args.seed, max_rows=args.max_rows, cols=cols)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{data['dataset']}_{ts}.json"
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Saved → {out_path}")
