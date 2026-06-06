import json
import sys
from collections import defaultdict
from pathlib import Path


def _pct(correct: int, total: int) -> str:
    if total == 0:
        return "  n/a"
    return f"{correct/total*100:5.1f}%"


def report(result_paths: list[str]) -> None:
    all_rows = []

    for path in result_paths:
        data = json.loads(Path(path).read_text())
        dataset = data["dataset"]
        model = data["model"]

        by_kind: dict[str, dict] = defaultdict(lambda: {"row": [0, 0], "col": [0, 0]})

        for r in data["results"]:
            kind = r["task_kind"]
            by_kind[kind]["row"][0] += int(r["row_correct"])
            by_kind[kind]["row"][1] += 1
            by_kind[kind]["col"][0] += int(r["col_correct"])
            by_kind[kind]["col"][1] += 1

        for kind, counts in by_kind.items():
            row_n, col_n = counts["row"][1], counts["col"][1]
            row_acc = counts["row"][0] / row_n if row_n else 0
            col_acc = counts["col"][0] / col_n if col_n else 0
            delta = col_acc - row_acc
            all_rows.append((dataset, model, kind, row_acc, col_acc, delta, row_n))

    totals: dict[str, dict] = defaultdict(lambda: {"row": [0, 0], "col": [0, 0]})
    for path in result_paths:
        data = json.loads(Path(path).read_text())
        for r in data["results"]:
            totals[data["dataset"]]["row"][0] += int(r["row_correct"])
            totals[data["dataset"]]["row"][1] += 1
            totals[data["dataset"]]["col"][0] += int(r["col_correct"])
            totals[data["dataset"]]["col"][1] += 1

    w = 14
    header = f"{'dataset':<{w}} {'model':<14} {'task_type':<14} {'row_acc':>8} {'col_acc':>8} {'delta':>8} {'n':>4}"
    print(header)
    print("-" * len(header))

    for dataset, model, kind, row_acc, col_acc, delta, n in sorted(all_rows):
        delta_str = f"{delta:+.1%}"
        print(f"{dataset:<{w}} {model:<14} {kind:<14} {row_acc:>7.1%}  {col_acc:>7.1%}  {delta_str:>8}  {n:>3}")

    print("-" * len(header))
    for ds, counts in sorted(totals.items()):
        rn = counts["row"][1]
        cn = counts["col"][1]
        ra = counts["row"][0] / rn if rn else 0
        ca = counts["col"][0] / cn if cn else 0
        d = ca - ra
        print(f"{ds:<{w}} {'(all)':<14} {'TOTAL':<14} {ra:>7.1%}  {ca:>7.1%}  {d:+.1%}  {rn:>3}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m src.report results/*.json")
        sys.exit(1)
    report(sys.argv[1:])
