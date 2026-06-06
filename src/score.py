import re


def _normalize(s: str) -> str:
    s = s.strip().lower().strip('"\'')
    # normalize 88.0 → 88 for numeric strings
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return s


def score(predicted: str, ground_truth: str) -> bool:
    return _normalize(predicted) == _normalize(ground_truth)


def score_row_list(predicted: str, ground_truth: str) -> bool:
    def parse_pairs(s: str) -> set[tuple[str, str]]:
        pairs = set()
        for part in re.split(r"[,\n]+\s*", s):
            if "=" in part:
                k, _, v = part.partition("=")
                pairs.add((_normalize(k), _normalize(v)))
        return pairs

    expected = parse_pairs(ground_truth)
    got = parse_pairs(predicted)
    return expected.issubset(got)
