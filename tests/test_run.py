import pytest
from src.run import build_parser


def test_parser_requires_model():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["data/foo.csv", "--id-col", "name"])


def test_parser_requires_id_col():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["data/foo.csv", "--model", "qwen2.5:3b"])


def test_parser_accepts_all_args():
    parser = build_parser()
    args = parser.parse_args([
        "data/foo.csv",
        "--model", "qwen2.5:3b",
        "--id-col", "name",
        "--n", "50",
        "--seed", "7",
    ])
    assert args.csv_path == "data/foo.csv"
    assert args.model == "qwen2.5:3b"
    assert args.id_col == "name"
    assert args.n == 50
    assert args.seed == 7


def test_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["data/foo.csv", "--model", "qwen2.5:3b", "--id-col", "name"])
    assert args.n == 20
    assert args.seed == 42
    assert args.max_rows == 15
    assert args.cols is None


def test_parser_accepts_max_rows():
    parser = build_parser()
    args = parser.parse_args([
        "data/foo.csv", "--model", "qwen2.5:3b", "--id-col", "name", "--max-rows", "5",
    ])
    assert args.max_rows == 5


def test_parser_accepts_cols():
    parser = build_parser()
    args = parser.parse_args([
        "data/foo.csv", "--model", "qwen2.5:3b", "--id-col", "name", "--cols", "age,city",
    ])
    assert args.cols == "age,city"
