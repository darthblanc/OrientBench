import pytest
from src.score import score, score_row_list


def test_exact_match():
    assert score("Alice", "Alice") is True


def test_case_insensitive():
    assert score("alice", "Alice") is True


def test_strips_whitespace():
    assert score("  30  ", "30") is True


def test_strips_quotes():
    assert score('"NYC"', "NYC") is True


def test_wrong_answer():
    assert score("Bob", "Alice") is False


def test_numeric_string_match():
    assert score("88", "88") is True


def test_numeric_float_vs_int():
    assert score("88.0", "88") is True


def test_row_list_all_present():
    answer = "age=30, city=NYC, score=88"
    predicted = "age=30, city=NYC, score=88"
    assert score_row_list(predicted, answer) is True


def test_row_list_partial_missing():
    answer = "age=30, city=NYC, score=88"
    predicted = "age=30, city=NYC"
    assert score_row_list(predicted, answer) is False


def test_row_list_order_independent():
    answer = "age=30, city=NYC, score=88"
    predicted = "score=88, age=30, city=NYC"
    assert score_row_list(predicted, answer) is True


def test_row_list_newline_separated():
    answer = "age=30, city=NYC, score=88"
    predicted = "age=30\ncity=NYC\nscore=88"
    assert score_row_list(predicted, answer) is True


def test_row_list_extra_pairs_ok():
    # model includes id_col which isn't in ground truth — should still pass
    answer = "age=30, city=NYC"
    predicted = "id=1, age=30, city=NYC"
    assert score_row_list(predicted, answer) is True


def test_row_list_newline_missing_pair():
    answer = "age=30, city=NYC, score=88"
    predicted = "age=30\ncity=NYC"
    assert score_row_list(predicted, answer) is False
