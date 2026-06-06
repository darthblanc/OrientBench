import pandas as pd
import pytest
from src.tasks import Task
from src.prompt import build_prompt

SLICE = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})

TASK = Task(
    kind="cell_recall",
    question="What is the age of Alice?",
    answer="30",
    df_slice=SLICE,
)


def test_prompt_contains_question():
    p = build_prompt(TASK, "row")
    assert "What is the age of Alice?" in p


def test_prompt_contains_csv_data():
    p = build_prompt(TASK, "row")
    assert "Alice" in p
    assert "30" in p


def test_row_orientation_has_header_row():
    p = build_prompt(TASK, "row")
    assert "name,age" in p


def test_col_orientation_has_field_rows():
    p = build_prompt(TASK, "col")
    assert "name,Alice,Bob" in p
    assert "age,30,25" in p


def test_prompt_ends_with_answer_instruction():
    for orient in ("row", "col"):
        p = build_prompt(TASK, orient)
        assert "exact answer" in p.lower() or "only" in p.lower()


def test_no_think_tag_in_prompt():
    for orient in ("row", "col"):
        p = build_prompt(TASK, orient)
        assert "/think" not in p
        assert "<think>" not in p
