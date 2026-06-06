import pandas as pd
import pytest
from src.tasks import Task, generate_tasks

FIXTURE = pd.DataFrame({
    "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "age":  [30, 25, 35, 28, 31],
    "city": ["NYC", "LA", "Chicago", "NYC", "LA"],
    "score": [88, 72, 91, 65, 79],
})


def test_generate_tasks_returns_list():
    tasks = generate_tasks(FIXTURE, id_col="name", n=10, seed=42)
    assert isinstance(tasks, list)
    assert len(tasks) == 10


def test_task_has_required_fields():
    tasks = generate_tasks(FIXTURE, id_col="name", n=4, seed=0)
    for t in tasks:
        assert isinstance(t, Task)
        assert t.kind in ("cell_recall", "attr_scan", "comparison", "row_list")
        assert isinstance(t.question, str) and t.question
        assert isinstance(t.answer, str) and t.answer
        assert isinstance(t.df_slice, pd.DataFrame)
        assert len(t.df_slice) > 0


def test_cell_recall_answer_is_correct():
    tasks = generate_tasks(FIXTURE, id_col="name", n=40, seed=7)
    recalls = [t for t in tasks if t.kind == "cell_recall"]
    assert recalls, "expected at least one cell_recall task"
    for t in recalls:
        # answer must appear somewhere in the slice
        assert any(str(t.answer) in str(v) for col in t.df_slice.columns for v in t.df_slice[col].astype(str))


def test_comparison_answer_is_max_id():
    tasks = generate_tasks(FIXTURE, id_col="name", n=40, seed=3)
    comparisons = [t for t in tasks if t.kind == "comparison"]
    assert comparisons, "expected at least one comparison task"
    for t in comparisons:
        # The answer should be the name of the entity with the highest numeric value
        assert t.answer in t.df_slice["name"].values


def test_slice_max_15_rows():
    big = pd.concat([FIXTURE] * 10, ignore_index=True)
    tasks = generate_tasks(big, id_col="name", n=10, seed=1)
    for t in tasks:
        assert len(t.df_slice) <= 15


def test_max_rows_parameter():
    big = pd.concat([FIXTURE] * 10, ignore_index=True)
    tasks = generate_tasks(big, id_col="name", n=10, seed=1, max_rows=3)
    for t in tasks:
        assert len(t.df_slice) <= 3


def test_cols_filters_columns():
    tasks = generate_tasks(FIXTURE, id_col="name", n=8, seed=0, cols=["age", "city"])
    for t in tasks:
        assert set(t.df_slice.columns) == {"name", "age", "city"}


def test_cols_excludes_unlisted_columns():
    tasks = generate_tasks(FIXTURE, id_col="name", n=8, seed=0, cols=["score"])
    for t in tasks:
        assert "age" not in t.df_slice.columns
        assert "city" not in t.df_slice.columns


def test_cols_none_keeps_all_columns():
    tasks = generate_tasks(FIXTURE, id_col="name", n=4, seed=0, cols=None)
    for t in tasks:
        assert set(t.df_slice.columns) == set(FIXTURE.columns)


def test_attr_scan_answer_exists_in_slice():
    tasks = generate_tasks(FIXTURE, id_col="name", n=40, seed=5)
    scans = [t for t in tasks if t.kind == "attr_scan"]
    assert scans, "expected at least one attr_scan task"
    for t in scans:
        assert t.answer in t.df_slice["name"].values


def test_row_list_answer_contains_all_attributes():
    tasks = generate_tasks(FIXTURE, id_col="name", n=40, seed=9)
    row_lists = [t for t in tasks if t.kind == "row_list"]
    assert row_lists, "expected at least one row_list task"
    for t in row_lists:
        for col in FIXTURE.columns:
            if col != "name":
                assert col in t.answer
