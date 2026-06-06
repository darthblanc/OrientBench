import pandas as pd
import pytest
from src.orient import to_row_wise, to_col_wise


@pytest.fixture
def df():
    return pd.DataFrame({
        "name": ["Alice", "Bob"],
        "age": [30, 25],
        "city": ["NYC", "LA"],
    })


def test_row_wise_has_header_first_line(df):
    result = to_row_wise(df)
    lines = result.strip().splitlines()
    assert lines[0] == "name,age,city"


def test_row_wise_data_rows(df):
    result = to_row_wise(df)
    lines = result.strip().splitlines()
    assert lines[1] == "Alice,30,NYC"
    assert lines[2] == "Bob,25,LA"


def test_row_wise_line_count(df):
    result = to_row_wise(df)
    lines = result.strip().splitlines()
    assert len(lines) == 3  # 1 header + 2 data rows


def test_col_wise_each_row_is_field(df):
    result = to_col_wise(df)
    lines = result.strip().splitlines()
    assert lines[0].startswith("name,")
    assert lines[1].startswith("age,")
    assert lines[2].startswith("city,")


def test_col_wise_values_on_field_row(df):
    result = to_col_wise(df)
    lines = result.strip().splitlines()
    assert lines[0] == "name,Alice,Bob"
    assert lines[1] == "age,30,25"
    assert lines[2] == "city,NYC,LA"


def test_col_wise_line_count(df):
    result = to_col_wise(df)
    lines = result.strip().splitlines()
    assert len(lines) == 3  # one line per column/field


def test_roundtrip_preserves_values(df):
    row = to_row_wise(df)
    col = to_col_wise(df)
    # Both orientations must contain all original values
    for val in ["Alice", "Bob", "30", "25", "NYC", "LA"]:
        assert val in row
        assert val in col
