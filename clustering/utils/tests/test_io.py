import pandas as pd
import pytest

from clustering_utils.io import read_table, read_tables, write_table, write_tables


@pytest.mark.parametrize("fmt", ["csv", "arrow"])
def test_single_table_round_trip(tmp_path, fmt):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]})
    path = tmp_path / f"table.{fmt}"
    write_table(df, path, fmt)
    reloaded = read_table(path, fmt)
    pd.testing.assert_frame_equal(df, reloaded, check_dtype=False)


def test_write_table_creates_parent_directories(tmp_path):
    df = pd.DataFrame({"a": [1]})
    path = tmp_path / "nested" / "dir" / "table.csv"
    write_table(df, path, "csv")
    assert path.exists()


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        write_table(pd.DataFrame({"a": [1]}), "x", "json")
    with pytest.raises(ValueError):
        read_table("x", "json")


@pytest.mark.parametrize("fmt", ["csv", "arrow"])
def test_write_tables_and_read_tables_round_trip(tmp_path, fmt):
    hits = pd.DataFrame({"x": [1, 2]})
    clusters = pd.DataFrame({"n": [1]})
    paths = write_tables(tmp_path, fmt, {"hits": hits, "clusters": clusters})
    assert set(paths) == {"hits", "clusters"}
    assert all(p.exists() for p in paths.values())

    tables = read_tables(tmp_path, fmt, ["hits", "clusters"])
    pd.testing.assert_frame_equal(hits, tables["hits"], check_dtype=False)
    pd.testing.assert_frame_equal(clusters, tables["clusters"], check_dtype=False)
