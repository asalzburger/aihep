"""Read/write hits, clusters, truth, and contributions as CSV or Apache
Arrow -- the same four tables and layout `sensor.io` uses, so a split run
directory is a drop-in replacement for a `sensor` run directory.

The actual (de)serialization lives in `clustering_utils.io` (shared with
`sensor`/`tracker`); this module just names this package's four tables.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from clustering_utils.io import Format, read_table, read_tables, write_table, write_tables

__all__ = ["Format", "read_run", "read_table", "write_run", "write_table"]


def write_run(
    output_dir: str | Path,
    fmt: Format,
    hits: pd.DataFrame,
    clusters: pd.DataFrame,
    truth: pd.DataFrame,
    contributions: pd.DataFrame,
) -> dict[str, Path]:
    return write_tables(
        output_dir, fmt, {"hits": hits, "clusters": clusters, "truth": truth, "contributions": contributions}
    )


def read_run(output_dir: str | Path, fmt: Format) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = read_tables(output_dir, fmt, ["hits", "clusters", "truth", "contributions"])
    return tables["hits"], tables["clusters"], tables["truth"], tables["contributions"]
