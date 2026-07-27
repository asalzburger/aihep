"""Read/write digitized hits and clusters tables as CSV or Apache Arrow.

The actual (de)serialization lives in `clustering_utils.io` (shared with
`clustering/sensor`); this module just names tracker's two tables.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from clustering_utils.io import Format, read_table, read_tables, write_table, write_tables

__all__ = ["Format", "read_run", "read_table", "write_run", "write_table"]


def write_run(
    output_dir: str | Path, fmt: Format, hits: pd.DataFrame, clusters: pd.DataFrame
) -> dict[str, Path]:
    return write_tables(output_dir, fmt, {"hits": hits, "clusters": clusters})


def read_run(output_dir: str | Path, fmt: Format) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = read_tables(output_dir, fmt, ["hits", "clusters"])
    return tables["hits"], tables["clusters"]
