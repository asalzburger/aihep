"""Read hits/clusters/truth/contributions as CSV or Apache Arrow -- the
same four tables `sensor`/`splitting` use, so this package can read any of
their run directories directly.

The actual (de)serialization lives in `clustering_utils.io` (shared with
`sensor`/`tracker`/`splitting`); this module just names the four tables.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from clustering_utils.io import Format, read_tables

__all__ = ["Format", "read_run"]


def read_run(input_dir: str | Path, fmt: Format) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = read_tables(input_dir, fmt, ["hits", "clusters", "truth", "contributions"])
    return tables["hits"], tables["clusters"], tables["truth"], tables["contributions"]
