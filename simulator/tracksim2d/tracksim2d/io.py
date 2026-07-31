"""Read/write particles, hits and deposits tables as CSV or Apache Arrow.

``write_run``/``read_run`` deliberately stay a two-table (particles, hits)
pair: they predate calorimetry and downstream packages (`tracking/graphs`)
unpack exactly two values from them. Deposits are written when passed and read
back through the separate :func:`read_deposits`, so adding a calorimeter never
changed anyone's call site."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from .edm import DEPOSITS_COLUMNS

Format = Literal["csv", "arrow"]

EXTENSIONS = {"csv": "csv", "arrow": "arrow"}


def write_table(df: pd.DataFrame, path: str | Path, fmt: Format) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "arrow":
        table = pa.Table.from_pandas(df, preserve_index=False)
        with ipc.new_file(path, table.schema) as writer:
            writer.write_table(table)
    else:
        raise ValueError(f"Unknown format: {fmt!r}")


def read_table(path: str | Path, fmt: Format) -> pd.DataFrame:
    path = Path(path)
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "arrow":
        with ipc.open_file(path) as reader:
            return reader.read_pandas()
    raise ValueError(f"Unknown format: {fmt!r}")


def write_run(
    output_dir: str | Path,
    fmt: Format,
    particles: pd.DataFrame,
    hits: pd.DataFrame,
    deposits: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write a run's tables. ``deposits`` is optional -- a tracker-only run has
    none, and omitting it writes no deposits file at all."""
    output_dir = Path(output_dir)
    ext = EXTENSIONS[fmt]
    paths = {
        "particles": output_dir / f"particles.{ext}",
        "hits": output_dir / f"hits.{ext}",
    }
    write_table(particles, paths["particles"], fmt)
    write_table(hits, paths["hits"], fmt)
    if deposits is not None:
        paths["deposits"] = output_dir / f"deposits.{ext}"
        write_table(deposits, paths["deposits"], fmt)
    return paths


def read_run(output_dir: str | Path, fmt: Format) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(particles, hits)``. See :func:`read_deposits` for the third table."""
    output_dir = Path(output_dir)
    ext = EXTENSIONS[fmt]
    particles = read_table(output_dir / f"particles.{ext}", fmt)
    hits = read_table(output_dir / f"hits.{ext}", fmt)
    return particles, hits


def read_deposits(output_dir: str | Path, fmt: Format) -> pd.DataFrame:
    """The calorimeter deposits of a run, or an empty table with the right
    columns if the run was tracker-only."""
    path = Path(output_dir) / f"deposits.{EXTENSIONS[fmt]}"
    if not path.exists():
        return pd.DataFrame(columns=DEPOSITS_COLUMNS)
    return read_table(path, fmt)
