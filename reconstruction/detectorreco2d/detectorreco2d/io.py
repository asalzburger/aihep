"""Read/write tracks and clusters tables as CSV or Apache Arrow -- same
two-format convention (and the same tiny wrapper shape) as
`detectorsim2d.io`, kept independent rather than imported so this package
never has to agree with `detectorsim2d` on a shared IO module."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from .edm import CLUSTERS_COLUMNS, TRACKS_COLUMNS

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
    output_dir: str | Path, fmt: Format, tracks: pd.DataFrame, clusters: pd.DataFrame | None = None
) -> dict[str, Path]:
    """Write a reconstructed run's tables. ``clusters`` is optional -- a
    tracker-only run (no calorimeter deposits upstream) has none, and
    omitting it writes no clusters file at all."""
    output_dir = Path(output_dir)
    paths = {"tracks": output_dir / f"tracks.{EXTENSIONS[fmt]}"}
    write_table(tracks, paths["tracks"], fmt)
    if clusters is not None:
        paths["clusters"] = output_dir / f"clusters.{EXTENSIONS[fmt]}"
        write_table(clusters, paths["clusters"], fmt)
    return paths


def read_tracks(output_dir: str | Path, fmt: Format) -> pd.DataFrame:
    return read_table(Path(output_dir) / f"tracks.{EXTENSIONS[fmt]}", fmt)


def read_clusters(output_dir: str | Path, fmt: Format) -> pd.DataFrame:
    """The clusters of a reconstructed run, or an empty table with the right
    columns if the run had none."""
    path = Path(output_dir) / f"clusters.{EXTENSIONS[fmt]}"
    if not path.exists():
        return pd.DataFrame(columns=CLUSTERS_COLUMNS)
    return read_table(path, fmt)


def read_run(output_dir: str | Path, fmt: Format) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(tracks, clusters)``, mirroring `detectorsim2d.io.read_run`."""
    return read_tracks(output_dir, fmt), read_clusters(output_dir, fmt)


__all__ = [
    "TRACKS_COLUMNS",
    "CLUSTERS_COLUMNS",
    "Format",
    "read_clusters",
    "read_run",
    "read_table",
    "read_tracks",
    "write_run",
    "write_table",
]
