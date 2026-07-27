"""Read/write hits, clusters, and truth tables as CSV or Apache Arrow."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

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
    hits: pd.DataFrame,
    clusters: pd.DataFrame,
    truth: pd.DataFrame,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    ext = EXTENSIONS[fmt]
    paths = {
        "hits": output_dir / f"hits.{ext}",
        "clusters": output_dir / f"clusters.{ext}",
        "truth": output_dir / f"truth.{ext}",
    }
    write_table(hits, paths["hits"], fmt)
    write_table(clusters, paths["clusters"], fmt)
    write_table(truth, paths["truth"], fmt)
    return paths


def read_run(output_dir: str | Path, fmt: Format) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    ext = EXTENSIONS[fmt]
    hits = read_table(output_dir / f"hits.{ext}", fmt)
    clusters = read_table(output_dir / f"clusters.{ext}", fmt)
    truth = read_table(output_dir / f"truth.{ext}", fmt)
    return hits, clusters, truth
