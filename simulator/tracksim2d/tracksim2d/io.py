"""Read/write particles and hits tables as CSV or Apache Arrow."""

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
    output_dir: str | Path, fmt: Format, particles: pd.DataFrame, hits: pd.DataFrame
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    ext = EXTENSIONS[fmt]
    paths = {
        "particles": output_dir / f"particles.{ext}",
        "hits": output_dir / f"hits.{ext}",
    }
    write_table(particles, paths["particles"], fmt)
    write_table(hits, paths["hits"], fmt)
    return paths


def read_run(output_dir: str | Path, fmt: Format) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    ext = EXTENSIONS[fmt]
    particles = read_table(output_dir / f"particles.{ext}", fmt)
    hits = read_table(output_dir / f"hits.{ext}", fmt)
    return particles, hits
