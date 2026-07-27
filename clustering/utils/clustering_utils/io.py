"""Generic CSV/Apache Arrow table read/write, shared by every `clustering/`
package (`sensor`, `tracker`). Each of those keeps its own thin `io.py`
wrapper with a domain-specific `write_run`/`read_run` (named tables, typed
tuple return) built on the generic helpers here -- the actual
serialization logic isn't duplicated anymore.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

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


def write_tables(output_dir: str | Path, fmt: Format, tables: dict[str, pd.DataFrame]) -> dict[str, Path]:
    """Write each named table to `output_dir/{name}.{ext}`. Returns the paths written to."""
    output_dir = Path(output_dir)
    ext = EXTENSIONS[fmt]
    paths = {name: output_dir / f"{name}.{ext}" for name in tables}
    for name, df in tables.items():
        write_table(df, paths[name], fmt)
    return paths


def read_tables(output_dir: str | Path, fmt: Format, names: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Read each named table back from `output_dir/{name}.{ext}`."""
    output_dir = Path(output_dir)
    ext = EXTENSIONS[fmt]
    return {name: read_table(output_dir / f"{name}.{ext}", fmt) for name in names}
