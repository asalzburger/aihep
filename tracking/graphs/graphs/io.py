"""Read/write an edges table as CSV or Apache Arrow.

A graph's nodes are just a `detectorsim2d` hits table (see `graphs.edm`), so
they already round-trip through `detectorsim2d.io`; this module only adds the
edges half, reusing `detectorsim2d.io.write_table`/`read_table` (both fully
generic over any `pandas.DataFrame`) rather than re-implementing CSV/Arrow
IO.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from detectorsim2d.io import EXTENSIONS, Format, read_table, write_table

from .edm import TrackGraph


def write_edges(path: str | Path, edges: pd.DataFrame, fmt: Format) -> None:
    write_table(edges, path, fmt)


def read_edges(path: str | Path, fmt: Format) -> pd.DataFrame:
    return read_table(path, fmt)


def write_graph(output_dir: str | Path, fmt: Format, graph: TrackGraph) -> dict[str, Path]:
    """Write ``graph.edges`` into ``output_dir`` (``graph.nodes`` is a
    `detectorsim2d` hits table -- write it separately via
    `detectorsim2d.io.write_run`/`write_table` if it isn't already on disk)."""
    path = Path(output_dir) / f"edges.{EXTENSIONS[fmt]}"
    write_edges(path, graph.edges, fmt)
    return {"edges": path}


def read_graph(output_dir: str | Path, fmt: Format, nodes: pd.DataFrame) -> TrackGraph:
    """Read the edges table `write_graph` wrote back out, paired with
    ``nodes`` (the hits table it was built from, read separately -- e.g. via
    `detectorsim2d.io.read_run`)."""
    edges = read_edges(Path(output_dir) / f"edges.{EXTENSIONS[fmt]}", fmt)
    return TrackGraph(nodes=nodes, edges=edges)
