"""Build a candidate track graph (nodes = hits, edges = candidate hit-to-hit
connections) from a `detectorsim2d` hits table, according to a chosen
`~graphs.prescription.Prescription`.

Works on hits from *any* `detector2d` layout -- a flat plane stack (like the
Denby example, `tracking/denby`) or a concentric barrel (like
`simulator/detectorsim2d/configs/barrel6.yaml`) -- without caring which: only
`x`, `y`, `layer_id`, and `event_id` from the hits table are ever used,
never the layer geometry itself.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from .edm import EDGES_COLUMNS, TrackGraph
from .prescription import ConnectionRules, FullyConnected, Prescription, Regional


def _wrap_phi(delta: float) -> float:
    """Wrap a phi difference into ``(-pi, pi]`` -- phi is cyclic, so a raw
    subtraction is wrong right across the +-pi seam."""
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def _in_range(value: float, bounds: tuple[float, float] | None) -> bool:
    if bounds is None:
        return True
    lo, hi = bounds
    return lo <= value <= hi


def _edge_row(event_id, edge_id: int, src: pd.Series, dst: pd.Series) -> dict:
    src_phi = math.atan2(src["y"], src["x"])
    dst_phi = math.atan2(dst["y"], dst["x"])
    dx = dst["x"] - src["x"]
    dy = dst["y"] - src["y"]
    return dict(
        event_id=event_id,
        edge_id=edge_id,
        src_hit_id=int(src["hit_id"]),
        dst_hit_id=int(dst["hit_id"]),
        delta_layer_id=dst["layer_id"] - src["layer_id"],
        delta_r=math.hypot(dst["x"], dst["y"]) - math.hypot(src["x"], src["y"]),
        delta_phi=_wrap_phi(dst_phi - src_phi),
        delta_x=dx,
        delta_y=dy,
        distance=math.hypot(dx, dy),
    )


def _fully_connected_pairs(n: int, directed: bool) -> list[tuple[int, int]]:
    if directed:
        return list(itertools.permutations(range(n), 2))
    return list(itertools.combinations(range(n), 2))


def _regional_pairs(event_hits: pd.DataFrame, phi_width: float) -> list[tuple[int, int]]:
    phi = np.arctan2(event_hits["y"].to_numpy(), event_hits["x"].to_numpy())
    region = np.floor((phi + math.pi) / phi_width).astype(int)
    buckets: dict[int, list[int]] = {}
    for idx, r in enumerate(region):
        buckets.setdefault(int(r), []).append(idx)
    pairs: list[tuple[int, int]] = []
    for idx_list in buckets.values():
        pairs.extend(itertools.combinations(idx_list, 2))
    return pairs


def _connection_rule_pairs(event_hits: pd.DataFrame, rules: ConnectionRules) -> list[tuple[int, int]]:
    n = len(event_hits)
    x = event_hits["x"].to_numpy()
    y = event_hits["y"].to_numpy()
    layer_id = event_hits["layer_id"].to_numpy()
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)

    pairs: list[tuple[int, int]] = []
    for i, j in itertools.permutations(range(n), 2):
        if rules.delta_layer_id is not None and not _in_range(layer_id[j] - layer_id[i], rules.delta_layer_id):
            continue
        if rules.delta_r is not None and not _in_range(r[j] - r[i], rules.delta_r):
            continue
        if rules.delta_x is not None and not _in_range(x[j] - x[i], rules.delta_x):
            continue
        if rules.delta_phi is not None and not _in_range(_wrap_phi(phi[j] - phi[i]), rules.delta_phi):
            continue
        pairs.append((i, j))
    return pairs


def _candidate_pairs(event_hits: pd.DataFrame, prescription: Prescription) -> list[tuple[int, int]]:
    if isinstance(prescription, FullyConnected):
        return _fully_connected_pairs(len(event_hits), prescription.directed)
    if isinstance(prescription, Regional):
        return _regional_pairs(event_hits, prescription.phi_width)
    if isinstance(prescription, ConnectionRules):
        return _connection_rule_pairs(event_hits, rules=prescription)
    raise TypeError(f"Unknown prescription type: {type(prescription)!r}")


def build_edges(hits: pd.DataFrame, prescription: Prescription) -> pd.DataFrame:
    """Build just the edges table for ``hits`` (a `detectorsim2d` hits table,
    one or more events) under ``prescription``. Edges are only ever formed
    within a single event -- hits from different events never connect."""
    rows: list[dict] = []
    edge_id = 0
    for event_id, event_hits in hits.groupby("event_id", sort=True):
        event_hits = event_hits.reset_index(drop=True)
        for i, j in _candidate_pairs(event_hits, prescription):
            rows.append(_edge_row(event_id, edge_id, event_hits.iloc[i], event_hits.iloc[j]))
            edge_id += 1
    return pd.DataFrame(rows, columns=EDGES_COLUMNS)


def build_graph(hits: pd.DataFrame, prescription: Prescription) -> TrackGraph:
    """Build the full `~graphs.edm.TrackGraph` (nodes = ``hits`` unchanged,
    edges = `build_edges`) for ``hits`` under ``prescription``."""
    return TrackGraph(nodes=hits, edges=build_edges(hits, prescription))
