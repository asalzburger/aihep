"""Candidate track segments -- the Hopfield network's "neurons" (Denby 1988,
section 8): "a neuron (i,j) represents the directed segment from point i to
point j". A valid track is a non-bifurcating chain of "on" segments, so we
only ever generate segments between *nearby* points (the paper's R_c
locality cutoff) -- most point pairs aren't plausible track segments and
would just bloat the network for no benefit.

This module only knows about plain (x, y) points -- no detector, no layers
-- so it works on any 2D hit set, Denby's or otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Segment:
    """A candidate directed track segment (one Hopfield neuron) from hit
    `start_id` to hit `end_id`. Both directions between a pair of nearby
    points are generated as separate segments (see `build_segments`) since
    nothing here assumes tracks come from a common, known vertex -- exactly
    the generality the paper claims ("tracks do not need to come from the
    origin, nor do they need to be helices")."""

    index: int
    start_id: int
    end_id: int
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]

    @property
    def vector(self) -> tuple[float, float]:
        return (self.end_xy[0] - self.start_xy[0], self.end_xy[1] - self.start_xy[1])

    @property
    def length(self) -> float:
        dx, dy = self.vector
        return math.hypot(dx, dy)


def build_segments(x: np.ndarray, y: np.ndarray, r_c: float, layer_ids: np.ndarray | None = None) -> list[Segment]:
    """Every ordered pair of distinct points within `r_c` of each other,
    i.e. every plausible candidate segment in *both* directions. O(n^2),
    which is fine for the point counts a didactic/demo event has (a few
    hundred at most) -- no spatial index needed.

    `layer_ids`, if given, excludes any pair of points that share a layer
    id. Two hits recorded by the same detector plane can never be the two
    ends of one physical track segment -- a real particle produces at most
    one hit per plane -- so such a pair is never a valid candidate
    regardless of how close together it happens to be in (x, y); without
    this, points from different tracks that happen to land near each other
    on the same layer (common right where several tracks fan out from one
    vertex) generate spurious same-layer segments that have no physical
    meaning and only add noise for the dynamics to (usually, not always)
    suppress on its own.
    """
    n = len(x)
    segments: list[Segment] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if layer_ids is not None and layer_ids[i] == layer_ids[j]:
                continue
            dx, dy = x[j] - x[i], y[j] - y[i]
            dist = math.hypot(dx, dy)
            if 0.0 < dist <= r_c:
                segments.append(
                    Segment(
                        index=len(segments),
                        start_id=i,
                        end_id=j,
                        start_xy=(float(x[i]), float(y[i])),
                        end_xy=(float(x[j]), float(y[j])),
                    )
                )
    return segments


def mean_consecutive_hit_distance(hits: pd.DataFrame, group_col: str = "particle_id") -> float:
    """The paper's <r>: "the mean distance between adjacent points on the
    same track". Used only to *calibrate* R_c ahead of time from ground
    truth -- never fed into the segment-building or relaxation themselves,
    which only ever see (x, y). This is the same simplification the paper's
    own tests made (their test events were simulated too); a real detector
    would calibrate R_c from known hit-spacing statistics instead.

    Points within each group are ordered by projecting onto the group's own
    principal direction (via SVD), so this works regardless of how the
    points happen to be indexed or which detector layer they came from.
    """
    distances: list[float] = []
    for _, group in hits.groupby(group_col):
        points = group[["x", "y"]].to_numpy()
        if len(points) < 2:
            continue
        centered = points - points.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        order = np.argsort(centered @ vt[0])
        ordered = points[order]
        steps = np.diff(ordered, axis=0)
        distances.extend(np.hypot(steps[:, 0], steps[:, 1]))
    if not distances:
        raise ValueError("no group in `hits` has >= 2 points; cannot estimate <r>")
    return float(np.mean(distances))
