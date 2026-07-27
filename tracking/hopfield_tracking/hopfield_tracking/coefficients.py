"""T_ij: the physics of Hopfield track finding (Denby 1988, section 8).

For two *distinct* segments i=(a,b), j=(c,d) (hit ids), exactly one of the
following holds -- this partition is the crux of the whole method:

- **reverse pair**, `(a,b) == (d,c)`: the same edge traversed both ways ->
  inhibit (a track can't use one physical edge in both directions at once).
- **chain**, `b == c` or `a == d`: one segment's end is the other's start,
  i.e. they can be joined end-to-end -> **type 1** (reinforcing for a
  smooth continuation, inhibiting for a sharp turn -- the sign comes out of
  the coefficient itself, see `type1_coefficient`).
- **compete**, `a == c` or `b == d`: both segments start (or both end) at
  the same point -- they're alternatives, not a chain -> inhibit.
- **no shared point**: candidates for **type 2** (circle-fit consistency),
  if close enough to be worth considering; otherwise `T_ij = 0`.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .network import Segment

PairKind = Literal["reverse", "chain", "compete", "type2"]


def classify_pair(seg_a: Segment, seg_b: Segment) -> PairKind:
    a, b = seg_a.start_id, seg_a.end_id
    c, d = seg_b.start_id, seg_b.end_id
    if a == d and b == c:
        return "reverse"
    if b == c or a == d:
        return "chain"
    if a == c or b == d:
        return "compete"
    return "type2"


def _cos_angle(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)


def type1_coefficient(seg_a: Segment, seg_b: Segment, n: int = 5, r_scale: float = 1.0) -> float:
    """T_ij = cos(theta_ij)^n / (r_ij / r_scale) (paper's formula, with
    r_ij -- the length of the two segments' vector sum -- expressed in units
    of `r_scale`). theta_ij = angle between the two segments' directions.

    `n` odd (paper: n=5) makes this signed: a smooth continuation
    (theta near 0) gives cos^n near +1 over a small r -> strongly
    reinforcing; a sharp reversal (theta near pi) gives cos^n near -1 *and*
    the vectors nearly cancel (r near 0) -> strongly inhibiting.

    `r_scale` matters: the paper's `T_ij` is dimensionful (1/length), so its
    magnitude is only comparable to the (dimensionless) type-2 coefficient
    and the inhibition constant in whatever unit system makes `r_ij ~ O(1)`.
    Pass a characteristic length (e.g. the mean hit spacing, `<r>`) so this
    works regardless of the absolute coordinate units -- without it, this
    coefficient silently shrinks to nothing whenever positions are in
    "large-number" units like pixels, and type 2 + inhibition dominate.
    """
    va, vb = seg_a.vector, seg_b.vector
    cos_theta = _cos_angle(va, vb)
    r = math.hypot(va[0] + vb[0], va[1] + vb[1])
    # floor r at a small fraction of r_scale: two chained segments that
    # nearly reverse on each other (a sharp U-turn) have a vector sum near
    # zero, which would otherwise blow this coefficient up numerically --
    # it should just be a strongly (but finitely) inhibiting value.
    r = max(r, 0.05 * r_scale)
    return (cos_theta**n) * r_scale / r


def fit_circle(points: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) least-squares circle fit through `points` (N, 2).
    Returns (center_x, center_y, radius). Same method used to fit the
    Denby event's tracks in `tracking/denby/python/denby_svg.py`."""
    x, y = points[:, 0], points[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = math.sqrt(max(sol[2] + cx**2 + cy**2, 0.0))
    return cx, cy, r


def circle_fit_chi2(points: np.ndarray) -> float:
    """Normalized residual of the best-fit circle through `points`: ~0 for
    points that lie exactly on a circle, growing as they deviate from one."""
    cx, cy, r = fit_circle(points)
    if r == 0.0:
        return 1.0
    radial = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    return float(np.mean(((radial - r) / r) ** 2))


def type2_coefficient(seg_a: Segment, seg_b: Segment, r_c: float) -> float:
    """T_ij ~ 1 - chi^2 of a circle fit through the two segments' 4
    endpoints -- rewards pairs of segments consistent with lying on a
    common circular arc even though they don't share a point (useful for
    resolving close-together or concentric tracks). Gated by the paper's
    own locality cutoffs: endpoints not too far apart, and the fitted
    radius not implausibly small next to the segments themselves.
    """
    closest_gap = min(
        math.hypot(pa[0] - pb[0], pa[1] - pb[1])
        for pa in (seg_a.start_xy, seg_a.end_xy)
        for pb in (seg_b.start_xy, seg_b.end_xy)
    )
    if closest_gap > 3.0 * r_c:
        return 0.0

    points = np.array([seg_a.start_xy, seg_a.end_xy, seg_b.start_xy, seg_b.end_xy])
    longer = max(seg_a.length, seg_b.length)
    _, _, radius = fit_circle(points)
    if longer == 0.0 or radius < 3.0 * longer:
        return 0.0

    chi2 = circle_fit_chi2(points)
    return max(0.0, 1.0 - chi2)


#: The paper's own type-1 formula gives an interior segment on a straight
#: chain a coefficient of ~0.5 to each of its two neighbors -- exactly
#: enough to be *neutrally* stable (see dynamics.py's module docstring for
#: the full argument), which in practice decays to the trivial all-off
#: solution rather than growing into the correct one. A scale of ~2-3x
#: is what actually makes the correct chain a linearly *growing* solution
#: from a near-zero start, which is what the sigmoid needs to then saturate.
#: (Calibrated together with dynamics.DEFAULT_GAIN and cli.run's default
#: `inhibition`, on the recovered Denby event, once same-layer candidate
#: segments were excluded -- see network.py::build_segments and the README.)
DEFAULT_TYPE1_SCALE = 2.5


def build_weight_matrix(
    segments: list[Segment],
    n: int = 5,
    r_c: float | None = None,
    r_scale: float = 1.0,
    inhibition: float = -1.0,
    type1_scale: float = DEFAULT_TYPE1_SCALE,
    type2_scale: float = 1.0,
    use_type2: bool = True,
) -> np.ndarray:
    """Assemble the full symmetric T_ij matrix for one event. `r_scale`
    should be a characteristic length (e.g. the mean hit spacing) -- see
    `type1_coefficient` for why it matters. See `DEFAULT_TYPE1_SCALE` for
    why `type1_scale` isn't just 1."""
    n_seg = len(segments)
    t = np.zeros((n_seg, n_seg))
    for i in range(n_seg):
        for j in range(i + 1, n_seg):
            kind = classify_pair(segments[i], segments[j])
            if kind in ("reverse", "compete"):
                value = inhibition
            elif kind == "chain":
                value = type1_scale * type1_coefficient(segments[i], segments[j], n=n, r_scale=r_scale)
            else:  # type2
                value = (
                    type2_scale * type2_coefficient(segments[i], segments[j], r_c)
                    if use_type2 and r_c is not None
                    else 0.0
                )
            t[i, j] = value
            t[j, i] = value
    return t
