"""Core geometry: turning a particle track into charge deposited per pixel.

See the plan / project notes for the derivation. Short version: a particle
track ``x(z) = x0 + z*dxdz``, ``y(z) = y0 + z*dydz`` through a slab of
thickness ``t``, combined with a constant Lorentz drift in x applied to the
drifting charge, collects at the readout plane along the straight segment

    P0 = (x0 + t*lorentz_slope, y0)   at z=0 (entry face, fully drifted)
    P1 = (x0 + t*dxdz, y0 + t*dydz)   at z=t (readout face, no drift left)

Charge is deposited uniformly in z, which (since both P0->P1 and z are
affine) is the same as uniform along the P0->P1 segment's arc length.
"""

from __future__ import annotations

import math
from collections import defaultdict


def charge_endpoints(
    x0: float,
    y0: float,
    dxdz: float,
    dydz: float,
    thickness_um: float,
    lorentz_slope: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return (P0, P1), the two ends of the charge-collection segment."""
    p0 = (x0 + thickness_um * lorentz_slope, y0)
    p1 = (x0 + thickness_um * dxdz, y0 + thickness_um * dydz)
    return p0, p1


def true_center_position(
    x0: float, y0: float, dxdz: float, dydz: float, thickness_um: float
) -> tuple[float, float]:
    """The particle's own trajectory (x0 + z*dxdz, y0 + z*dydz), evaluated at
    the slab's mid-thickness plane z = thickness/2.

    This is the "true" hit position used as the residual reference: it is
    the track's own path, not the Lorentz-drifted charge-collection segment
    from `charge_endpoints` (Lorentz drift shifts where charge ends up, not
    where the particle went).
    """
    z_mid = thickness_um / 2.0
    return x0 + z_mid * dxdz, y0 + z_mid * dydz


def path_length_through_slab(dxdz: float, dydz: float, thickness_um: float) -> float:
    return thickness_um * math.sqrt(1.0 + dxdz**2 + dydz**2)


def deposited_charge(dxdz: float, dydz: float, thickness_um: float, charge_per_um: float) -> float:
    """Total charge deposited by a track, proportional to its 3D path
    length through the slab (constant dE/dx)."""
    return charge_per_um * path_length_through_slab(dxdz, dydz, thickness_um)


def _interior_crossings(a0: float, a1: float, pitch: float) -> list[float]:
    """t in (0, 1) where the segment a0->a1 crosses a grid line spaced by pitch."""
    if a0 == a1:
        return []
    lo, hi = (a0, a1) if a0 < a1 else (a1, a0)
    k_lo = math.floor(lo / pitch) + 1
    k_hi = math.ceil(hi / pitch) - 1
    if k_hi < k_lo:
        return []
    da = a1 - a0
    return [
        t
        for k in range(k_lo, k_hi + 1)
        if 0.0 < (t := (k * pitch - a0) / da) < 1.0
    ]


def segment_pixel_fractions(
    p0: tuple[float, float],
    p1: tuple[float, float],
    pitch_x_um: float,
    pitch_y_um: float,
    n_pixels_x: int,
    n_pixels_y: int,
) -> dict[tuple[int, int], float]:
    """Intersect the segment p0->p1 against a regular pixel grid.

    Returns {(ix, iy): fraction} where fraction is the portion (0-1] of the
    segment's arc length that falls inside that pixel. Fractions sum to 1.0
    unless part of the segment falls outside the grid (that charge is lost,
    as it would be off the physical sensor).
    """
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0

    if dx == 0.0 and dy == 0.0:
        ix, iy = math.floor(x0 / pitch_x_um), math.floor(y0 / pitch_y_um)
        if 0 <= ix < n_pixels_x and 0 <= iy < n_pixels_y:
            return {(ix, iy): 1.0}
        return {}

    ts = {0.0, 1.0}
    ts.update(_interior_crossings(x0, x1, pitch_x_um))
    ts.update(_interior_crossings(y0, y1, pitch_y_um))
    breakpoints = sorted(ts)

    fractions: dict[tuple[int, int], float] = defaultdict(float)
    for t_a, t_b in zip(breakpoints[:-1], breakpoints[1:]):
        span = t_b - t_a
        if span <= 0.0:
            continue
        t_mid = (t_a + t_b) / 2.0
        x_mid, y_mid = x0 + dx * t_mid, y0 + dy * t_mid
        ix, iy = math.floor(x_mid / pitch_x_um), math.floor(y_mid / pitch_y_um)
        if 0 <= ix < n_pixels_x and 0 <= iy < n_pixels_y:
            fractions[(ix, iy)] += span

    return dict(fractions)
