"""Infer the Denby event's particle parameters straight out of the reference
SVG and write them to resources/denby_event.csv.

The 4 blue dashed tracks are circular arcs (confirmed by an essentially-exact
circle fit, rms residual <= 0.06px) sharing one common vertex -- the green
marker dot, cross-checked against the tracks' own pairwise circle-circle
intersections in `denby_svg.derive_vertex` (see README.md). Per track we
recover (x0, y0, phi0, charge, radius) directly in the SVG's own pixel
coordinate system: "assume a constant field, stay in SVG coordinates" means
storing the already-resolved signed radius rather than inventing a
pt/Bz decomposition that isn't observable from a single static picture.

`charge` is a *convention*, not a measurement: +1 for a left/CCW curl
(radius > 0), -1 for right/CW (radius < 0) -- flipping the assumed field
direction would flip every charge and leave the picture identical.

Run from this directory (`tracking/denby/python/`) with the project venv:

    ../.venv/bin/python fit_event.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from denby_svg import derive_vertex, fit_circle_to_svg_path, parse_event_tracks, sample_cubic_path

from detectorsim2d.edm import PARTICLES_COLUMNS

RESOURCES = Path(__file__).resolve().parent.parent / "resources"
EVENT_SVG = RESOURCES / "denby_detector_event.svg"
EVENT_CSV = RESOURCES / "denby_event.csv"


def _resolve_direction_and_radius(
    vertex: tuple[float, float], far_point: tuple[float, float], center: tuple[float, float], r_geom: float
) -> tuple[float, float]:
    """Of the two tangent directions at `vertex` (one per curl sign), pick the
    one whose *forward* (increasing arc length) sweep reaches `far_point`
    the short way around -- the only physically sensible choice for a track
    drawn as a single, non-looping arc from the vertex out to the edge of
    the figure."""
    x0, y0 = vertex
    cx, cy = center
    theta_vertex = math.atan2(y0 - cy, x0 - cx)
    theta_far = math.atan2(far_point[1] - cy, far_point[0] - cx)

    forward_sweep_if_ccw = (theta_far - theta_vertex) % (2 * math.pi)
    forward_sweep_if_cw = (theta_vertex - theta_far) % (2 * math.pi)
    radius = r_geom if forward_sweep_if_ccw <= forward_sweep_if_cw else -r_geom

    phi0 = math.atan2((x0 - cx) / radius, (cy - y0) / radius)
    return phi0, radius


def fit_track(d: str, vertex: tuple[float, float]) -> dict:
    points = sample_cubic_path(d)
    cx, cy, r_geom = fit_circle_to_svg_path(d)
    far_index = int(np.argmax(np.hypot(points[:, 0] - vertex[0], points[:, 1] - vertex[1])))
    far_point = tuple(points[far_index])

    phi0, radius = _resolve_direction_and_radius(vertex, far_point, (cx, cy), r_geom)
    charge = 1.0 if radius > 0 else -1.0
    return dict(x0=vertex[0], y0=vertex[1], phi0=phi0, charge=charge, radius=radius)


def fit_event() -> pd.DataFrame:
    svg_text = open(EVENT_SVG).read()
    vertex = derive_vertex(svg_text)
    rows = [
        dict(event_id=0, particle_id=particle_id, **fit_track(d, vertex))
        for particle_id, d in enumerate(parse_event_tracks(svg_text))
    ]
    return pd.DataFrame(rows, columns=PARTICLES_COLUMNS)


def main() -> None:
    particles = fit_event()
    particles.to_csv(EVENT_CSV, index=False)
    print(particles.to_string(index=False))
    print(f"wrote {EVENT_CSV}")


if __name__ == "__main__":
    main()
