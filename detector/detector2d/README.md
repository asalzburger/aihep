# detector2d

Geometry and trajectory-intersection primitives for a 2D (x, y only) tracking
detector. No pandas/IO here on purpose — this package is the pure-math layer
that `simulator/tracksim2d` builds events on top of.

In 2D, a 3D "plane" detector becomes a straight line segment, and a 3D barrel
"cylinder" becomes a circle. A charged particle in a constant field pointing
out of the plane (`Bz`) moves on a circular arc; with no field (or a neutral
particle) it moves on a straight line — the `R -> infinity` limit of the same
arc.

## Project layout

| file | contents |
|---|---|
| `detector2d/geometry.py` | `LineLayer`, `CircleLayer` (the two detector-layer shapes) and `Trajectory` (a particle's path: `radius=None` is straight, a finite signed radius is a circular arc). |
| `detector2d/intersect.py` | `intersect(trajectory, layer)` / `first_intersection(...)`: the four straight/arc x line/circle intersection cases, returning `Hit(s, x, y, local_coord)`. |
| `detector2d/field.py` | `signed_radius(pt, charge, bz, k)`: optional helper converting physical `(pt, charge, Bz)` into the signed radius `Trajectory` wants; skip it if you already know the radius (e.g. one fit from a picture). |

## Core model

- **`Trajectory(x0, y0, phi0, radius=None)`** — starts at `(x0, y0)` heading at
  angle `phi0`. `radius=None`/`inf` is a straight track. A finite signed
  `radius` is a circular arc: positive curls left (counter-clockwise) as the
  particle moves forward, negative curls right — this is exactly `1/kappa`
  for the usual signed curvature `kappa = q*Bz/pt`.
  - `position(s)` / `direction_at(s)`: closed-form point/heading at arc
    length `s` (`s >= 0` is forward).
  - `center`: the arc's center, `(x0 - R*sin(phi0), y0 + R*cos(phi0))`.
- **`LineLayer(layer_id, p1, p2, pitch=None)`** — a straight layer, the
  segment `p1 -> p2`.
- **`CircleLayer(layer_id, center, radius, pitch=None)`** — a circular layer.
  `pitch` on either layer type is an optional segmentation cell size (used by
  `clustering/tracker`, not by this package).
- **`intersect(trajectory, layer) -> list[Hit]`** — every intersection with
  `s > 0`, sorted by increasing `s`. `Hit.local_coord` is distance-from-`p1`
  for a `LineLayer`, or angle-around-center (radians) for a `CircleLayer`.
  `first_intersection(...)` returns just the earliest one, or `None`.

A circular trajectory is inverted back to an arc length via the smallest
positive `s` found within one extra turn either side of the direct angle —
adequate for tracks crossing a detector stack without looping more than
~1.5 times before reaching a layer (see the `test_arc_beyond_quarter_turn_is_still_found`
test for the wraparound case this specifically guards against).

## Setup

```bash
cd detector/detector2d
python3 -m venv .venv
.venv/bin/pip install -e . -r requirements.txt
```

## Using it as a library

```python
from detector2d import CircleLayer, LineLayer, Trajectory, first_intersection, signed_radius

layer = LineLayer(layer_id=0, p1=(10.0, -50.0), p2=(10.0, 50.0))

straight = Trajectory(x0=0.0, y0=0.0, phi0=0.0)
hit = first_intersection(straight, layer)  # Hit(s=10.0, x=10.0, y=0.0, local_coord=50.0)

radius = signed_radius(pt=2.0, charge=+1, bz=1.5)  # meters, if you're in physical units
bent = Trajectory(x0=0.0, y0=0.0, phi0=0.0, radius=radius)
hit = first_intersection(bent, layer)
```

## Tests

```bash
.venv/bin/python -m pytest tests/
```

Covers line-line, line-circle (miss/tangent/two-point), arc-line and
arc-circle intersection (including a closed-form 3-4-5 circle-circle check
and the branch-wraparound edge case above), plus straight-vs-`radius=inf`
equivalence.
