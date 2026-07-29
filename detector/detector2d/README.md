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
| `detector2d/barrel.py` | `build_barrel_circle`/`build_barrel_modules`: build one cylindrical (barrel) layer either as a bare `CircleLayer`, or as a ring of identical, tilted, overlapping `LineLayer` sensor modules (all sharing one `layer_id`) -- see "Building a barrel layer" below. |
| `detector2d/config.py` | Declarative detector-layout description: parse a plain dict (e.g. loaded from YAML by a downstream package) into the flat layer list above -- `build_layers_from_raw` dispatches on a hand-listed `layers:` spec vs. a higher-level `detector:` spec (`DetectorConfig`, expanded via `barrel.py`) -- see "Describing a detector layout" below. |

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

## Building a barrel layer

A real cylindrical tracker layer isn't one continuous circle -- it's a ring
of flat sensor modules, usually tilted and overlapping their neighbors so a
track can cross two adjacent modules and give redundant hits near the
overlap (used for alignment in real detectors). `detector2d/barrel.py`
builds both the plain-circle simplification and this more detailed layer:

```python
from detector2d.barrel import build_barrel_circle, build_barrel_modules

# simplified: one bare circle
layer = build_barrel_circle(layer_id=0, radius=29.0, pitch=0.1)

# detailed: a ring of tilted, overlapping LineLayer modules, all layer_id=0
modules = build_barrel_modules(
    layer_id=0, radius=29.0, half_length=4.0, tilt=0.1745,  # ~10 degrees
    overlap_fraction=0.15, pitch=0.1,
)
```

Each module's proximal edge is anchored exactly on the circle; it extends
`2*half_length` in the tangent direction, tilted by `tilt` (radians) toward
the center. The module count is *derived*, not configured: it's whatever
makes the ring of tilted modules cover the requested `overlap_fraction` of
each module's own angular width with its neighbor (0 tilt/overlap gives
plain edge-to-edge tiling). See `module_reach` and `n_modules_for_overlap`
if you want the derivation directly.

**A curved trajectory can cross a barrel layer more than twice, for two
unrelated reasons -- don't conflate them.** (1) The intended one: two
adjacent modules' overlap, giving two hits close together in both position
and arc length. (2) A pre-existing, physically real property of any
nonzero-curvature `Trajectory` starting exactly at the origin: its own
circular arc passes through the origin, so it always crosses a concentric
circular layer at two points (unless the layer is out of reach) -- generally
far apart in arc length (a first pass, then a second after most of a loop).
A single `CircleLayer` only ever reports the nearer of these two via
`first_intersection`; decomposing a layer into many `LineLayer` modules
(as `build_barrel_modules` does) surfaces *both*, since each module object is
tested independently. If you only want to see the overlap-driven kind, look
at hits close together in arc length (`path_length` in `tracksim2d`'s
`hits` table), not just "more than one hit on this layer_id".

## Describing a detector layout from a config dict

`detector2d/config.py` is the detector-*description* layer: it turns a plain
dict (typically loaded from YAML by a downstream package like `tracksim2d`)
into the flat `LineLayer`/`CircleLayer` list above. It knows nothing about
simulation (particles, fields, IO) -- that's `tracksim2d`'s job; this module
only builds the layout.

Two mutually exclusive forms, both handled by `build_layers_from_raw(raw)`:

```yaml
layers:
  - {kind: line, layer_id: 0, p1: [10.0, -50.0], p2: [10.0, 50.0], pitch: 1.0}
  - {kind: circle, layer_id: 5, center: [0.0, 0.0], radius: 5.0, pitch: 0.5}
```

a flat, hand-listed list of layer specs (`kind: line` needs `p1`/`p2`;
`kind: circle` needs `center`/`radius`; both take an optional `pitch`, the
digitization cell size consumed by `clustering/tracker`, not by this
package) -- or, for a cylindrical (barrel) tracker:

```yaml
detector:
  mode: detailed   # or simplified
  layers:
    - {layer_id: 0, radius: 29.0,  kind: precision}
    - {layer_id: 3, radius: 100.0, kind: outer}
    # ...
  module_types:
    precision: {half_length: 4.0, tilt_deg: 10.0, overlap_fraction: 0.15, pitch: 0.1}
    outer:     {half_length: 8.0, tilt_deg: 8.0,  overlap_fraction: 0.10, pitch: 0.5}
```

a higher-level `DetectorConfig` that expands into the same flat layer list
via `build_detector_layers` (see `simulator/tracksim2d/configs/barrel6.yaml`
for a full working example). `mode: simplified` expands each layer to a bare
`CircleLayer`; `mode: detailed` expands it into a ring of tilted `LineLayer`
modules via `build_barrel_modules` (module size/tilt/overlap looked up per
`kind` in `module_types`; the module count is derived, not configured --
see "Building a barrel layer" above).

`build_layers_from_raw` takes the *top-level* config dict (not just the
`detector:` sub-dict), raises if both `detector:` and `layers:` are present,
and returns `[]` if neither is. `parse_layer`, `parse_detector_config`, and
`build_detector_layers` are the individual steps, exposed separately if a
caller wants just one of them.

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
equivalence. `test_barrel.py` covers the tiling/tilt/overlap math and checks
that a track aimed into a module overlap really does produce two hits.
