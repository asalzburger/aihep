# detector2d

Geometry and trajectory-intersection primitives for a 2D (x, y only) particle
detector. No pandas/IO here on purpose — this package is the pure-math layer
that `simulator/tracksim2d` builds events on top of.

In 2D, a 3D "plane" detector becomes a straight line segment, and a 3D barrel
"cylinder" becomes a circle. A charged particle in a constant field pointing
out of the plane (`Bz`) moves on a circular arc; with no field (or a neutral
particle) it moves on a straight line — the `R -> infinity` limit of the same
arc. When the field is *not* constant but changes with radius (as in a real
detector), the path becomes a chain of such arcs — see "Piecewise fields"
below.

## Project layout

| file | contents |
|---|---|
| `detector2d/geometry.py` | `LineLayer`, `CircleLayer` (the two detector-layer shapes) and `Trajectory` (a particle's path: `radius=None` is straight, a finite signed radius is a circular arc). |
| `detector2d/intersect.py` | `intersect(trajectory, layer)` / `first_intersection(...)`: the four straight/arc x line/circle intersection cases, returning `Hit(s, x, y, local_coord)`. |
| `detector2d/field.py` | `signed_radius(pt, charge, bz, k)`: optional helper converting physical `(pt, charge, Bz)` into the signed radius `Trajectory` wants; skip it if you already know the radius (e.g. one fit from a picture). Plus `FieldRegions`, a piecewise-radial field map -- see "Piecewise fields" below. |
| `detector2d/propagate.py` | `propagate(...)` / `SegmentedTrajectory`: follow a particle through a `FieldRegions` map, producing one arc per region, behind the same interface a single `Trajectory` has. `intersect_path(...)` is the matching generalization of `intersect`. |
| `detector2d/barrel.py` | `build_barrel_circle`/`build_barrel_modules`: build one cylindrical (barrel) layer either as a bare `CircleLayer`, or as a ring of identical, tilted, overlapping `LineLayer` sensor modules (all sharing one `layer_id`) -- see "Building a barrel layer" below. |
| `detector2d/calorimeter.py` | `CaloRing`/`build_calo_stack`: a circular sampling layer that also knows its own azimuthal *cell* structure, and radial stacks of them -- see "Calorimeter layers" below. |
| `detector2d/polygon.py` | `build_polygon`/`build_polygon_triplet_station`/`build_muon_system`: polygonal (octagonal) chamber stations, each side a triplet of closely spaced planes -- see "Muon stations" below. |
| `detector2d/config.py` | Declarative detector-layout description: parse a plain dict (e.g. loaded from YAML by a downstream package) into the flat layer list above -- `build_layers_from_raw` handles the tracker (`layers:` or `detector:`) plus the optional `calorimeter:` and `muon:` blocks -- see "Describing a detector layout" below. |

## Core model

- **`Trajectory(x0, y0, phi0, radius=None)`** — starts at `(x0, y0)` heading at
  angle `phi0`. `radius=None`/`inf` is a straight track. A finite signed
  `radius` is a circular arc: positive curls left (counter-clockwise) as the
  particle moves forward, negative curls right — this is exactly `1/kappa`
  for the usual signed curvature `kappa = q*Bz/pt`.
  - `position(s)` / `direction_at(s)`: closed-form point/heading at arc
    length `s` (`s >= 0` is forward).
  - `center`: the arc's center, `(x0 - R*sin(phi0), y0 + R*cos(phi0))`.
- **`LineLayer(layer_id, p1, p2, pitch=None, system="tracker")`** — a straight
  layer, the segment `p1 -> p2`.
- **`CircleLayer(layer_id, center, radius, pitch=None, system="tracker")`** — a
  circular layer. `pitch` on either layer type is an optional segmentation cell
  size (used by `clustering/tracker`, not by this package). `system` names the
  subsystem (`"tracker"`, `"ecal"`, `"hcal"`, `"muon"`) so simulation code can
  dispatch on it rather than on `layer_id` ranges; it defaults to `"tracker"`,
  so pre-existing layouts are unaffected.
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

## Calorimeter layers

A tracking layer only has to say *a particle crossed here*; a calorimeter
layer has to say *and it put this much energy into cell 37*, so the layer
itself owns its azimuthal binning. `CaloRing` is a **subclass** of
`CircleLayer` — every intersection routine keeps working on it unchanged —
that adds `n_phi` cells, a `phi_offset`, and a radial `thickness`:

```python
from detector2d.calorimeter import build_calo_stack

ecal = build_calo_stack(
    layer_id_base=100, r_inner=210.0, n_layers=3, thickness=30.0,
    n_phi=256, system="ecal", phi_stagger=[0.0, 0.5, 0.0],
)
ring = ecal[0]
ring.cell_index(0.31)        # which cell an azimuth falls in (wraps at +-pi)
ring.cell_center_phi(20)     # ...and back
ring.cell_edges(20)          # (low, high), unwrapped so low < high always
```

Ring `i` sits at the middle of its own slab (`r_inner + (i+0.5)*thickness`),
so the stack's full radial extent is exactly
`[r_inner, r_inner + n_layers*thickness]`, and `pitch` is derived from
`n_phi` rather than given independently.

**`phi_stagger` is the interesting parameter.** It shifts layer `i` by that
fraction of a cell. Staggering the middle layer of an ECAL by half a cell
means its cell *boundaries* fall at the *centers* of the layers above and
below it — so a shower landing exactly on a boundary in layers 0 and 2 (the
worst case, its energy split evenly between two cells, with no way to say
which side it favoured) lands mid-cell in layer 1, which resolves it. See
`test_calorimeter.py` for that property stated as a test.

## Muon stations

A muon spectrometer isn't a cylinder: it's a small number of large flat
chambers arranged as a polygon around the beam line. `polygon.py` builds
them at three levels:

```python
from detector2d.polygon import build_muon_system

muon = build_muon_system(
    layer_id_base=300, apothem_inner=520.0, station_spacing=100.0,
    n_stations=3, n_planes=3, n_sides=8, triplet_gap=8.0,
)   # 3 stations x 8 sides x 3 planes = 72 LineLayers
```

Each *side* is a **triplet** — three closely spaced parallel planes, so one
crossing gives three measurements and a local direction, which is what lets a
station stand on its own as a track-segment finder. All 8 sides of one plane
share a `layer_id` (the same convention `barrel.py` uses for a barrel layer's
modules); `layer_id`s are allocated `base + station*10 + plane`, so the
station a hit belongs to is readable straight off it.

Size is the **apothem** — origin to the *middle of a side*, not to a vertex.
A ray from the origin therefore crosses each plane exactly once, giving a
muon 9 hits. The one exception is a track aimed exactly at a vertex, which
clips both sides meeting there and gives that plane two hits: the polygon's
version of the module overlap `barrel.py` builds in deliberately, and real
(chambers do overlap at a station's corners).

## Piecewise fields, and multi-segment trajectories

A real detector's field is not one constant: it is strong inside the solenoid
(the tracker), essentially zero in the calorimeters outside it, and *reversed*
in the muon system where the flux returns. `FieldRegions` describes that as
concentric shells of constant `bz`, and `propagate` turns it into a chain of
arcs:

```python
from detector2d.field import FieldRegion, FieldRegions
from detector2d.propagate import propagate, intersect_path

field = FieldRegions(regions=(
    FieldRegion(r_max=210.0, bz=+2.0),   # tracker: strong
    FieldRegion(r_max=480.0, bz= 0.0),   # calorimeters: none
    FieldRegion(r_max=None,  bz=-1.0),   # muon system: half, reversed
), k=0.2998)

path = propagate(0.0, 0.0, phi0=0.3, charge=+1, pt=200.0, field=field,
                 world_radius=800.0, max_path_length=4000.0)
path.position(s), path.direction_at(s)   # same interface as Trajectory
hits = intersect_path(path, layer)       # same interface as intersect
```

`SegmentedTrajectory` deliberately presents the *same* interface a single
`Trajectory` does, so drawing and hit-finding code consumes either without
branching, and the single-region case is a strict pass-through — one segment,
geometrically identical to the plain `Trajectory` it would have been.

Consecutive arcs join with continuous position **and** direction; each
segment's arc is only valid inside its own region, so `intersect_segmented`
discards crossings past a segment's end (they would put hits where the
particle never went) and reports the survivors' `s` as a *global* arc length.

**`max_path_length` is not optional in practice.** A particle whose bend
radius is too small to leave the tracker circles forever; that cap is what
makes propagation terminate. `world_radius` stops it once it has left the
detector.

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

Two further **optional and independent** blocks describe what sits outside the
tracker, and simply append to whichever form built it:

```yaml
calorimeter:                 # a mapping of stack name -> CaloStackConfig;
  ecal:                      # the key doubles as the layers' `system` tag
    {layer_id_base: 100, r_inner: 210.0, n_layers: 3, thickness: 30.0,
     n_phi: 256, phi_stagger: [0.0, 0.5, 0.0]}
  hcal:
    {layer_id_base: 200, r_inner: 300.0, n_layers: 2, thickness: 80.0, n_phi: 64}

muon:
  {layer_id_base: 300, apothem_inner: 520.0, station_spacing: 100.0,
   n_stations: 3, n_planes: 3, n_sides: 8, triplet_gap: 8.0}
```

`build_layers_from_raw` takes the *top-level* config dict (not just the
`detector:` sub-dict), raises if both `detector:` and `layers:` are present,
and returns `[]` if none of the keys is. `parse_layer`,
`parse_detector_config`, `build_detector_layers`,
`parse_calorimeter_config`/`build_calorimeter_layers`, and
`parse_muon_config`/`build_muon_layers` are the individual steps, exposed
separately if a caller wants just one of them.

The `field:` block is parsed by `parse_field_regions` into a `FieldRegions`,
in either the piecewise form or the original scalar one:

```yaml
field:
  k: 0.2998
  regions:                     # inside-out; omit the last r_max for "beyond"
    - {r_max: 210.0, bz:  2.0}
    - {r_max: 480.0, bz:  0.0}
    - {bz: -1.0}
# ...or, still: field: {bz: 1.0}   -- one constant field everywhere
```

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
`test_calorimeter.py` covers cell indexing (including wraparound) and the
half-cell stagger property; `test_polygon.py` covers the apothem/vertex
geometry, the triplet spacing, the exactly-one-crossing-per-plane invariant
and the vertex double-crossing exception; `test_propagate.py` covers region
lookup, single-region pass-through, continuity across boundaries, the
reversed-field curvature flip, the stopping conditions, and global arc
lengths from `intersect_segmented`.
