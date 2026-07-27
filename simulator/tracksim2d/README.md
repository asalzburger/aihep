# tracksim2d

Particle event generation, CSV/Arrow IO, and visualization for a 2D tracking
detector, built on top of [`detector/detector2d`](../../detector/detector2d)'s
geometry/intersection primitives. Mirrors the shape of
[`clustering/sensor`](../../clustering/sensor): an `edm.py`
schema module, a YAML-driven `config.py`, `simulate.py`, `io.py`, `vis.py`,
and a `cli.py` tying them together.

## Project layout

| module | contents |
|---|---|
| `tracksim2d.edm` | The two per-event table schemas (`particles`, `hits`) shared by `simulate`, `io`, and `vis`. |
| `tracksim2d.config` | `SimConfig`: detector layout (a list of `detector2d` `LineLayer`/`CircleLayer`), `FieldConfig` (constant out-of-plane `Bz`), `ParticleGunConfig` (random particle sampling), loaded from YAML. |
| `tracksim2d.simulate` | `hits_for_particles(particles_df, layers)` propagates *any* particles table through a layout; `simulate_events(config)` additionally samples the particles from a `ParticleGunConfig`. |
| `tracksim2d.io` | Read/write `particles`/`hits` tables as CSV or Apache Arrow. |
| `tracksim2d.vis` | `plot_event(...)` (matplotlib) and `export_svg(...)` (dependency-free raw SVG, exact circular arcs via native `A` path commands). |

`tracksim2d.cli` wires `simulate` -> `io` -> `vis` into `run`/`visualize`
subcommands, the same split as `sensor.cli`.

## Setup

```bash
cd simulator/tracksim2d
python3 -m venv .venv
.venv/bin/pip install -e ../../detector/detector2d -e . -r requirements.txt
```

(`detector2d` is a sibling path package, not on PyPI, so it's installed
explicitly rather than listed as a `tracksim2d` dependency.)

## Run the simulation

```bash
.venv/bin/python -m tracksim2d.cli run \
  --config configs/default.yaml \
  --n-events 100 \
  --output-dir out/ \
  --format arrow \
  --seed 42
```

Writes `out/particles.arrow` and `out/hits.arrow` (`--format csv` for
`.csv`). All flags are optional; with no `--config`, the layout defaults to
an empty layer list (particles, but no hits).

## Visualize an event

```bash
.venv/bin/python -m tracksim2d.cli visualize \
  --config configs/default.yaml --output-dir out/ --format arrow \
  --event-id 0 --save event0.png
```

Drop `--save` for an interactive window. Detector layers are drawn dashed
gray, each particle gets one color from a fixed colorblind-safe palette,
hits are outlined markers, vertices are stars. `--track-length` sets how far
to draw a particle that has no hits (default 100).

For a raw-SVG export instead (e.g. to overlay against a reference figure in
the *same* coordinate system, as `tracking/denby` does):

```python
from tracksim2d.vis import export_svg

export_svg(config.layers, particles, hits, "event0.svg", width=900, height=900)
```

## Configuration

```yaml
layers:
  - {kind: line, layer_id: 0, p1: [10.0, -50.0], p2: [10.0, 50.0], pitch: 1.0}
  - {kind: circle, layer_id: 5, center: [0.0, 0.0], radius: 5.0, pitch: 0.5}

field:
  bz: 1.0          # constant field out of the 2D plane
  k: 0.2998        # R[len] = pt / (k * |q| * bz)

gun:
  n_particles: 3
  vertex_x: 0.0
  vertex_y: 0.0
  vertex_spread_x: 0.0
  vertex_spread_y: 2.0
  phi_min: -0.3
  phi_max: 0.3
  charges: [-1.0, 1.0]
  pt_min: 2.0
  pt_max: 10.0

n_events: 1
seed: null
```

`layers` is a flat list of `detector2d` layer specs (`kind: line` needs
`p1`/`p2`; `kind: circle` needs `center`/`radius`; both take an optional
`pitch`, the digitization cell size consumed by
`clustering/tracker`, not by this package). Each event draws one
nominal direction/charge/`pt` per particle from `gun`, resolves a signed bend
radius via `detector2d.field.signed_radius(pt, charge, field.bz, field.k)`
(`bz=0` or `charge=0` -> a straight track), and keeps the earliest crossing
of every layer as that particle's hit.

If you already know a particle's `(x0, y0, phi0, charge, radius)` — e.g.
fit straight out of a reference picture, as in `tracking/denby` — skip the
gun entirely and call `hits_for_particles(your_particles_df, layers)`
directly.

## Output schema (event data model)

Defined once in `tracksim2d/edm.py`, reused by `simulate`, `io`, `vis`.
Joined by `event_id` (`hits` also by `particle_id`):

**`particles`**: `event_id, particle_id, x0, y0, phi0, charge, radius`
(`radius` NaN/infinite = straight track)

**`hits`**: `event_id, particle_id, layer_id, hit_id, x, y, s_local, path_length`
(`s_local` = position along the layer; `path_length` = arc length along the
trajectory to reach the hit)

## Using it as a library

```python
from tracksim2d.config import SimConfig, ParticleGunConfig, FieldConfig
from detector2d.geometry import LineLayer

config = SimConfig(
    layers=[LineLayer(layer_id=0, p1=(10.0, -50.0), p2=(10.0, 50.0))],
    magnetic_field=FieldConfig(bz=1.0),
    gun=ParticleGunConfig(n_particles=5),
    n_events=20,
)
from tracksim2d.simulate import simulate_events
particles, hits = simulate_events(config)
```

## Tests

```bash
.venv/bin/python -m pytest tests/
```

Covers config load/merge, straight- and curved-particle hit geometry
(including the NaN-radius = straight-track path), seeded reproducibility,
CSV/Arrow round-trip, and the SVG exporter (straight vs. chunked-arc `d`
paths, element counts).
