# sensor

A very simple simulation of charged particles crossing a pixelated silicon
slab: particles fly through at some angle (optionally with a constant
Lorentz drift), deposit charge along their path, that charge is digitized
onto a pixel grid, hit pixels are grouped into clusters, and everything is
written out in columnar form (CSV or Apache Arrow) with a matplotlib viewer.

## Project layout

The `sensor` package is split into four submodules:

| submodule | contents |
|---|---|
| `sensor.edm` | **Event data model**: the transient per-event table schemas (`hits`, `clusters`, `truth` column lists) shared by `sim`, `io`, and `vis` — the single source of truth for what a row of each table looks like. |
| `sensor.sim` | Everything that produces an event: detector/particle configuration (`config.py`), track/pixel-grid geometry (`geometry.py`), particle generation (`simulate.py`), digitization — diffusion/noise/threshold (`digitize.py`), and connected-component cluster finding (`clustering.py`). |
| `sensor.io` | Read/write `hits`/`clusters`/`truth` tables as CSV or Apache Arrow (the serialization itself lives in [`clustering/utils`](../utils), shared with `clustering/tracker`). |
| `sensor.vis` | Matplotlib visualization: single-event display and cluster summary plots. |

`sensor.cli` ties these together into the `run`/`visualize` commands
described below; it's not one of the four core submodules, just the entry
point that wires `sim` → `io` → `vis`.

## Setup

```bash
cd clustering/sensor
python3 -m venv .venv
.venv/bin/pip install -e ../utils -r requirements.txt
```

(`clustering_utils` is a sibling path package, not on PyPI, so it's
installed explicitly rather than listed in `requirements.txt`.)

## Run the simulation

```bash
.venv/bin/python -m sensor.cli run \
  --config configs/default.yaml \
  --n-events 100 \
  --output-dir out/ \
  --format arrow \
  --seed 42
```

This writes `out/hits.arrow`, `out/clusters.arrow`, `out/truth.arrow`
(use `--format csv` for `.csv` files instead). All flags are optional:

| flag | default | meaning |
|---|---|---|
| `--config` | none (built-in defaults) | YAML file, see [Configuration](#configuration) |
| `--n-events` | from config (`1`) | overrides `n_events` |
| `--seed` | from config (`null`, i.e. random) | overrides `seed` |
| `--output-dir` | `out` | where the three tables are written |
| `--format` | `csv` | `csv` or `arrow` |

Running with no `--config` at all uses the documented defaults (150 µm
thick, 25×50 µm pixels, 200×200 pixel grid, no Lorentz drift, ±30% uniform
incident angle, 1 particle per event).

## Visualize an event

```bash
.venv/bin/python -m sensor.cli visualize \
  --config configs/default.yaml \
  --output-dir out/ \
  --format arrow \
  --event-id 0 \
  --save event0.png
```

Drop `--save` to open an interactive matplotlib window instead. This reads
the tables straight from `--output-dir`, so it does not re-run the
simulation — `--config` here is only used to know the detector geometry
(pitch, grid size, thickness, Lorentz slope) for drawing the pixel grid and
truth tracks, and should be the same config the run used.

The plot shows: hit pixels colored by charge (yellow = low, red = high),
each cluster's bounding box outlined in red, and the truth charge-collection
track (drift-corrected) for every particle in blue.

Extra flags:

- `--zoom NX NY` — instead of the full sensor, show only an `NX x NY` pixel
  window centered on the event's largest cluster (falls back to the grid
  center if the event has no clusters). Useful once the grid is much bigger
  than a typical cluster (a couple pixels on a 200x200 grid).
- `--grid` — overlay light gray lines at every pixel boundary, in view.
- `--readout-threshold` (default `0.15`) — pixels with charge at or below
  this are treated as not read out: dropped from the display and from
  cluster boxes, mimicking a real front-end's readout threshold. This is a
  visualization-only cut, independent of `digitization.threshold` in the
  simulation config (which controls what's written to `hits` in the first
  place).
- `--digital` — show pixels above the readout threshold as flat on/off
  instead of charge-graded color (no colorbar; a single "hit (on)" swatch
  in the legend instead).

```bash
.venv/bin/python -m sensor.cli visualize \
  --output-dir out/ --format arrow --event-id 0 \
  --zoom 20 30 --grid --save event0_zoomed.png

# binary on/off view with a custom readout threshold
.venv/bin/python -m sensor.cli visualize \
  --output-dir out/ --format arrow --event-id 0 \
  --zoom 20 30 --grid --digital --readout-threshold 0.2 --save event0_digital.png
```

For summary plots (cluster size / charge distributions across many
events), call the plotting function directly:

```python
from sensor.io import read_run
from sensor.vis import plot_cluster_summary

hits, clusters, truth = read_run("out/", "arrow")
fig = plot_cluster_summary(clusters)
fig.savefig("summary.png", dpi=150)
```

## Configuration

All parameters live in a YAML file passed via `--config`; any key you omit
falls back to its default (see `sensor/sim/config.py`).
`configs/default.yaml` documents every field:

```yaml
detector:
  thickness_um: 150.0       # silicon slab thickness
  pitch_x_um: 25.0            # pixel pitch in x
  pitch_y_um: 50.0             # pixel pitch in y
  n_pixels_x: 200                # simulation area, in pixels
  n_pixels_y: 200
  lorentz_slope: 0.0                # constant Lorentz drift (dx/dz-equivalent); 0 = no drift

particle:
  angle_spread: 0.3           # uniform: max |dxdz|,|dydz|; gauss: sigma, around nominal
  angle_distribution: uniform  # "uniform" or "gauss"
  nominal_dxdz: 0.0
  nominal_dydz: 0.0
  charge_per_um: 0.006666666666666667  # 1/150: perpendicular track deposits charge = 1.0

multi:
  n_particles: 1               # only consulted when > 1
  opening_angle_x: 0.05
  opening_angle_y: 0.05
  opening_distribution: uniform  # "uniform", "gauss", or "exponential"

digitization:
  diffusion_sigma_um: 0.0    # 0 disables Gaussian blur of the charge grid
  noise_sigma: 0.0             # 0 disables per-pixel additive noise
  threshold: 0.0                 # pixels with charge <= threshold are not hits

n_events: 1
seed: null
cluster_connectivity: 8   # 4 or 8 neighbor connectivity for cluster finding
```

Notes on how these interact:

- **Lorentz drift**: shifts where charge is *collected* in x, proportional
  to how far it has to drift through the slab. It does not change the
  particle's own trajectory, so it elongates clusters in x independently of
  the incident angle. `lorentz_slope=0` is a plain straight-line track.
- **Incident angle**: each event first draws one nominal `(dxdz, dydz)` from
  `particle.angle_spread`/`angle_distribution` (uniform ±spread, or Gaussian
  around `nominal_dxdz`/`nominal_dydz`). With `n_particles == 1` that's the
  particle's direction.
- **Multi-particle events**: when `multi.n_particles > 1`, every particle in
  the event shares the same vertex and starts from the same nominal
  direction above, then gets its own offset drawn from
  `multi.opening_angle_x/y` and `opening_distribution` (uniform, Gaussian,
  or exponential) — simulating a small shower/jet of particles from one
  point rather than `n_particles` unrelated single-particle events.
- **Digitization** (diffusion/noise/threshold) is off by default, giving a
  purely geometric, deterministic charge pattern. Turn on `diffusion_sigma_um`
  and/or `noise_sigma` for a more realistic (stochastic) detector response.

## Output schema (event data model)

The schemas below are defined once, in `sensor/edm.py`, and reused
by `sim` (produces them), `io` (persists them), and `vis` (reads them).
Three tables, joined by `event_id` (and `truth`/`hits` also by
`particle_id`/`cluster_id` respectively):

**`hits`** — one row per hit pixel:
`event_id, ix, iy, x_center_um, y_center_um, charge, cluster_id`

**`clusters`** — one row per cluster:
`event_id, cluster_id, n_pixels, charge_sum, x_centroid_um, y_centroid_um, x_span_pixels, y_span_pixels`

**`truth`** — one row per simulated particle:
`event_id, particle_id, x0_um, y0_um, dxdz, dydz, charge_deposited, path_length_um`

## Using it as a library

```python
from sensor.sim import SimConfig
from sensor.cli import run_simulation

config = SimConfig()
config.n_events = 50
config.detector.lorentz_slope = 0.2
config.multi.n_particles = 3

hits, clusters, truth = run_simulation(config)
```

## Tests

```bash
.venv/bin/python -m pytest tests/
```

Covers the core line/pixel-grid intersection geometry (path-length
conservation, drift-shifted endpoints, clipping at the grid edge) and
end-to-end sanity checks (a perpendicular track hits exactly one pixel,
Lorentz drift elongates clusters in x, CSV/Arrow round-trip losslessly).
