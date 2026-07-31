# sensor

A very simple simulation of charged particles crossing a pixelated silicon
slab: particles fly through at some angle (optionally with a constant
Lorentz drift), deposit charge along their path, that charge is digitized
onto a pixel grid, hit pixels are grouped into clusters, and everything is
written out in columnar form (CSV or Apache Arrow) with a matplotlib viewer.

## Project layout

The `sensor` package is split into five submodules:

| submodule | contents |
|---|---|
| `sensor.edm` | **Event data model**: the transient per-event table schemas (`hits`, `clusters`, `truth` column lists) shared by `sim`, `io`, and `vis` — the single source of truth for what a row of each table looks like. |
| `sensor.sim` | Everything that produces an event: detector/particle configuration (`config.py`), track/pixel-grid geometry (`geometry.py`), particle generation (`simulate.py`), digitization — diffusion/noise/threshold (`digitize.py`), and connected-component cluster finding with a readout threshold and parallel charge-weighted/digital centroids (`clustering.py`). |
| `sensor.io` | Read/write `hits`/`clusters`/`truth`/`contributions` tables as CSV or Apache Arrow (the serialization itself lives in [`clustering/utils`](../utils), shared with `clustering/tracker`). |
| `sensor.analysis` | Reconstruction-quality analysis: traces clusters back to the truth particle(s) that produced them (`cluster_purity`) and computes reconstructed-minus-true position residuals for both centroid definitions, matched exactly by charge contribution where possible. |
| `sensor.vis` | Matplotlib visualization: single-event display (2D pixel grid or a presenter-only 3D slab-crossing illustration), cluster summary plots, and residual plots. |

`sensor.cli` ties these together into the `run`/`visualize`/`analyse`
commands described below; it's not one of the five core submodules, just the
entry point that wires `sim` → `io` → `analysis`/`vis`.

## Setup

```bash
cd clustering/sensor
python3 -m venv .venv
.venv/bin/pip install -e ../utils -e ../../viz/style -e . -r requirements.txt
```

(`clustering_utils` and `viz_style` are sibling path packages, not on
PyPI, so they're installed explicitly rather than listed in
`requirements.txt`.)

## Run the simulation

```bash
.venv/bin/python -m sensor.cli run \
  --config configs/default.yaml \
  --n-events 100 \
  --output-dir out/ \
  --format arrow \
  --seed 42
```

This writes `out/hits.arrow`, `out/clusters.arrow`, `out/truth.arrow`,
`out/contributions.arrow` (use `--format csv` for `.csv` files instead).
All flags are optional:

| flag | default | meaning |
|---|---|---|
| `--config` | none (built-in defaults) | YAML file, see [Configuration](#configuration) |
| `--n-events` | from config (`1`) | overrides `n_events` |
| `--seed` | from config (`null`, i.e. random) | overrides `seed` |
| `--output-dir` | `out` | where the four tables are written |
| `--format` | `csv` | `csv` or `arrow` |
| `--readout-threshold` | from config (`0.0`) | overrides `readout_threshold`: pixels with charge at or below this are dropped *before clustering* (see [Readout threshold](#readout-threshold)) |

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
track (drift-corrected) for every particle in blue, marked with a dot at
its entry point and a filled triangle at its exit point.

Extra flags:

- `--zoom NX NY` — instead of the full sensor, show only an `NX x NY` pixel
  window centered as closely as possible on the event's largest cluster's
  actual pixel footprint (falls back to the grid center if the event has no
  clusters) -- not a charge-weighted centroid, which would skew the window
  off-center toward whichever pixel happens to be charge-heavier. Useful
  once the grid is much bigger than a typical cluster (a couple pixels on a
  200x200 grid).
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
- `--type {digital,charge}...` (default `charge`) — which reconstructed
  centroid(s) to mark per cluster (diamond = charge-weighted, square =
  digital), alongside each truth particle's true position (star). Pass
  both to compare them directly on the same event.
- `--style {print,present}` (default `print`) — `print` is the look above
  (titles, axis labels, legend); `present` drops all of that, for putting
  the figure straight into a slide. See [`viz/style`](../../viz/style).

```bash
.venv/bin/python -m sensor.cli visualize \
  --output-dir out/ --format arrow --event-id 0 \
  --zoom 20 30 --grid --save event0_zoomed.png

# binary on/off view with a custom readout threshold
.venv/bin/python -m sensor.cli visualize \
  --output-dir out/ --format arrow --event-id 0 \
  --zoom 20 30 --grid --digital --readout-threshold 0.2 --save event0_digital.png

# mark both reconstructed centroids next to the true position
.venv/bin/python -m sensor.cli visualize \
  --output-dir out/ --format arrow --event-id 0 \
  --zoom 12 12 --grid --type charge digital --save event0_centroids.png
```

### 3D view

`--view 3d` swaps the pixel-grid display above for a presenter-only
illustration of the event's truth particle(s) as straight lines crossing a
somewhat transparent slab (entry face at z=0 to the readout face at
z=`thickness_um`, marked with a dot at entry and a filled triangle at exit,
same convention as the 2D view) — meant to show at a glance *why* a tilted
track leaves charge across more than one pixel. Every hit pixel is drawn
as a solid 3D block spanning the slab's full thickness (colored by charge,
same colormap as the 2D view, but with no colorbar) — the actual silicon
column the track traversed. Each block's own top face sits on the readout
surface, so together they read as the 2D charge/cluster image projected
onto the top of the slab; a very light pixel-boundary grid covers the rest
of that top surface for context. No cluster boxes, reconstructed
centroids, title/axes, or legend; it always renders this way (there's no
`--style` for it):

```bash
.venv/bin/python -m sensor.cli visualize \
  --output-dir out/ --format arrow --event-id 0 \
  --view 3d --save event0_3d.png
```

`--zoom NX NY` still applies (default `20 20` pixels, centered as closely
as possible on the event's largest cluster's actual pixel footprint, same
as the 2D view -- falling back to the truth particles' mean entry point
only if there are no hits above `--readout-threshold`) — the full sensor
is many times wider/taller than it is thick, so an unzoomed box would
render as a flat pancake with no visible tilt. `--readout-threshold` still
applies too (same convention as the 2D view: pixels with charge at or below
this aren't marked as hit). `--grid`/`--digital`/`--type` are 2D-only and
have no effect on `--view 3d`.

### Readout threshold

There are three, deliberately independent, thresholds in the pipeline:

| threshold | where | effect |
|---|---|---|
| `digitization.threshold` | `run`, inside digitization | pixels with charge at or below this never become a `hits` row at all (models electronic noise floor). |
| `readout_threshold` | `run`, inside `cluster_hits` | pixels with charge at or below this are dropped **before clustering** — they don't count toward `n_pixels`/`charge_sum`/centroids and can't glue two clusters together. This is the one that shapes what actually gets analyzed (cluster size, residuals, …). |
| `visualize`'s `--readout-threshold` | `visualize`, display only | re-applies the same kind of cut on top of the already-written `hits`/`clusters`, purely for what's drawn — it does not change any stored data or re-cluster anything. |

For an analysis that reflects a realistic front-end cut (e.g. for residuals
or cluster-size plots), set `readout_threshold` via config or `run
--readout-threshold` so it's baked into the written `clusters` table, rather
than relying on `visualize`'s cosmetic version.

## Analyse a run (cluster-quality plots)

```bash
.venv/bin/python -m sensor.cli analyse \
  --config configs/default.yaml \
  --output-dir out/ \
  --format arrow \
  --plot residual clustersize \
  --type charge digital \
  --axis x y \
  --save-dir plots/
```

This reads `hits`/`clusters`/`truth`/`contributions` from `--output-dir`
(same as `visualize`) and writes one PNG per requested plot into
`--save-dir` (`residual.png`, `clustersize.png`); drop `--save-dir` to show
them interactively instead.

Flags:

- `--plot {residual,clustersize}...` (default: both) — which plot(s) to
  produce.
  - `residual` — histogram(s) of reconstructed-minus-true position, per
    truth particle (see [Residuals](#residuals) below).
  - `clustersize` — the existing cluster-size/charge-sum distributions
    (`--type`/`--axis` don't affect this one).
- `--type {digital,charge}...` (default: `charge`) — which centroid
  definition(s) the residual plot uses; pass both to overlay them.
- `--axis {x,y}...` (default: both) — which axis/axes the residual plot
  shows (one subplot each).
- `--bins` (default `50`) — histogram bin count for the residual plot.

### Residuals

For each truth particle, its **true position** is its own trajectory
evaluated at the sensor's mid-thickness plane (`x0 + t/2 * dxdz`, `y0 + t/2
* dydz`) — unlike the drift-corrected track drawn in `visualize`, this is
the particle's own path, not where the collected charge ends up, so it's
independent of `lorentz_slope`. The residual is `reconstructed - true`, in
µm, per axis.

The truth particle needs a cluster to compare against — `analyse` (and
`plot_residual`/`compute_residuals` as a library) matches it using the
*exact* charge-contribution link from `contributions` whenever that table
is available (which `analyse` always passes), falling back to
nearest-centroid-by-position otherwise. See
[Tracing hits/clusters to truth particles](#tracing-hitsclusters-to-truth-particles)
for why the exact link matters — nearest-position alone can mis-assign a
truth particle to the wrong cluster once clusters can overlap (e.g.
`multi.n_particles > 1`, as in `configs/p3.yaml`).

Both plotting functions are also usable as a library:

```python
from sensor.io import read_run
from sensor.vis import plot_cluster_summary, plot_residual

hits, clusters, truth, contributions = read_run("out/", "arrow")

fig = plot_cluster_summary(clusters)
fig.savefig("summary.png", dpi=150)

fig = plot_residual(
    clusters, truth, config.detector, types=("charge", "digital"), axis=("x", "y"),
    hits=hits, contributions=contributions,  # exact matching; omit for nearest-position
)
fig.savefig("residuals.png", dpi=150)
```

Or compute residuals directly without plotting, e.g. for your own analysis:

```python
from sensor.analysis import compute_residuals

residuals = compute_residuals(clusters, truth, config.detector, type="charge", hits=hits, contributions=contributions)
residuals["residual_x_um"].std()  # x resolution, charge-weighted centroid
```

## Tracing hits/clusters to truth particles

Every particle's raw per-pixel charge deposit (before diffusion/noise/
threshold) is kept in the `contributions` table — `event_id, particle_id,
ix, iy, charge` — instead of being summed away when particles share an
event. This is what makes it possible to trace a hit pixel, or a whole
cluster, back to the truth particle(s) that produced it, which matters as
soon as `multi.n_particles > 1` and tracks can land in the same or
touching pixels (`configs/p3.yaml`'s 3-particle, wide-opening-angle showers
routinely merge into a single cluster).

`sensor.analysis` provides three functions built on the join between `hits`
(which pixel ended up in which cluster) and `contributions` (which
particle put how much charge into which pixel):

```python
from sensor.analysis import cluster_purity, dominant_particle_per_cluster, dominant_cluster_per_particle

# one row per (event, cluster, particle) that contributed to it, with the
# fraction of that cluster's charge it accounts for
purity = cluster_purity(hits, contributions)

# one row per cluster: its single dominant contributing particle + fraction
# (fraction << 1 flags a merged/overlapping cluster)
dominant_particle_per_cluster(hits, contributions)

# one row per particle: the cluster it contributed the most charge to
# (the exact truth-to-cluster link used by residuals above)
dominant_cluster_per_particle(hits, contributions)
```

Electronic noise (`digitization.noise_sigma`) isn't attributable to any
particle and never appears in `contributions`; diffusion (a linear
operation) is exact in the sum, so `contributions` is exact whenever
`diffusion_sigma_um == 0` (the default) and a close approximation
otherwise.

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
  n_particles: 1               # fixed count, or the range's upper bound in "uniform" mode
  n_particles_mode: fixed      # "fixed" (always n_particles) or "uniform" (n_particles_min..n_particles, inclusive)
  n_particles_min: 1           # range lower bound; only used when n_particles_mode is "uniform"
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
readout_threshold: 0.0    # pixels with charge <= this are dropped before clustering (front-end cut)
```

Notes on how these interact:

- **Lorentz drift**: shifts where charge is *collected* in x, proportional
  to how far it has to drift through the slab. It does not change the
  particle's own trajectory, so it elongates clusters in x independently of
  the incident angle. `lorentz_slope=0` is a plain straight-line track.
- **Incident angle**: each event first draws one nominal `(dxdz, dydz)` from
  `particle.angle_spread`/`angle_distribution` (uniform ±spread, or Gaussian
  around `nominal_dxdz`/`nominal_dydz`). With one particle that's the
  particle's direction.
- **Particle multiplicity**: `n_particles_mode` picks how many particles an
  event gets, drawn fresh per event: `fixed` always uses `n_particles`
  exactly (e.g. `n_particles: 3` → always 3, the previous/only behavior);
  `uniform` draws a whole number uniformly from `n_particles_min` to
  `n_particles` inclusive (e.g. `n_particles_min: 1`, `n_particles: 3` →
  1, 2, or 3 with equal probability — see `configs/p_uniform.yaml`).
- **Multi-particle events**: whenever an event gets more than one particle,
  every particle in it shares the same vertex and starts from the same
  nominal direction above, then gets its own offset drawn from
  `multi.opening_angle_x/y` and `opening_distribution` (uniform, Gaussian,
  or exponential) — simulating a small shower/jet of particles from one
  point rather than unrelated single-particle events.
- **Digitization** (diffusion/noise/threshold) is off by default, giving a
  purely geometric, deterministic charge pattern. Turn on `diffusion_sigma_um`
  and/or `noise_sigma` for a more realistic (stochastic) detector response.
- **`readout_threshold`** is separate from `digitization.threshold` — see
  [Readout threshold](#readout-threshold) — and is what actually shapes the
  `clusters` table (and therefore cluster-size/residual analysis), not just
  what gets displayed.

## Output schema (event data model)

The schemas below are defined once, in `sensor/edm.py`, and reused
by `sim` (produces them), `io` (persists them), and `vis`/`analysis` (read
them). Four tables, joined by `event_id` (and `truth`/`hits`/`contributions`
also by `particle_id`/`cluster_id`/`particle_id` respectively):

**`hits`** — one row per hit pixel:
`event_id, ix, iy, x_center_um, y_center_um, charge, cluster_id`

**`clusters`** — one row per cluster:
`event_id, cluster_id, n_pixels, charge_sum, x_centroid_um, y_centroid_um, x_centroid_digital_um, y_centroid_digital_um, x_span_pixels, y_span_pixels`

`x_centroid_um`/`y_centroid_um` are charge-weighted; `x_centroid_digital_um`/
`y_centroid_digital_um` are the unweighted (digital, on/off) centroid of the
same pixels — computed in parallel so the two reconstruction schemes can be
compared directly (see [Analyse a run](#analyse-a-run-cluster-quality-plots)).

**`truth`** — one row per simulated particle:
`event_id, particle_id, x0_um, y0_um, dxdz, dydz, charge_deposited, path_length_um`

**`contributions`** — one row per (particle, pixel) it deposited charge
into, before diffusion/noise/threshold:
`event_id, particle_id, ix, iy, charge`

The join key for tracing a hit/cluster back to its truth particle(s), see
[Tracing hits/clusters to truth particles](#tracing-hitsclusters-to-truth-particles).

## Using it as a library

```python
from sensor.sim import SimConfig
from sensor.cli import run_simulation

config = SimConfig()
config.n_events = 50
config.detector.lorentz_slope = 0.2
config.multi.n_particles = 3

hits, clusters, truth, contributions = run_simulation(config)
```

## Tests

```bash
.venv/bin/python -m pytest tests/
```

Covers the core line/pixel-grid intersection geometry (path-length
conservation, drift-shifted endpoints, clipping at the grid edge, the
true-position/mid-thickness projection) and end-to-end sanity checks (a
perpendicular track hits exactly one pixel, Lorentz drift elongates
clusters in x, CSV/Arrow round-trip losslessly including `contributions`,
the readout threshold drops low-charge pixels before clustering, digital
vs. charge-weighted centroids agree for single-pixel clusters and diverge
for asymmetric ones, `contributions` sums back to the exact combined charge
grid, and truth-to-cluster matching/residuals fall back to nearest-cluster
without `contributions` but resolve a deliberately misleading
nearest-position case correctly once the exact charge link is supplied).
