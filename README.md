# aihep

Didactic HEP tracking/clustering/ML exercises, built up as a set of small,
single-purpose Python packages rather than one monolith -- each package
does one thing, depends explicitly on its siblings (never reaches past its
own directory), and is independently pip-installable, runnable, and tested.
Every package's own README is the source of truth for its API and CLI;
this file is the map of how they all fit together, and should stay current
as packages are added or their dependencies change.

## Dependency graph

| package | imports (local, in-repo) |
|---|---|
| `detector2d` | -- |
| `clustering_utils` | -- |
| `viz_style` | -- |
| `detectorsim2d` | `detector2d`, `viz_style` |
| `sensor` | `clustering_utils`, `viz_style` |
| `tracker` | `detector2d`, `clustering_utils` |
| `splitting` | `clustering_utils` (consumes `sensor`-*shaped* run directories, but never imports `sensor`) |
| `multiplicity` | `clustering_utils`, `viz_style` (trains on `sensor`-shaped run directories, same as `splitting`) |
| `graphs` | `detectorsim2d` (transitively `detector2d`), `viz_style` |
| `hopfield_tracking` | `viz_style` (plain `(x, y)` hit points, no `detector2d` dependency) |
| `denby` (scripts) | `detector2d`, `detectorsim2d`, `hopfield_tracking` |
| `soho` (scripts) | -- |
| `detectorreco2d` | `detector2d`, `detectorsim2d` |
| `flavor_tagging` | `detector2d`, `detectorsim2d`, `detectorreco2d`, `viz_style` |

Nothing here is circular, and each package's `pyproject.toml`/
`requirements.txt` deliberately omits its local (sibling) dependencies --
they're installed explicitly as path installs (see [Install](#install)) so
`pip install .` for one package never tries to fetch a sibling from PyPI.

## Modules

| module | what it does |
|---|---|
| [`detector/detector2d`](detector/detector2d) | 2D detector geometry and trajectory-intersection primitives (line/circle layers, straight/arc trajectories, segmented calorimeter rings, polygonal muon stations, and piecewise-radial field maps with the multi-arc propagation they imply). No pandas/IO -- the pure-math layer everything else builds on. |
| [`simulator/detectorsim2d`](simulator/detectorsim2d) | Particle event generation, CSV/Arrow IO, and visualization for a full 2D detector -- tracker, EM and hadronic calorimeters, muon system -- on top of `detector2d`. Particles have species, and the three interaction classes (EM / hadronic / muon) stop in three different places. |
| [`clustering/utils`](clustering/utils) | Shared CSV/Apache Arrow table read/write, factored out once from `sensor`/`tracker`'s originally-duplicated IO. |
| [`viz/style`](viz/style) | Centralized matplotlib theming (`Theme`: `print`, the default, matching each package's original look; `present`, no titles anywhere and no axes for detector/sensor spatial displays) and a canonical color palette (Okabe-Ito categorical cycle + named semantic colors), consumed via an optional `theme=` parameter and CLI `--style` flag across `sensor`/`detectorsim2d`/`graphs`/`hopfield_tracking`/`multiplicity`. |
| [`clustering/sensor`](clustering/sensor) | Simulates charged particles crossing a pixelated silicon slab: digitization, connected-component clustering, and analysis/visualization. |
| [`clustering/tracker`](clustering/tracker) | Turns `detectorsim2d` hits into clusters via 1D adjacent-cell grouping along each detector layer -- the tracking-detector analogue of `sensor`'s pixel clustering. |
| [`clustering/splitting`](clustering/splitting) | Splits a `sensor` cluster that merged more than one truth particle's hits back apart, behind a pluggable `Splitter` interface. Ships one splitter (`TruthSplitter`, a ground-truth oracle/baseline). |
| [`clustering/multiplicity`](clustering/multiplicity) | Small MLP that predicts a cluster's particle multiplicity (1/2/3) from its pixel pattern alone -- the question a real (non-truth) splitter needs answered before it can split a cluster. |
| [`clustering/soho`](clustering/soho) | Standalone exercise (no local-package dependencies): clusters the ~536 casualty markers on John Snow's 1854 Soho cholera map a few different ways and checks how close each lands to the real Broad Street pump. |
| [`tracking/graphs`](tracking/graphs) | Builds candidate track graphs (nodes = hits, edges = candidate connections) from `detectorsim2d` hits under a configurable, YAML-driven prescription (fully-connected / regional / explicit rules), with optional ground-truth edge labeling. |
| [`tracking/hopfield_tracking`](tracking/hopfield_tracking) | From-scratch reimplementation of Denby (1988)'s Hopfield-network track finder. Works on plain `(x, y)` hit points -- no `detector2d` dependency. |
| [`tracking/denby`](tracking/denby) | Recreates the Denby (1988) paper's own worked example end to end: harmonizes the detector drawn in the paper's figure, fits the 4-track event back into particle parameters, re-simulates it through `detectorsim2d`, and runs `hopfield_tracking` on it. Script collection, not an installable package. |
| [`reconstruction/detectorreco2d`](reconstruction/detectorreco2d) | Turns `detectorsim2d` truth into tracks/clusters: every charged particle's `(d0, phi0, pt)` and every showering particle's calorimeter energy, each smeared by a Gaussian whose width shrinks with energy (`sigma = a + b / x`). Muons get no special treatment here (charged like anything else); they're excluded from cluster reconstruction instead, since a MIP trail isn't a shower. |
| [`flavor_tagging`](flavor_tagging) | Worked flavor-tagging example on top of `detectorsim2d`'s `jets` gun mode and `detectorreco2d`: b-jets get a displaced-vertex-shifted impact parameter, ~20% more tracks, ~15% higher pt, and a muon far more often than light jets (15% vs. 2%, standing in for semileptonic B decay). Validation plots check the reconstructed pipeline reproduces its own configured knobs. |

## Install

Every package with a `pyproject.toml` is meant to be `pip install -e`'d
(never installed non-editably -- these are exercises you iterate on, not
frozen releases), and lists only its *PyPI* dependencies in that file;
sibling in-repo packages are installed alongside it as explicit path
installs. `clustering/soho` and `tracking/denby` are plain script
collections (no `pyproject.toml`) -- their own `requirements.txt` covers
just their PyPI dependencies.

### Everything, in one command

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
```

The root [`requirements.txt`](requirements.txt) aggregates every package
above via `-e <path>` (each pulling in its own `pyproject.toml`
dependencies) plus `-r <path>/requirements.txt` for the two script
collections -- one `pip` invocation, one dependency resolution, no
duplicated version pins. `clustering/soho`'s `05_animation.py` additionally
needs `ffmpeg` on `PATH` (e.g. `brew install ffmpeg`); everything else only
needs the pip install above.

### Just one module

Each module's own README has a narrower `Setup` section installing only
that module and its direct siblings, e.g. from `clustering/tracker`:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ../../detector/detector2d -e ../utils -e . -r requirements.txt
```

## Tests

Each package's tests are self-contained under its own `tests/` directory
and are run per-package (several packages have same-named test files --
e.g. more than one `test_io.py`/`test_config.py` -- which collide under a
single repo-wide `pytest` collection run without `__init__.py` markers, so
don't try to run them all at once from the root):

```bash
cd clustering/tracker   # or any other package
.venv/bin/python -m pytest tests/
```

(`.venv` above is either that package's own venv per its README, or the
shared root one from [Install](#install) -- both work.)

## Data

`data/` and every package's `out/` are git-ignored -- run outputs and any
downloaded/generated datasets live there, never committed. Regenerate them
by running the relevant package's simulation/pipeline commands (see each
module's README).
