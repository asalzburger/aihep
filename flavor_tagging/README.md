# flavor_tagging

A first, from-scratch flavor-tagging exercise: simulate a mix of light jets
and b-jets with [`detectorsim2d`](../simulator/detectorsim2d)'s `jets` gun
mode, smear them into tracks/clusters with
[`detectorreco2d`](../reconstruction/detectorreco2d), and check the result
against the three observables a real tagger would build on -- reference
material for what LEP-era b-tagging actually measured is in
`resources/Z-bb-LEP.pdf`.

A **b-jet** here stands in for a jet containing a B hadron: instead of every
particle starting at the primary vertex, the whole jet's particles share one
vertex displaced 10-90 units along the jet axis (an invisible flight before
the B hadron decays), and -- the three knobs this exercise is actually
about, all opt-in additions to `detectorsim2d`'s gun (see
`ParticleGunConfig.b_jet_track_boost`/`b_jet_pt_boost`/`jet_muon_fraction`/
`b_jet_muon_fraction`) --

- **~20% more tracks** than an equivalent light jet (`b_jet_track_boost: 0.2`)
- **~15% higher pt** (`b_jet_pt_boost: 0.15`)
- **a muon far more often**: 15% of b-jets vs. 2% of light jets
  (`b_jet_muon_fraction`/`jet_muon_fraction`) -- standing in for a
  semileptonic B decay, historically one of the most powerful single
  b-tagging handles (soft-lepton tagging).

The displaced vertex alone already gives b-jet tracks a wider impact
-parameter (`d0`) distribution than light-jet tracks (whose trajectories
genuinely pass through the origin) -- see [`detector2d.geometry.Trajectory.d0`](../detector/detector2d/detector2d/geometry.py)
and the module docstring of `detectorreco2d.reconstruct` for why.

## Project layout

| module | contents |
|---|---|
| `flavor_tagging.pipeline` | `simulate_jets`, `reconstruct_run`, `run_pipeline` (all three in one call), `summarize_jets` (per-jet track/muon counts, the table the validation plots are built from). |
| `flavor_tagging.vis` | `plot_track_d0`, `plot_cluster_energy`, `plot_track_multiplicity`, `plot_muon_multiplicity`, and `make_validation_plots` (all four at once). |
| `flavor_tagging.cli` | Three subcommands mirroring the pipeline's three stages -- see below. |
| `configs/jets_bjets.yaml` | `detectorsim2d.config.SimConfig`: the full detector from `simulator/detectorsim2d/configs/jets.yaml`, gun in `jets` mode with the four b-jet knobs above turned on. |
| `configs/reco.yaml` | `detectorreco2d.config.RecoConfig`: track/cluster resolution tuned to this gun's pt range (50-300). |

## Setup

```bash
cd flavor_tagging
python3 -m venv .venv
.venv/bin/pip install \
  -e ../detector/detector2d -e ../simulator/detectorsim2d \
  -e ../reconstruction/detectorreco2d -e ../viz/style -e . \
  -r requirements.txt
```

(`detector2d`, `detectorsim2d`, `detectorreco2d`, `viz_style` are sibling
path packages, not on PyPI, so they're installed explicitly.)

## Running the pipeline

```bash
python -m flavor_tagging.cli simulate    --output-dir out/sim   --format arrow --seed 42
python -m flavor_tagging.cli reconstruct --sim-dir out/sim --output-dir out/reco --format arrow
python -m flavor_tagging.cli validate    --reco-dir out/reco --output-dir out/plots
```

Each subcommand defaults to `configs/jets_bjets.yaml`/`configs/reco.yaml`
(override with `--config`) and to `arrow` for the intermediate tables. Or,
as a library, in one call:

```python
from flavor_tagging.pipeline import run_pipeline
from flavor_tagging.vis import make_validation_plots

particles, hits, deposits, tracks, clusters = run_pipeline(seed=42)
make_validation_plots(tracks, clusters, "out/plots")
```

## The validation plots

`make_validation_plots` (or `cli validate`) writes four PNGs:

| plot | shows |
|---|---|
| `track_d0.png` | Reconstructed track `d0` (impact parameter), light jets vs. b-jets -- the b-jet distribution is visibly wider, from the displaced vertex. |
| `cluster_energy.png` | Truth vs. reconstructed cluster energy, overlaid -- the resolution smearing itself. |
| `track_multiplicity.png` | Tracks per jet, light vs. b-jet -- the `b_jet_track_boost` effect (b-jets average ~20% more). |
| `muon_multiplicity.png` | Muons per jet, light vs. b-jet -- the `jet_muon_fraction`/`b_jet_muon_fraction` split (2% vs. 15% of jets contain one). |

All four use the truth `is_b_jet` label carried end to end from the gun
(`detectorsim2d.edm.PARTICLES_COLUMNS`) through reconstruction
(`detectorreco2d.edm`) to `summarize_jets` -- they validate that the
pipeline reproduces its own configured knobs, not a trained tagger's
output. Building an actual tagger (e.g. a classifier on
`d0`/track-count/muon-count) is the natural next exercise on top of this.

## Tests

```bash
.venv/bin/python -m pytest tests/
```

Covers the pipeline (b-jets and light jets both appear, reconstruction
carries jet truth through, `run_pipeline` is reproducible under an explicit
seed, `summarize_jets`'s per-jet track/muon counting) and the plots
(smoke tests: each one runs end to end on synthetic tracks/clusters and
writes a real file).

**Note:** run pytest (and any `python -c "import flavor_tagging"` sanity
check) from *inside* this directory, not the repo root -- a bare `import
flavor_tagging` run with the repo root as the working directory resolves to
this directory itself as an implicit namespace package (Python treats any
directory as one once it's on `sys.path`) rather than the installed
package, which is missing its actual contents. Every other package in this
repo avoids the collision by not sharing its top-level directory name with
anything importable from the root; this one does, purely because
`flavor_tagging/` (the repo folder) and `flavor_tagging` (the package) are
the same name -- same as `clustering/tracker`'s `tracker/` vs. `tracker`,
harmless as long as you `cd` into the package directory first, which the
commands above already do.
