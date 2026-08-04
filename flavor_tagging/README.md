# flavor_tagging

A first, from-scratch flavor-tagging exercise: simulate a mix of light jets
and b-jets with [`detectorsim2d`](../simulator/detectorsim2d)'s `jets` gun
mode, smear them into tracks/clusters with
[`detectorreco2d`](../reconstruction/detectorreco2d), check the result
against the observables a real tagger would build on, then actually train
one -- a small MLP that tags a jet as light or b from those observables
alone, evaluated on an independently simulated test set. Reference material
for what LEP-era b-tagging actually measured is in
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
| `flavor_tagging.dataset` | `build_dataset`: turns one reconstructed run's `tracks`/`clusters` into the b-tagger's fixed-size per-jet feature/label arrays. `compute_standardization`/`standardize` for feature scaling. |
| `flavor_tagging.model` | `BTaggerMLP`: a small binary-classification MLP (two 32-node hidden layers, sigmoid output). |
| `flavor_tagging.train` | `train_model`, `save_checkpoint`/`load_checkpoint`. |
| `flavor_tagging.evaluate` | `evaluate_model` (accuracy, confusion matrix, ROC, AUC), `plot_roc`, `plot_confusion_matrix`, `plot_score_distribution`. |
| `flavor_tagging.cli` | Five subcommands mirroring the pipeline's stages -- see below. |
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
path packages, not on PyPI, so they're installed explicitly. `requirements.txt`
additionally pulls in `torch`/`scikit-learn` for the b-tagger.)

## Running the pipeline

Simulate and reconstruct **two independent runs** -- different seeds -- one
to train the tagger on, one to evaluate it on:

```bash
python -m flavor_tagging.cli simulate --output-dir out/sim_train --format arrow --seed 42
python -m flavor_tagging.cli simulate --output-dir out/sim_test  --format arrow --seed 4242

python -m flavor_tagging.cli reconstruct --sim-dir out/sim_train --output-dir out/reco_train --format arrow
python -m flavor_tagging.cli reconstruct --sim-dir out/sim_test  --output-dir out/reco_test  --format arrow

python -m flavor_tagging.cli validate --reco-dir out/reco_train --output-dir out/plots

python -m flavor_tagging.cli train    --reco-dir out/reco_train --output out/model.pt
python -m flavor_tagging.cli evaluate --model out/model.pt --reco-dir out/reco_test --save-dir out/tagger_plots
```

Each subcommand defaults to `configs/jets_bjets.yaml`/`configs/reco.yaml`
(override with `--config`) and to `arrow` for the intermediate tables. Or,
as a library, in one call:

```python
from flavor_tagging.pipeline import run_pipeline
from flavor_tagging.vis import make_validation_plots
from flavor_tagging.train import train_model
from flavor_tagging.evaluate import evaluate_model

_particles, _hits, _deposits, train_tracks, train_clusters = run_pipeline(seed=42)
_particles, _hits, _deposits, test_tracks, test_clusters = run_pipeline(seed=4242)

make_validation_plots(train_tracks, train_clusters, "out/plots")

model, preprocessing, history = train_model(train_tracks, train_clusters)
result = evaluate_model(model, preprocessing, test_tracks, test_clusters)
print(result["accuracy"], result["roc_auc"])
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
output. Turning those same observables into an actual tagger is next.

## The b-tagger

`flavor_tagging.dataset.build_dataset` turns one reconstructed run into one
example per truth jet, with a fixed-size feature vector regardless of that
jet's own track count:

- an **overcommitted array of leading-track `d0`** (see
  `compute_n_track_slots`): the busiest jet in the training set sets the
  slot count (26, in a 300-event run of `configs/jets_bjets.yaml`) -- most
  jets, especially light ones, fill only a fraction of it, sorted
  by descending reconstructed `pt` and zero-padded past however many
  tracks that jet actually had
- **`n_tracks`** -- the `b_jet_track_boost` effect, as a single number
- **`n_muons`** -- the `jet_muon_fraction`/`b_jet_muon_fraction` split
- **`total_cluster_energy`/`n_clusters`** -- the calorimeter-side summary

Features are standardized (`compute_standardization`/`standardize`, fit on
the training set only) before going into `model.BTaggerMLP`, a small
binary-classification MLP (two 32-node hidden layers, sigmoid output,
binary cross-entropy loss) -- `train.train_model` trains it with a
stratified train/validation split for monitoring, and `evaluate.evaluate_model`
checks a trained checkpoint against a run it never saw the training set's
own simulated seed for.

A 300-event training run (`seed=42`) against an independently simulated
300-event test run (`seed=4242`, disjoint from training start to finish --
different `detectorsim2d`/`detectorreco2d` random draws end to end) reaches:

| metric | value |
|---|---|
| accuracy | 0.976 |
| AUC | 0.978 |
| light-jet mistag rate | 0.0% (0/744) |
| b-jet tagging efficiency | 85% (118/139) |

`evaluate.py`'s three plots (`roc.png`, `confusion_matrix.png`, `score.png`)
show this isn't a degenerate classifier: the score distribution has light
jets piled at 0 and most b-jets piled at 1, but with a genuine minority of
b-jets the network scores low on -- exactly the ones without a wide `d0`,
extra tracks, or a muon to give them away, i.e. the ones that are honestly
hard to tell from a light jet with these features.

## Tests

```bash
.venv/bin/python -m pytest tests/
```

Covers the pipeline (b-jets and light jets both appear, reconstruction
carries jet truth through, `run_pipeline` is reproducible under an explicit
seed, `summarize_jets`'s per-jet track/muon counting), the validation plots
(smoke tests: each one runs end to end on synthetic tracks/clusters and
writes a real file), and the b-tagger -- `dataset.build_dataset` (feature
shape/labels, `d0`-slot ordering and zero-padding, truncation when an
evaluation set's busiest jet exceeds a `n_track_slots` fixed from training,
feature standardization), the model (output shape/range), training
(loss decreases, checkpoint round-trips, and -- on a fast, deterministic,
trivially-separable synthetic signal, not the real simulated data above --
that it actually learns and generalizes to an independent dataset), and the
evaluation plots (same smoke-test pattern).

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
