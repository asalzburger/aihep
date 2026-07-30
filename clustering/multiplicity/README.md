# multiplicity

A small MLP that predicts a cluster's **particle multiplicity** (1, 2, or 3
particles) from its pixel pattern alone -- the question a real (non-truth)
[`splitting`](../splitting) splitter would need answered before it can even
decide *whether* to split a cluster, let alone how.

Trains on any [`sensor`](../sensor)-shaped run directory
(`configs/p1.yaml`/`p2.yaml`/`p3.yaml`/`p123.yaml`-style output: single- or
mixed-multiplicity clusters), and validates -- ROC curves, confusion
matrix -- on a genuinely independent one.

## The model

- **Input**: a fixed-size `(N, M)` pixel-charge matrix per cluster. `N`/`M`
  are *derived from the training data*, not hardcoded: `dataset.compute_matrix_shape`
  takes the largest x/y pixel span among all "real" clusters (see below)
  across every `--input-dir` passed to `train`, and adds 1 in each
  direction -- enough room for the largest cluster seen, with at least one
  pixel of margin. A cluster's own bounding box is centered within that
  fixed canvas (not pinned to a corner), so the network sees a consistent,
  position-independent representation regardless of cluster size or where
  it happened to sit on the sensor.
- **Architecture**: `Flatten -> Linear(N*M, 32) -> ReLU -> Linear(32, 32) -> ReLU -> Linear(32, 1) -> Sigmoid`.
  Two 32-node hidden layers, one output node.
- **Output encoding**: ordinal regression, not a 3-way softmax -- particle
  counts have a natural order (1 < 2 < 3), so a single sigmoid score in
  `[0, 1]` is trained against an equally-spaced target (1 particle -> 0.0,
  2 -> 0.5, 3 -> 1.0, `model.encode_label`) with MSE loss, and decoded back
  into a class via 3 equal-width bins: `[0, 0.33)` -> 1, `[0.33, 0.66)` ->
  2, `[0.66, 1]` -> 3 (`model.decode_score`).
- **What counts as a training example**: one row per *real* cluster -- a
  cluster with at least one contributing truth particle (joining `hits`
  against `contributions`, same idea as `sensor.analysis.cluster_purity`).
  Pure-noise clusters (all of `p1`/`p2`/`p3`'s single- or few-pixel
  digitization-noise clusters -- the large majority of clusters in those
  directories) aren't a 1/2/3-particle example at all and are excluded.
  `n_particles` = the number of distinct particles that deposited any
  charge in the cluster, which is what `p1`/`p2`/`p3`/`p123` are
  constructed to test: in all of them, an event's particles share one
  vertex and (with a non-trivial opening angle) end up merged into a
  single connected-component cluster, so "particles per cluster" and
  "particles per event" coincide.

## Setup

```bash
cd clustering/multiplicity
python3 -m venv .venv
.venv/bin/pip install -e ../utils -e ../../viz/style -e . -r requirements.txt
```

(`clustering_utils` and `viz_style` are sibling path packages, not on
PyPI, so they're installed explicitly rather than listed in
`requirements.txt`. `torch` is listed and pulls in Apple Silicon MPS
support automatically on macOS -- no separate plugin needed.)

## Training

```bash
.venv/bin/python -m multiplicity.cli train \
  --input-dir ../sensor/p1 ../sensor/p2 ../sensor/p3 \
  --format arrow \
  --epochs 50 \
  --output model.pt
```

`--input-dir` takes one or more directories and pools every cluster from
all of them into one dataset -- this is what makes the same `train`
command work equally well on `p1`+`p2`+`p3` (three single-multiplicity
directories) or a single `p123` directory (already-mixed multiplicity), or
any combination.

| flag | default | meaning |
|---|---|---|
| `--input-dir` | *required* | one or more `sensor`-shaped run directories |
| `--format` | `arrow` | `csv` or `arrow` |
| `--epochs` | `50` | training epochs |
| `--batch-size` | `256` | |
| `--val-fraction` | `0.15` | held out (stratified) for in-training validation/monitoring only -- not the independent test set `evaluate` uses |
| `--lr` | `1e-3` | Adam learning rate |
| `--device` | `auto` | `auto` picks MPS on Apple Silicon (e.g. a Mac Studio M2 Ultra) if available, else CPU; can be forced to `cpu`/`mps` |
| `--seed` | `0` | train/val split + shuffling |
| `--output` | `model.pt` | checkpoint path (state dict + the fixed input matrix shape) |

This network is tiny (a few thousand parameters, `N*M` is typically in the
tens) and the datasets are at most a few hundred thousand clusters, so
training is fast on either CPU or MPS.

## Evaluating on an independent test set

```bash
.venv/bin/python -m multiplicity.cli evaluate \
  --model model.pt \
  --input-dir ../sensor/p123 \
  --format arrow \
  --save-dir plots/
```

Loads the checkpoint (including its fixed matrix shape, so the independent
set's clusters are embedded identically to training even if their own
largest span is smaller or larger), predicts, and reports:

- **accuracy** and a **3x3 confusion matrix** (true vs. predicted particle
  count), printed and saved as `plots/confusion_matrix.png`.
- **one-vs-rest ROC curves**, one per class, saved as `plots/roc.png`. The
  ranking score per class is derived from the single ordinal output:
  class 3 ranks directly on the raw score, class 1 on the inverted score,
  and class 2 (bounded on both sides) on proximity to its own target value
  (`evaluate._class_score`) -- the natural generalization of "higher
  score = more particles" to a middle class.

Use a directory the model never trained on for this -- e.g. train on
`p1`+`p2`+`p3`, evaluate on `p123` (or a fresh generation with a different
`--seed` through `sensor.cli run`).

## Project layout

| module | contents |
|---|---|
| `multiplicity.io` | Reads `hits`/`clusters`/`truth`/`contributions` from a `sensor`-shaped run directory (serialization lives in [`clustering/utils`](../utils)). |
| `multiplicity.dataset` | `compute_matrix_shape`, `build_dataset`: turns one or more run directories into `(matrices, n_particles)` arrays. |
| `multiplicity.model` | `MultiplicityMLP`, `encode_label`/`decode_score`. |
| `multiplicity.train` | `train_model`, `save_checkpoint`/`load_checkpoint`. |
| `multiplicity.evaluate` | `evaluate_model`, `plot_roc`, `plot_confusion_matrix`. |
| `multiplicity.device` | MPS/CPU device selection. |

`multiplicity.cli` wires these into the `train`/`evaluate` commands above.

## Tests

```bash
.venv/bin/python -m pytest tests/
```

Uses small synthetic run directories (a cluster of `k` adjacent pixels, one
particle each, for `k` in {1, 2, 3} -- built in `tests/conftest.py`) rather
than the real multi-hundred-thousand-cluster `p1`/`p2`/`p3` data, so the
suite runs in seconds: matrix-shape/labeling/centering in `dataset.py`,
label encode/decode round-trips and bin boundaries plus the MLP's shape in
`model.py`, and an end-to-end train -> save/load -> evaluate pass.
