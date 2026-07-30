# splitting

Given a [`sensor`](../sensor) run whose clusters may merge more than one
truth particle's hits into one connected component (routine once
`multi.n_particles > 1` with a non-trivial opening angle, e.g.
`configs/p123.yaml`), decide how to split a cluster back apart into
per-particle sub-clusters -- and make the splitting *algorithm* itself a
pluggable, swappable piece, so future (non-truth, reconstruction-only)
splitters can be dropped in and compared against this one.

This first splitter is an **oracle**: it looks at the ground-truth
`contributions` table (which particle deposited how much charge into which
pixel -- see `sensor`'s README) and assigns each hit pixel to whichever
particle contributed the most charge to it. It can't be used on real data
(no `contributions` table exists there), but it's the proof of concept and
the reference/baseline other splitters should be validated against.

## Project layout

| module | contents |
|---|---|
| `splitting.edm` | Table schemas (`hits`, `clusters`, `truth`, `contributions`) -- mirrors `sensor.edm` exactly, so a split run directory is a drop-in replacement for a `sensor` run directory. |
| `splitting.io` | Read/write the four tables as CSV or Apache Arrow (serialization lives in [`clustering/utils`](../utils), shared with `sensor`/`tracker`). |
| `splitting.base` | `Splitter`: the pluggable interface every splitting algorithm implements. |
| `splitting.pipeline` | `apply_splitter(splitter, hits, clusters, contributions)`: turns any `Splitter`'s per-pixel decision into a self-consistent (hits, clusters) pair -- renumbers `cluster_id` and recomputes cluster aggregates. Shared by every splitter; this is the part you *don't* reimplement per algorithm. |
| `splitting.truth_splitter` | `TruthSplitter`: the ground-truth oracle splitter described above. |
| `splitting.registry` | `SPLITTERS`: name -> `Splitter` class, the exchange point the CLI's `--splitter` flag reads from. |

`splitting.cli` wires these into the `run` command described below.

## Setup

```bash
cd clustering/splitting
python3 -m venv .venv
.venv/bin/pip install -e ../utils -e . -r requirements.txt
```

(`clustering_utils` is a sibling path package, not on PyPI, so it's
installed explicitly rather than listed in `requirements.txt`.)

## Splitting a run

```bash
.venv/bin/python -m splitting.cli run \
  --splitter truth \
  --input-dir resources/p123 \
  --output-dir out/p123_truth_split \
  --format arrow
```

Reads `hits`/`clusters`/`truth`/`contributions` from `--input-dir` (any
`sensor`-shaped run directory) and writes a complete new run directory:
`hits`/`clusters` reflect the split, `truth`/`contributions` are copied
through unchanged (splitting never touches ground truth). Because the
output has exactly `sensor`'s schema, `sensor.cli visualize`/`analyse` (or
`sensor.vis.plot_event` with `--zoom`) work on it unchanged -- handy for
eyeballing what a splitter actually did to a specific event.

`resources/p123` above isn't checked in -- it's generated on demand:
clusters from 1-, 2-, and 3-particle events, produced via
`configs/p123.yaml` in `sensor`. `pytest -m p123_resources` (re)creates it
before running the tests that need it (see Tests below); to create it
yourself for the command above, from `clustering/sensor`:

```bash
.venv/bin/python -m sensor.cli run \
  --config configs/p123.yaml --n-events 1000 --seed 123 \
  --output-dir ../splitting/resources/p123 --format arrow
```

`--splitter` selects from `splitting.registry.SPLITTERS` (currently just
`truth`); `--format` is `csv` or `arrow`.

## Writing a new splitter

Implement `base.Splitter`:

```python
from splitting.base import Splitter
import pandas as pd

class MySplitter(Splitter):
    name = "my-splitter"

    def split_key(self, hits: pd.DataFrame, clusters: pd.DataFrame, contributions: pd.DataFrame) -> pd.Series:
        """One label per row of `hits` (same length/order). Pixels in the
        same original cluster_id that get the same label stay together;
        different labels split them apart. Labels are only ever compared
        within one (event_id, cluster_id) group."""
        ...
```

then register it:

```python
# splitting/registry.py
from .my_splitter import MySplitter
SPLITTERS[MySplitter.name] = MySplitter
```

and it's immediately usable as `--splitter my-splitter` -- `pipeline.apply_splitter`
(renumbering `cluster_id` densely per event, recomputing `n_pixels`/
`charge_sum`/both centroids/spans) is unchanged and shared by every
splitter, so a new algorithm is *only* the per-pixel labeling decision.

## The `truth` splitter

`TruthSplitter.split_key` assigns each hit pixel the `particle_id` that
deposited the most charge into it (`truth_splitter.dominant_particle_per_pixel`),
falling back to a sentinel (`NO_CONTRIBUTION_KEY = -1`) for a pixel with no
truth contribution at all -- a purely noise-driven hit, or an occasional
noise pixel that happens to land next to a real cluster and gets glued on
by connectivity (this does happen in `resources/p123`: 15 of its 999
truth-bearing clusters have exactly this). Those pixels are split off into
their own tiny cluster rather than merged into any particle's split.

**This can't reach perfect purity, even as an oracle.** In `configs/p123.yaml`
(and similar wide-opening-angle multi-particle configs), every particle in
an event starts from the *same* vertex, so their charge-collection paths
genuinely overlap in individual pixels near the vertex -- about 26% of
`resources/p123`'s truth-contributed pixels have charge from more than one
particle. Splitting can pick the dominant owner per pixel, but it can't
subdivide a single pixel, so some residual cross-particle contamination is
a hard floor of *any* pixel-grained splitter, truth-based or not. Concretely,
on `resources/p123`: the mean "dominant particle's share of its cluster's
charge" goes from 0.64 (unsplit) to 0.86 (split), and the count of exactly
single-particle clusters goes from 357 (only the already-single-particle
events) to 933 -- a real improvement, not a complete fix. See
`tests/test_resources_p123.py`.

## Tests

```bash
.venv/bin/python -m pytest tests/                    # everything, incl. regenerating resources/p123
.venv/bin/python -m pytest tests/ -m "not p123_resources"  # skip the ones that need sensor's venv
```

- `test_pipeline.py` -- the generic renumbering/aggregation machinery
  (`apply_splitter`), using a trivial test-double splitter: splitting a
  merged cluster by an arbitrary key, a constant key being a no-op, and
  rejecting a splitter that returns the wrong number of labels.
- `test_truth_splitter.py` -- `TruthSplitter` on small, hand-built
  scenarios: picks the larger contributor per pixel, cleanly separates a
  cluster shared by two particles plus a glued-on noise pixel, and the
  no-contribution sentinel.
- `test_resources_p123.py` -- integration test against real (not
  hand-built) data: hits/charge are conserved, cluster count never
  decreases, and purity improves substantially without claiming perfection
  (see above). Tagged `@pytest.mark.p123_resources`: the `p123_resources`
  fixture in `conftest.py` (re)generates `resources/p123` by shelling out
  to `sensor.cli run --config configs/p123.yaml` once per test session
  before these run, so they're never skipped for a missing/stale
  directory -- only if `sensor`'s own venv isn't set up, in which case the
  fixture itself skips with a clear reason.
