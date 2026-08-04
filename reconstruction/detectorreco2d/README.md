# detectorreco2d

Turns `detectorsim2d` truth into what a real reconstruction would actually
measure: every charged particle's `(d0, phi0, pt)` smeared into a **track**,
every showering particle's total calorimeter deposit smeared into a
**cluster** -- both by an independent Gaussian whose width shrinks with
energy, `sigma = a + b / x` (see [`detectorreco2d.reconstruct.resolution`](detectorreco2d/reconstruct.py)).
Muons get no special treatment in track reconstruction (they're charged like
anything else); what *does* treat them specially is cluster reconstruction,
which excludes them -- a muon's calorimeter footprint is a thin MIP trail,
not a shower, so "muons act like tracks" in practice means their only
reconstructed object is a track.

## Project layout

| module | contents |
|---|---|
| `detectorreco2d.edm` | `TRACKS_COLUMNS`, `CLUSTERS_COLUMNS` -- see the module docstring for what each column means. |
| `detectorreco2d.config` | `RecoConfig` (`track: TrackResolution`, `cluster: ClusterResolution`, `seed`), each a `Resolution(a, b)` pair; `load_config` for YAML. |
| `detectorreco2d.reconstruct` | `resolution(a, b, x)`, `reconstruct_tracks`, `reconstruct_clusters`, and the `reconstruct` convenience wrapper that calls both. |
| `detectorreco2d.io` | Read/write `tracks`/`clusters` tables as CSV or Apache Arrow, mirroring `detectorsim2d.io`. |
| `detectorreco2d.cli` | `python -m detectorreco2d.cli run --sim-dir ... --config ... --output-dir ...` -- reconstructs a run previously written by `detectorsim2d.cli run`. |

## Setup

```bash
cd reconstruction/detectorreco2d
python3 -m venv .venv
.venv/bin/pip install -e ../../detector/detector2d -e ../../simulator/detectorsim2d -e . -r requirements.txt
```

(`detector2d` and `detectorsim2d` are sibling path packages, not on PyPI, so
they're installed explicitly.)

## Using it as a library

```python
from detectorsim2d.simulate import simulate_events
from detectorreco2d.config import load_config
from detectorreco2d.reconstruct import reconstruct

particles, hits, deposits = simulate_events(sim_config)   # from detectorsim2d
reco_config = load_config("configs/default.yaml")
tracks, clusters = reconstruct(particles, deposits, reco_config)
```

## The resolution law

Every smeared quantity uses the same shape, `sigma(x) = a + b / x`, where
`x` is that track's own `pt` (for `d0`, `phi0`, and `pt` itself) or that
cluster's own truth `energy`: a higher-energy object gets measured more
precisely, asymptoting to the floor `a` as `x -> infinity`, dominated by
`b / x` at low `x`. `a = b = 0` (the dataclass default) means no smearing
at all -- reconstructed values equal truth exactly, which is what the test
suite's baseline (`RecoConfig()`) relies on.

```yaml
track_resolution:
  d0:   { a: 0.05, b: 2.0 }
  phi0: { a: 0.001, b: 0.02 }
  pt:   { a: 0.05, b: 1.0 }
cluster_resolution:
  energy: { a: 0.05, b: 2.0 }
seed: 7
```

## Output schema (event data model)

Defined once in `detectorreco2d/edm.py`. Joined by `event_id`/`particle_id`
back to the `detectorsim2d` particles table they were reconstructed from:

**`tracks`**: `event_id, particle_id, jet_id, is_b_jet, species, charge, d0_true, d0, phi0_true, phi0, pt_true, pt`

**`clusters`**: `event_id, particle_id, jet_id, is_b_jet, species, energy_true, energy`

`jet_id`/`is_b_jet`/`species` are truth passthrough from the particles
table (`-1`/`False` for a non-jets particle), kept for validating/labeling a
downstream flavor-tagging exercise -- a real reconstruction would not know
any of these for free.

## What's deliberately simplified (first pass)

- **Cluster finding is truth-level.** `reconstruct_clusters` sums a
  particle's own deposits by `particle_id` rather than running a real
  cell-clustering algorithm -- the point of this package is the resolution
  smearing, not clustering (see [`clustering/tracker`](../../clustering/tracker)
  for that exercise on hits). A charged hadron's ECAL MIP trail and HCAL
  shower are summed together into one cluster.
- **No charge misreconstruction, no phi wraparound handling.** `d0`/`phi0`
  are smeared as plain unbounded Gaussians; a `phi0` smear can in principle
  wander past `+-pi` for a very wide resolution, which downstream code
  should keep in mind if it ever compares angles directly rather than via
  `sin`/`cos`.

## Tests

```bash
.venv/bin/python -m pytest tests/
```

Covers the resolution law itself (shrinks with `x`, floors at `a` for
`x <= 0`); track reconstruction (only charged particles, muons included,
exact passthrough at zero resolution, smearing statistics, the pt floor,
jet-truth passthrough and its default for particles tables that predate
jets mode); cluster reconstruction (per-particle deposit summing, muon and
species-free exclusion, the energy floor, empty-deposits edge case); config
loading (defaults, YAML, partial YAML); and CSV/Arrow round-tripping.
